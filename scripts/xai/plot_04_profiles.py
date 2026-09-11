#!/usr/bin/env python3
"""
Figure for step 2 of Sec. 3.3: the label-free physical profile of each GMM region.

Replaces the overlay produced by step 04 (`physics_per_component.pdf`), which
draws all K components as overlapping histograms in eight panels. At K=12 that is
unreadable — no component can be told from another, so the figure cannot support
the claim it exists to support. Here each region is summarised by where it sits on
each observable relative to the SM population as a whole, which is exactly the
"interpretable, data-driven profile for each region" the method promises.

What a cell shows: the median of that observable among the SM events assigned to
that component, minus the global SM median, in units of the SM's own standard
deviation. Positive means the region sits high in that observable. A diverging
scale is used because the quantity has a natural neutral point at zero, and the
colormap puts white exactly there — so a cell with no deviation and a cell that
could not be computed would look alike, and the latter is hatched instead.

The full distributions remain available as the step 04 overlay, which belongs in
an appendix where a reader can inspect shape rather than location.

Components are identified as C<k> only, and the profile uses no process labels at
any stage — that is what makes the characterisation transferable to data.

Usage:
    python scripts/xai/plot_04_profiles.py \\
        --run-dir /eos/.../xai_paper/vcreg_d256_seed3_smnorm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_XAI = Path(__file__).resolve().parent
_ROOT = _XAI.parents[1]
sys.path.insert(0, str(_XAI))
sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.constants import PHYSICS_LABELS, PHYSICS_VARS, SM_INDICES
from common.physics import load_matched_npz
from common.projection import build_sm_pca, check_gmm_dims, project
from common.style import DIVERGING, OI, diverging_norm

C_SIGNAL = OI["signal"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <run-dir>/04_profile/plots/component_profiles.pdf")
    p.add_argument("--annotate-above", type=float, default=0.35,
                   help="Label cells whose |value| exceeds this (selective direct labels)")
    p.add_argument(
        "--pca-dim", type=float, default=0.0,
        help="Project onto this many principal components before the GMM assignment. "
             "Must match the space --gmm-path was fitted in; same convention as "
             "04_profile_and_rank.py. Requires --pca-embeddings-dir.",
    )
    p.add_argument(
        "--pca-embeddings-dir", type=Path, default=None,
        help="Embeddings the PCA is fitted on (SM train only).",
    )
    p.add_argument("--pca-seed", type=int, default=3)
    p.add_argument(
        "--mark", action="append", default=None, metavar="LABEL:C1,C2",
        help="Mark the components a signal occupies, e.g. --mark 'HH->4b:4,5'. Repeat "
             "for a second signal; each gets its own symbol. The profile itself is a "
             "property of the SM partition and is identical whatever is marked, so "
             "several signals can share one figure. Without this the components listed "
             "in profile_meta.json are marked, which is only correct for one signal.",
    )
    p.add_argument("--min-events", type=int, default=200,
                   help="Components with fewer local SM events are left blank")
    p.add_argument(
        "--min-cell-frac", type=float, default=0.15,
        help="Blank a single cell when fewer than this share of the component's SM "
             "events yield a finite value. Mjj and |deta_jj| need two jets and MT needs "
             "a lepton, and the share that qualifies swings by component: Mjj is filled "
             "by 33 of C1's 18,186 events (0.2%%) against all 29,106 of C2's, and MT by "
             "4.0%% of C3 against 92%% of C2. Those cells are precise --- C1's Mjj "
             "bootstraps to +-0.01 --- but they describe a fraction of a per cent of the "
             "region while sitting in a grid whose other cells describe all of it. "
             "Default 0.15 sits inside the natural gap in the coverage (10.5%% to 21.5%% "
             "for MT, 0.2%% to 55.8%% for Mjj), so the cut is insensitive to its own "
             "value.",
    )
    p.add_argument(
        "--assignments", type=Path, default=None,
        help="Read component assignments from this assignments.npz (step 03) instead of "
             "refitting the PCA and re-running gmm.predict. The projection is refitted "
             "from the SM train embeddings, and passing a --pca-dim or --pca-seed other "
             "than the one the mixture was fitted with applies it in a basis it never "
             "saw, silently. Reusing the stored assignment removes that failure mode and "
             "guarantees the figure partitions the events exactly as the tables do.",
    )
    args = p.parse_args()

    run = args.run_dir
    meta = json.loads((run / "04_profile" / "profile_meta.json").read_text())
    k = int(meta["k"])
    populated = {int(c) for c in meta["populated_components"]}

    Z, y, phys = load_matched_npz(run / "04_profile" / "matched_sm_hh4b.npz")
    if args.assignments:
        stored = np.load(args.assignments)
        # Paired by position, so a different row order would misattribute every event
        # without changing a single count. The labels ride along in both files for
        # exactly this check.
        if not np.array_equal(stored["labels"], y):
            raise SystemExit(
                f"{args.assignments} holds {len(stored['labels']):,} labels that do not "
                f"match the {len(y):,} of the matched array; the two were built from "
                f"different event sets and cannot be paired by position.")
        assign = stored["assignments"]
        print(f"Assignments read from {args.assignments}")
    else:
        gmm = joblib.load(meta["gmm_path"])
        # Only the assignment is projected; the physics profile below is computed on the
        # events themselves, so nothing else depends on --pca-dim.
        pca = None
        if args.pca_dim and args.pca_dim > 0:
            if args.pca_embeddings_dir is None:
                raise SystemExit("--pca-dim requires --pca-embeddings-dir")
            pca = build_sm_pca(args.pca_embeddings_dir, args.pca_dim, seed=args.pca_seed)
        Z_gmm = project(pca, Z)
        check_gmm_dims(gmm, Z_gmm)
        assign = gmm.predict(Z_gmm)
    mask_sm = np.isin(y, SM_INDICES)

    out = args.output or (run / "04_profile" / "plots" / "component_profiles.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Reference median and scale, both from the Standard Model alone, and both over the
    # events where the observable is defined so the two describe the same population.
    #
    # Step 04's `phys_scale` is deliberately NOT used here. It is a standard deviation
    # over the whole matched array, signal included, because it exists to make the
    # Wasserstein ranking comparable across observables and there the signal is half of
    # every comparison. This figure contains no signal at all, so borrowing that scale
    # divides an SM-only numerator by a spread the signal helped set. On n_b-tag that is
    # not a rounding matter: the HH->4b block carries 2.37 tags per event against the
    # SM's 0.52 and inflates the scale by 19%, compressing the whole row. It would also
    # leave the figure dependent on which signal the run happened to carry — the dimuon
    # run gives 0.806 where this one gives 0.968, a 20% swing — although the partition
    # the figure describes is a property of the Standard Model alone, which is what lets
    # one figure serve several signals (see --mark).
    sm_ref, sm_scale = {}, {}
    for var in PHYSICS_VARS:
        allv = phys[var][mask_sm]
        allv = allv[np.isfinite(allv)]
        sm_ref[var] = float(np.median(allv)) if len(allv) else float("nan")
        sm_scale[var] = float(np.std(allv)) if len(allv) else float("nan")

    mat = np.full((len(PHYSICS_VARS), k), np.nan)
    counts = np.zeros(k, dtype=int)
    for ki in range(k):
        m = mask_sm & (assign == ki)
        counts[ki] = int(m.sum())
        if counts[ki] < args.min_events:
            continue
        for vi, var in enumerate(PHYSICS_VARS):
            vals = phys[var][m]
            vals = vals[np.isfinite(vals)]
            # Coverage, not precision, is what disqualifies a cell here: C1's Mjj median
            # is stable to +-0.01 under the bootstrap and still describes 33 of its
            # 18,186 events. See --min-cell-frac.
            if len(vals) < args.min_cell_frac * counts[ki]:
                continue
            ref, sc = sm_ref[var], sm_scale[var]
            if len(vals) and np.isfinite(ref) and np.isfinite(sc) and sc > 0:
                mat[vi, ki] = (np.median(vals) - ref) / sc

    lim = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    norm = diverging_norm(lim)
    lim = float(norm.vmax)

    fig, ax = plt.subplots(figsize=(max(7.5, k * 0.72), 4.4))
    # The colormap puts white on zero by construction, so a blanked cell must not also be
    # white: it would read as "no deviation from the SM" rather than "not computed", and
    # the two say opposite things. Grey plus a hatch separates them and survives a
    # greyscale print.
    cmap = DIVERGING.copy()
    cmap.set_bad("#d8d8d8")
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    for vi, ki in zip(*np.where(~np.isfinite(mat))):
        ax.add_patch(plt.Rectangle((ki - 0.5, vi - 0.5), 1, 1, fill=False,
                                   hatch="////", edgecolor="#9e9e9e", linewidth=0))

    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"C{i}" for i in range(k)])
    ax.set_yticks(np.arange(len(PHYSICS_VARS)))
    ax.set_yticklabels([PHYSICS_LABELS.get(v, v) for v in PHYSICS_VARS], fontsize=9)
    ax.set_xlabel("GMM component")

    # Selective direct labels: only cells that carry the profile.
    for vi in range(mat.shape[0]):
        for ki in range(mat.shape[1]):
            v = mat[vi, ki]
            if np.isfinite(v) and abs(v) >= args.annotate_above:
                ax.text(ki, vi, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(v) > 0.62 * lim else "0.15")

    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PHYSICS_VARS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # One marker per signal. Components can be shared, so the symbols are stacked
    # vertically rather than drawn on top of one another.
    SYMBOLS = ["$\\bigstar$", "$\\blacktriangle$", "$\\blacksquare$"]
    if args.mark:
        marks = []
        for i, spec in enumerate(args.mark):
            label, comps = spec.split(":", 1)
            marks.append((SYMBOLS[i % len(SYMBOLS)], label,
                          {int(c) for c in comps.split(",") if c.strip() != ""}))
    else:
        marks = [(SYMBOLS[0], "holds flagged signal", populated)]

    for row, (sym, _label, comps) in enumerate(marks):
        for ki in comps:
            ax.annotate(sym, xy=(ki, len(PHYSICS_VARS) - 0.5),
                        xytext=(0, -20 - 13 * row), textcoords="offset points",
                        ha="center", va="top", fontsize=10, color=C_SIGNAL,
                        annotation_clip=False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label("median deviation from SM, in global s.d.", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Physical profile of each latent region (SM events)", fontsize=11)
    legend = "     ".join(f"{sym}  {label}" for sym, label, _ in marks)
    ax.annotate(legend, xy=(0, 0), xycoords="axes fraction",
                xytext=(0, -46 - 13 * (len(marks) - 1)), textcoords="offset points",
                fontsize=8.5, color=C_SIGNAL, va="top")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    for ki in range(k):
        if counts[ki] < args.min_events:
            print(f"  C{ki}: blank (only {counts[ki]} local SM events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
