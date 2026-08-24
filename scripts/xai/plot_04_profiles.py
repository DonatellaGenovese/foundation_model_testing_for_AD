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
that component, minus the global SM median, in units of the observable's global
standard deviation (`phys_scale` from step 04, so the scaling matches the
Wasserstein ranking). Positive means the region sits high in that observable.
A diverging scale is used because the quantity has a natural neutral point at
zero; the midpoint is grey, never a hue.

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
    args = p.parse_args()

    run = args.run_dir
    meta = json.loads((run / "04_profile" / "profile_meta.json").read_text())
    k = int(meta["k"])
    phys_scale = meta["phys_scale"]
    populated = {int(c) for c in meta["populated_components"]}

    Z, y, phys = load_matched_npz(run / "04_profile" / "matched_sm_hh4b.npz")
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
            allv = phys[var][mask_sm]
            allv = allv[np.isfinite(allv)]
            sc = phys_scale.get(var, np.nan)
            if len(vals) and len(allv) and np.isfinite(sc) and sc > 0:
                mat[vi, ki] = (np.median(vals) - np.median(allv)) / sc

    lim = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    norm = diverging_norm(lim)
    lim = float(norm.vmax)

    fig, ax = plt.subplots(figsize=(max(7.5, k * 0.72), 4.4))
    im = ax.imshow(mat, cmap=DIVERGING, norm=norm, aspect="auto")

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
