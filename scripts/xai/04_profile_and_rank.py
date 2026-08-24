#!/usr/bin/env python3
"""
XAI paper pipeline — Step 04: local physics profiles + Wasserstein ranking (F2, T1).

For each populated GMM component, rank the 8 paper observables by how much they
separate the flagged signal from the local all-SM background — the same population the
mixture was fitted on and the components were profiled with, so one reference is used
from beginning to end. The local QCD background, the AE's trained-normal, is reported
as a control; it is undefined in components that contain almost no QCD.
Writes:

  plots/physics_per_component.pdf    (label-free per-component SM physics profile, Sec. 3.3 pt 3)
  plots/<signal>_vs_sm_k{i}.pdf       (F2, populated only)
  plots/ae_mse_<signal>_k{i}.pdf      (side diagnostic, not F2/T1 — local AE-MSE separation)
  plots/<signal>_vs_sm_missed_k{i}.pdf (additional, not F2 — SM vs flagged vs missed)
  wasserstein_rank.csv               (per var: W1/ratio vs all-SM-local AND vs QCD-local)
  wasserstein_rank.tex               (T1: primary all-SM reference + QCD control column)

Inputs (one of, always with --ae-checkpoint):
  A) --matched-npz  embeddings+labels+phys_* (preferred if already built)
  B) --vectorized-dir + --ckpt-path (+ optional --preproc-split-dir) to build matched data

Optional: --assignments-npz from step 03 reuses only its saved ae_threshold
(so the FPR-based cut stays consistent with step 03). Per-event AE scores and
GMM assignments are always recomputed on --matched-npz, since it is a
different, differently-ordered subset of events than step 03's assignments.npz
and the two are not index-aligned.

Usage:
    python scripts/xai/04_profile_and_rank.py \\
        --gmm-path /eos/.../gmm_K12.pkl \\
        --ae-checkpoint /eos/.../ae.ckpt \\
        --matched-npz /eos/.../matched_data.npz \\
        --output-dir /eos/.../xai_paper/profile \\
        [--min-frac 0.05] [--top-m 3]
"""

from __future__ import annotations

import argparse
import re
import csv
import math
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
from scipy.stats import wasserstein_distance

from common.ae_score import compute_ae_mse, flag_anomalies, resolve_ae_threshold
from common.constants import (
    CLASS_NAMES,
    HH4B_LABEL,
    PHYSICS_BINS,
    PHYSICS_LABELS,
    PHYSICS_VARS,
    SIG_LABELS,
    SM_INDICES,
)
from common.style import OI
from common.physics import (
    build_matched_arrays,
    load_matched_npz,
    save_matched_npz,
)
from common.projection import build_sm_pca, check_gmm_dims, project
from common.utils import save_json


def load_encoder(ckpt_path: Path, device: str):
    import collections
    import functools
    import typing

    import omegaconf
    import torch
    from src.models.collide2v_vicreg import COLLIDE2VVCRegLitModule

    torch.serialization.add_safe_globals(
        [
            functools.partial,
            torch.optim.AdamW,
            torch.optim.Adam,
            torch.optim.lr_scheduler.CosineAnnealingLR,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            omegaconf.ListConfig,
            omegaconf.DictConfig,
            omegaconf.dictconfig.DictConfig,
            omegaconf.nodes.AnyNode,
            omegaconf.base.Metadata,
            omegaconf.base.ContainerMetadata,
            collections.defaultdict,
            typing.Any,
            list,
            dict,
            int,
        ]
    )
    model = COLLIDE2VVCRegLitModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)
    return model


def make_encode_fn(model, device: str, batch_size: int = 2048):
    import torch

    @torch.no_grad()
    def encode(X_pp: np.ndarray) -> np.ndarray:
        embs = []
        for i in range(0, len(X_pp), batch_size):
            batch = torch.tensor(X_pp[i : i + batch_size], dtype=torch.float32).to(device)
            emb = model.encoder.get_embeddings(batch)
            embs.append(emb.cpu().numpy())
        return np.concatenate(embs, axis=0)

    return encode


def populated_components(frac: np.ndarray, min_frac: float, top_m: int | None) -> list[int]:
    order = np.argsort(frac)[::-1]
    keep = [int(i) for i in order if frac[i] >= min_frac]
    if top_m is not None:
        keep = keep[:top_m]
    if not keep and len(order):
        keep = [int(order[0])]
    return keep


def plot_physics_per_component(
    phys: dict,
    assign: np.ndarray,
    mask_sm: np.ndarray,
    k: int,
    path: Path,
    min_events: int = 10,
):
    """Label-free characterisation (paper Sec. 3.3 point 3): for each of the 8
    physics observables, overlay the distribution of the SM events in each GMM
    component. Profiles each latent region purely by physics — no SM class
    labels used."""
    cmap = plt.get_cmap("tab20", k)
    n_vars = len(PHYSICS_VARS)
    ncols = 4
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.asarray(axes).reshape(-1)

    for vi, var in enumerate(PHYSICS_VARS):
        ax = axes[vi]
        vals = phys[var]
        for ki in range(k):
            mask = mask_sm & (assign == ki) & np.isfinite(vals)
            if mask.sum() < min_events:
                continue
            ax.hist(vals[mask], bins=PHYSICS_BINS[var], density=True,
                    histtype="step", lw=1.2, color=cmap(ki), alpha=0.85)
        ax.set_xlabel(PHYSICS_LABELS[var], fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(var, fontsize=9)
        ax.tick_params(labelsize=7)

    for vi in range(n_vars, len(axes)):
        axes[vi].axis("off")
    handles = [plt.Line2D([0], [0], color=cmap(ki), lw=1.5, label=f"C{ki}") for ki in range(k)]
    fig.legend(handles=handles, fontsize=7, ncol=min(k, 5), loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"SM physics distributions per GMM component (K={k})", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_signal_vs_sm(
    phys: dict,
    mask_sig: np.ndarray,
    mask_sm: np.ndarray,
    component: int,
    signal_name: str,
    dominant_sm: str | None,
    path: Path,
):
    n_vars = len(PHYSICS_VARS)
    ncols = 4
    nrows = math.ceil(n_vars / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.0))
    axes = np.asarray(axes).reshape(-1)

    for vi, var in enumerate(PHYSICS_VARS):
        ax = axes[vi]
        vals = phys[var]
        s = vals[mask_sig]
        b = vals[mask_sm]
        s = s[np.isfinite(s)]
        b = b[np.isfinite(b)]
        bins = PHYSICS_BINS[var]
        if len(b):
            ax.hist(b, bins=bins, density=True, histtype="step", lw=1.6, color=OI["sm"], label="local SM")
        if len(s):
            ax.hist(s, bins=bins, density=True, histtype="step", lw=1.6, ls="--", color=OI["signal"], label=signal_name)
        ax.set_xlabel(PHYSICS_LABELS[var], fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if vi == 0:
            ax.legend(fontsize=7)

    for vi in range(n_vars, len(axes)):
        axes[vi].axis("off")

    fig.suptitle(
        f"{signal_name} (AE-flagged) vs local SM — component C{component}"
        + (f" (~{dominant_sm})" if dominant_sm else ""),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ae_mse_local(
    mse: np.ndarray,
    mask_sig: np.ndarray,
    mask_sm: np.ndarray,
    component: int,
    signal_name: str,
    dominant_sm: str | None,
    ae_threshold: float,
    path: Path,
):
    """Side diagnostic (not part of PHYSICS_VARS / Wasserstein ranking): AE-MSE
    of local SM vs AE-flagged signal within this component, against the global
    AE threshold. Shows whether the AE separation is clean or borderline
    locally, not just globally."""
    s = mse[mask_sig]
    b = mse[mask_sm]
    combined = np.concatenate([s, b]) if len(s) or len(b) else np.array([0.0, 1.0])
    bins = np.linspace(combined.min(), combined.max(), 30)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    if len(b):
        ax.hist(b, bins=bins, density=True, histtype="step", lw=1.6, color=OI["sm"], label="local SM")
    if len(s):
        ax.hist(s, bins=bins, density=True, histtype="step", lw=1.6, ls="--", color=OI["signal"], label=signal_name)
    ax.axvline(ae_threshold, color="k", ls="--", lw=1.2, label="AE threshold")
    ax.set_xlabel("AE MSE")
    ax.set_ylabel("Density")
    ax.set_title(
        f"AE-MSE: {signal_name} vs local SM — C{component}"
        + (f" (~{dominant_sm})" if dominant_sm else ""),
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_signal_vs_sm_missed(
    phys: dict,
    mask_flagged: np.ndarray,
    mask_missed: np.ndarray,
    mask_sm: np.ndarray,
    component: int,
    signal_name: str,
    dominant_sm: str | None,
    path: Path,
):
    """Additional plot (not F2): three-way overlay of local SM, AE-flagged, and
    AE-missed signal per physics variable, within this component."""
    n_vars = len(PHYSICS_VARS)
    ncols = 4
    nrows = math.ceil(n_vars / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.0))
    axes = np.asarray(axes).reshape(-1)

    for vi, var in enumerate(PHYSICS_VARS):
        ax = axes[vi]
        vals = phys[var]
        b = vals[mask_sm]
        f = vals[mask_flagged]
        m = vals[mask_missed]
        b = b[np.isfinite(b)]
        f = f[np.isfinite(f)]
        m = m[np.isfinite(m)]
        bins = PHYSICS_BINS[var]
        if len(b):
            ax.hist(b, bins=bins, density=True, histtype="stepfilled", color=OI["light"],
                     alpha=0.6, label=f"SM in C{component}")
        if len(f):
            ax.hist(f, bins=bins, density=True, histtype="step", lw=1.6,
                    color=OI["signal"], label="flagged")
        if len(m):
            ax.hist(m, bins=bins, density=True, histtype="step", lw=1.6, ls="--",
                    color=OI["sm"], label="missed")
        ax.set_xlabel(PHYSICS_LABELS[var], fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if vi == 0:
            ax.legend(fontsize=7)

    for vi in range(n_vars, len(axes)):
        axes[vi].axis("off")

    fig.suptitle(
        f"{signal_name}: SM vs AE-flagged vs AE-missed — component C{component}"
        + (f" (~{dominant_sm})" if dominant_sm else ""),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _w1(a: np.ndarray, b: np.ndarray, scale: float, min_n: int = 5) -> float:
    """Wasserstein-1 distance divided by `scale` (the observable's global std),
    making distances comparable across observables with different units. Uses
    the 1-homogeneity of W1: W1(a/s, b/s) = W1(a, b) / s."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) >= min_n and len(b) >= min_n and np.isfinite(scale) and scale > 0:
        return float(wasserstein_distance(a, b)) / scale
    return float("nan")


def wasserstein_rows(
    phys: dict,
    phys_scale: dict,
    mask_flag: np.ndarray,
    mask_qcd: np.ndarray,
    mask_sm: np.ndarray,
    mask_missed: np.ndarray,
    component: int,
    qcd_poor: bool,
) -> list[dict]:
    """Per variable, standardised Wasserstein-1 of flagged (and missed) signal
    against two reference backgrounds within this component. Each observable is
    scaled by its global standard deviation (`phys_scale`) so distances are
    comparable across observables (raw W1 would rank by unit magnitude).

    THE RANK IS TAKEN AGAINST ALL LOCAL SM, NOT AGAINST LOCAL QCD. The mixture is
    fitted on the Standard Model and every component is characterised by its own SM
    population, so keeping SM as the reference through to the ranking uses one
    background definition from beginning to end. Ranking on QCD instead, as an earlier
    version did, changed reference halfway and was not even defined everywhere: a
    component holding a leptonic signal can contain a single QCD event, leaving W1_qcd
    empty and forcing a silent per-component fallback. Verified not to change the
    outcome — the leading observable is identical under both references in all four
    populated components of the two signals, and the margin over the runner-up stays
    above a factor two.
      - all-SM-local: primary reference, sets the rank.
      - QCD-local: control, the AE's trained-normal; agreement confirms the winner
        does not depend on the reference choice. Undefined where QCD is too sparse.
    ratio = W1(ref, flagged) / W1(ref, missed): >>1 → the variable tracks the AE
    flag decision; ~1 → it only tracks generic signal-vs-background difference.
    (The ratio is scale-invariant; only the cross-variable ranking needs the
    standardisation.)"""
    rows = []
    for var in PHYSICS_VARS:
        sc = phys_scale[var]
        s = phys[var][mask_flag]
        m = phys[var][mask_missed]
        q = phys[var][mask_qcd]
        b = phys[var][mask_sm]

        w_qcd_flag, w_qcd_miss = _w1(s, q, sc), _w1(m, q, sc)
        w_sm_flag, w_sm_miss = _w1(s, b, sc), _w1(m, b, sc)
        ratio_qcd = (w_qcd_flag / w_qcd_miss) if (np.isfinite(w_qcd_flag) and np.isfinite(w_qcd_miss) and w_qcd_miss > 0) else float("nan")
        ratio_sm = (w_sm_flag / w_sm_miss) if (np.isfinite(w_sm_flag) and np.isfinite(w_sm_miss) and w_sm_miss > 0) else float("nan")

        rows.append(
            {
                "component": component,
                "variable": var,
                "W1_qcd": w_qcd_flag, "W1_qcd_missed": w_qcd_miss, "ratio_qcd": ratio_qcd,
                "W1_sm": w_sm_flag, "W1_sm_missed": w_sm_miss, "ratio_sm": ratio_sm,
                "n_flagged": int(np.isfinite(phys[var][mask_flag]).sum()),
                "n_missed": int(np.isfinite(m).sum()),
                "n_qcd": int(np.isfinite(q).sum()),
                "n_sm": int(np.isfinite(b).sum()),
            }
        )

    # One reference throughout: all local SM. `qcd_poor` is still computed and recorded
    # in profile_meta.json — step 06 reads it to pick its residual reference, and it is
    # worth reporting — but it no longer switches the ranking.
    rank_ref = "sm"
    rank_key = "W1_sm"
    finite = [r for r in rows if np.isfinite(r[rank_key])]
    finite.sort(key=lambda r: -r[rank_key])
    rank_map = {r["variable"]: i + 1 for i, r in enumerate(finite)}
    for r in rows:
        r["rank"] = rank_map.get(r["variable"], "")
        r["rank_ref"] = rank_ref
    return rows


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "component", "variable", "rank", "rank_ref",
        "W1_qcd", "W1_qcd_missed", "ratio_qcd",
        "W1_sm", "W1_sm_missed", "ratio_sm",
        "n_flagged", "n_missed", "n_qcd", "n_sm",
    ]
    fmt = {"W1_qcd": "%.6g", "W1_qcd_missed": "%.6g", "ratio_qcd": "%.3g",
           "W1_sm": "%.6g", "W1_sm_missed": "%.6g", "ratio_sm": "%.3g"}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fields}
            for key, spec in fmt.items():
                out[key] = "" if not np.isfinite(r[key]) else spec % r[key]
            w.writerow(out)
    print(f"Saved {path}")


def write_tex(rows: list[dict], path: Path, caption: str):
    """Compact booktabs fragment (paper T1): one block per component, vars sorted
    by rank. All-SM first, since that is what the rank is taken on, then local QCD as
    the control, written as an em dash where the component holds too few QCD events.

    The flagged/missed ratio is deliberately NOT a column. It is defined only where the
    autoencoder missed enough signal to measure a distance, and it is undefined in the
    component holding most of the dimuon signal, where nothing was missed at all. It
    stays in the CSV for anyone who wants it; step 04's own console output reports the
    missed counts that make it computable or not."""
    lines = [
        r"% Auto-generated by scripts/xai/04_profile_and_rank.py",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:wasserstein_rank}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Component & Observable & $W_1$(SM) & $W_1$(QCD) \\",
        r"\midrule",
    ]
    by_comp: dict[int, list] = {}
    for r in rows:
        by_comp.setdefault(int(r["component"]), []).append(r)
    for comp in sorted(by_comp):
        block = [r for r in by_comp[comp] if r["rank"] != ""]
        block.sort(key=lambda r: int(r["rank"]))
        for i, r in enumerate(block):
            comp_cell = f"C{comp}" if i == 0 else ""
            wq = f"{r['W1_qcd']:.3g}" if np.isfinite(r["W1_qcd"]) else "—"
            ws = f"{r['W1_sm']:.3g}" if np.isfinite(r["W1_sm"]) else "—"
            lines.append(f"{comp_cell} & {r['variable']} & {ws} & {wq} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gmm-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--pca-dim", type=float, default=0.0,
        help="Project the embeddings onto this many principal components before the "
             "GMM assignment only — the AE keeps scoring the full embedding, so no "
             "anomaly score or threshold changes. 0 disables; a value below 1 is "
             "read as a variance fraction. Must match the space --gmm-path was "
             "fitted in. Requires --pca-embeddings-dir.",
    )
    p.add_argument(
        "--pca-embeddings-dir", type=Path, default=None,
        help="Embeddings dir whose SM train split defines the PCA basis (same one "
             "the K scan used). Defaults to --embeddings-dir when that is given.",
    )
    p.add_argument("--pca-seed", type=int, default=3,
                   help="Seed used when the stored GMM's PCA was fitted")
    p.add_argument("--ae-checkpoint", type=Path, default=None, help="Required — scores AE-MSE on the matched embeddings")
    p.add_argument("--matched-npz", type=Path, default=None)
    p.add_argument(
        "--assignments-npz", type=Path, default=None,
        help="Optional: reuse ae_threshold saved by step 03 (per-event scores/assignments are always recomputed)",
    )
    p.add_argument("--vectorized-dir", type=Path, default=None)
    p.add_argument("--ckpt-path", type=Path, default=None, help="Encoder ckpt to build matched data")
    p.add_argument("--preproc-split-dir", type=Path, default=None,
                   help="Default: <vectorized>/../../preprocessed/test")
    p.add_argument("--max-per-class", type=int, default=20_000)
    p.add_argument("--signal-label", type=int, default=HH4B_LABEL)
    p.add_argument(
        "--label-components", action="store_true",
        help="Annotate figure titles with each component's dominant SM process. "
             "OFF by default, matching step 03: the characterisation the paper "
             "claims is label-free and model-agnostic. The composition is still "
             "written to profile_meta.json (dominant_sm_per_component) for "
             "internal validation — this flag only affects the figures.",
    )
    p.add_argument(
        "--ae-threshold", type=float, default=None,
        help="If omitted, use the val-calibrated threshold saved in --ae-checkpoint at --fpr "
             "(after --assignments-npz; falls back to self-calibrating on this run's QCD)",
    )
    p.add_argument("--fpr", type=float, default=0.10)
    p.add_argument("--min-frac", type=float, default=0.05, help="Min fraction of flagged signal in component")
    p.add_argument("--top-m", type=int, default=None, help="Keep at most top-m populated components")
    p.add_argument("--save-matched", type=Path, default=None, help="If building matched, also save here")
    p.add_argument(
        "--qcd-labels", type=int, nargs="+", default=[0],
        help="Labels counted as QCD (the AE's trained-normal, primary Wasserstein reference). "
             "Default [0]=QCD-inclusive.",
    )
    p.add_argument(
        "--min-qcd", type=int, default=50,
        help="Min QCD events in a component to use QCD as primary reference; below this the "
             "region is QCD-poor and falls back to the all-SM reference (declared in output).",
    )
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    gmm = joblib.load(args.gmm_path)
    k = int(gmm.n_components)

    # ── Load or build matched (z, y, phys) ───────────────────────────────────
    if args.matched_npz is not None:
        Z, y, phys = load_matched_npz(args.matched_npz)
        print(f"Loaded matched: {Z.shape}, labels={sorted(set(y.tolist()))}")
    else:
        if args.vectorized_dir is None or args.ckpt_path is None:
            raise ValueError("Provide --matched-npz OR (--vectorized-dir AND --ckpt-path)")
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        preproc = args.preproc_split_dir
        if preproc is None:
            preproc = args.vectorized_dir.parent.parent / "preprocessed" / "test"
        encoder = load_encoder(args.ckpt_path, device)
        encode = make_encode_fn(encoder, device)
        classes = list(SM_INDICES) + [args.signal_label]
        Z, y, phys = build_matched_arrays(
            args.vectorized_dir,
            preproc,
            classes,
            encode,
            max_per_class=args.max_per_class,
        )
        if args.save_matched:
            save_matched_npz(args.save_matched, Z, y, phys)

    # ── AE flags + GMM assignment ────────────────────────────────────────────
    # ae-checkpoint is always required: per-event MSE is always scored fresh on
    # Z (matched-npz isn't index-aligned with step 03's assignments.npz, so
    # per-event scores/flags/assignments can't be reused — see module docstring).
    if args.ae_checkpoint is None:
        raise ValueError("--ae-checkpoint required")
    mse = compute_ae_mse(args.ae_checkpoint, Z)

    ae_thr, thr_src = args.ae_threshold, "cli"
    if ae_thr is None and args.assignments_npz is not None:
        ad = np.load(args.assignments_npz)
        if "ae_threshold" in ad.files:
            ae_thr, thr_src = float(ad["ae_threshold"][0]), "assignments_npz"
    if ae_thr is None:
        ae_thr, thr_src = resolve_ae_threshold(
            mse, y, None, ae_ckpt_path=args.ae_checkpoint, bg_label=0, fpr=args.fpr
        )

    anomalous = flag_anomalies(mse, ae_thr)
    # The AE scored the full embedding above; only the assignment is projected, so
    # the flag set is identical with and without --pca-dim and the two runs differ
    # solely in how the flagged events are partitioned.
    pca = None
    if args.pca_dim and args.pca_dim > 0:
        pca_emb = args.pca_embeddings_dir or getattr(args, "embeddings_dir", None)
        if pca_emb is None:
            raise ValueError("--pca-dim requires --pca-embeddings-dir")
        pca = build_sm_pca(pca_emb, args.pca_dim, seed=args.pca_seed)
    Z_gmm = project(pca, Z)
    check_gmm_dims(gmm, Z_gmm)
    assign = gmm.predict(Z_gmm)
    print(f"AE threshold={ae_thr:.6f} ({thr_src})")

    sig = args.signal_label
    sig_name = SIG_LABELS.get(sig, CLASS_NAMES.get(sig, str(sig)))
    mask_sig_all = y == sig
    mask_flag = mask_sig_all & anomalous
    mask_missed = mask_sig_all & ~anomalous
    mask_sm = np.isin(y, SM_INDICES)
    mask_qcd = np.isin(y, args.qcd_labels)

    n_flag = int(mask_flag.sum())
    n_miss = int(mask_missed.sum())
    print(f"{sig_name} flagged: {n_flag}/{int(mask_sig_all.sum())} (missed: {n_miss})")

    # Label-free physics characterisation of every GMM component (paper 3.3 pt 3)
    plot_physics_per_component(phys, assign, mask_sm, k, plots / "physics_per_component.pdf")

    # Global per-observable scale (std over all finite matched values) so the
    # Wasserstein ranking is comparable across observables, not by unit magnitude.
    phys_scale = {}
    for var in PHYSICS_VARS:
        vals = phys[var][np.isfinite(phys[var])]
        phys_scale[var] = float(np.std(vals)) if len(vals) else float("nan")

    counts = np.array([(assign[mask_flag] == i).sum() for i in range(k)], dtype=float)
    frac = counts / counts.sum() if counts.sum() else counts
    comps = populated_components(frac, args.min_frac, args.top_m)
    print(f"Populated components (min_frac={args.min_frac}, top_m={args.top_m}): {comps}")

    # Dominant SM per component. Computed unconditionally — it goes into
    # profile_meta.json for internal validation — but it reaches the figure
    # titles only with --label-components. The paper characterises components by
    # physics alone, so naming one "(~tt hadr)" in a title contradicts the
    # label-free claim exactly as an axis label would.
    dominant = []
    for ki in range(k):
        m = mask_sm & (assign == ki)
        if not m.any():
            dominant.append("—")
            continue
        counts_c = [(y[m] == c).sum() for c in SM_INDICES]
        dominant.append(CLASS_NAMES[SM_INDICES[int(np.argmax(counts_c))]])
    plot_dominant = dominant if args.label_components else [None] * k

    all_rows = []
    # The per-component figures used to be named hh4b_vs_sm_k*.pdf whatever the signal,
    # so a second signal written to the same directory silently overwrote the first.
    sig_slug = re.sub(r"[^A-Za-z0-9]+", "_", sig_name).strip("_").lower()

    qcd_poor_comps = {}
    for ki in comps:
        m_sig = mask_flag & (assign == ki)
        m_loc = mask_sm & (assign == ki)
        m_miss = mask_missed & (assign == ki)
        m_qcd = mask_qcd & (assign == ki)
        qcd_poor = int(m_qcd.sum()) < args.min_qcd
        qcd_poor_comps[int(ki)] = qcd_poor
        if qcd_poor:
            print(f"  [C{ki}] QCD-poor ({int(m_qcd.sum())} < {args.min_qcd}) — ranking falls back to all-SM reference")
        plot_signal_vs_sm(
            phys,
            m_sig,
            m_loc,
            ki,
            sig_name,
            plot_dominant[ki],
            plots / f"{sig_slug}_vs_sm_k{ki}.pdf",
        )
        plot_ae_mse_local(
            mse,
            m_sig,
            m_loc,
            ki,
            sig_name,
            plot_dominant[ki],
            ae_thr,
            plots / f"ae_mse_{sig_slug}_k{ki}.pdf",
        )
        plot_signal_vs_sm_missed(
            phys,
            m_sig,
            m_miss,
            m_loc,
            ki,
            sig_name,
            plot_dominant[ki],
            plots / f"{sig_slug}_vs_sm_missed_k{ki}.pdf",
        )
        all_rows.extend(wasserstein_rows(phys, phys_scale, m_sig, m_qcd, m_loc, m_miss, ki, qcd_poor))

    write_csv(all_rows, out / "wasserstein_rank.csv")
    write_tex(
        all_rows,
        out / "wasserstein_rank.tex",
        caption=(
            f"Wasserstein-1 distances between AE-flagged {sig_name} and the local "
            f"Standard Model background within populated GMM components, standardised "
            f"by each observable's global standard deviation and ranked by that "
            f"distance. The local QCD column, the autoencoder's "
            f"trained normality, is a control; it is undefined where the component "
            f"holds too few QCD events."
        ),
    )

    # Top variable per component summary (by the primary rank)
    top = []
    for ki in comps:
        block = [r for r in all_rows if r["component"] == ki and r["rank"] == 1]
        if block:
            r0 = block[0]
            top.append({
                "component": ki,
                "dominant_variable": r0["variable"],
                "W1_qcd": r0["W1_qcd"],
                "W1_sm": r0["W1_sm"],
                "rank_ref": r0["rank_ref"],
            })

    meta = {
        "k": k,
        "signal_label": sig,
        "signal_name": sig_name,
        "ae_threshold": ae_thr,
        "ae_threshold_source": thr_src,
        "n_flagged": n_flag,
        "n_missed": n_miss,
        "frac_per_component": frac.tolist(),
        "populated_components": comps,
        "dominant_sm_per_component": dominant,
        "top_variable_per_component": top,
        "qcd_labels": args.qcd_labels,
        "min_qcd": args.min_qcd,
        "qcd_poor_per_component": qcd_poor_comps,
        "wasserstein_standardised": True,
        "phys_scale": phys_scale,
        "gmm_path": str(args.gmm_path),
    }
    save_json(out / "profile_meta.json", meta)
    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
