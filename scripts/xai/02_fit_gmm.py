#!/usr/bin/env python3
"""
XAI paper pipeline — Step 02: fit diagonal GMM at chosen K.

Reads K from --k or from k_selection.json (step 01). Saves:

  gmm_K{k}.pkl
  plots/cluster_composition.pdf   (optional, SM test)

Usage:
    python scripts/xai/02_fit_gmm.py \\
        --embeddings-dir /eos/.../embeddings \\
        --output-dir     /eos/.../xai_paper/gmm \\
        --k 12
"""

from __future__ import annotations

import argparse
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
from sklearn.mixture import GaussianMixture

from common.constants import CLASS_NAMES, SM_INDICES
from common.io_embeddings import filter_classes, filter_low_norm, load_train_val_test
from common.utils import load_json, save_json


def fit_gmm(Z: np.ndarray, k: int, seed: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=seed,
        max_iter=500,
        n_init=5,
        reg_covar=1e-6,
    )
    gmm.fit(Z)
    print(f"K={k}: converged={gmm.converged_}, n_iter={gmm.n_iter_}, BIC(train)={gmm.bic(Z):.1f}")
    return gmm


def composition_matrix(assignments: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Row-normalised: P(class | component)."""
    mat = np.zeros((k, len(SM_INDICES)), dtype=float)
    for ki in range(k):
        mask = assignments == ki
        if not mask.any():
            continue
        for ci, c in enumerate(SM_INDICES):
            mat[ki, ci] = ((labels == c) & mask).sum()
        s = mat[ki].sum()
        if s > 0:
            mat[ki] /= s
    return mat


def plot_composition(mat: np.ndarray, k: int, path: Path):
    names = [CLASS_NAMES[c] for c in SM_INDICES]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), max(4, k * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="P(class | component)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{i}" for i in range(k)])
    ax.set_title(f"SM cluster composition (K={k})")
    for i in range(k):
        j = int(mat[i].argmax())
        if mat[i, j] > 0.05:
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="navy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def resolve_k(args) -> int:
    if args.k is not None:
        return int(args.k)
    if args.k_selection_json is not None:
        data = load_json(args.k_selection_json)
        return int(data["selection"]["selected_K"])
    raise ValueError("Provide --k or --k-selection-json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--k-selection-json", type=Path, default=None)
    p.add_argument("--gmm-seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=500_000)
    p.add_argument("--filter-norm-percentile", type=float, default=0.0)
    p.add_argument("--no-composition-plot", action="store_true")
    args = p.parse_args()

    k = resolve_k(args)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    splits = load_train_val_test(args.embeddings_dir)
    X_tr, y_tr = filter_classes(*splits["train"], SM_INDICES)
    X_tr, y_tr, thr = filter_low_norm(X_tr, y_tr, args.filter_norm_percentile)

    rng = np.random.default_rng(args.gmm_seed)
    if len(X_tr) > args.max_train:
        sel = rng.choice(len(X_tr), args.max_train, replace=False)
        X_tr, y_tr = X_tr[sel], y_tr[sel]

    print(f"Fitting GMM K={k} on SM train {X_tr.shape}")
    gmm = fit_gmm(X_tr, k=k, seed=args.gmm_seed)

    gmm_path = out / f"gmm_K{k}.pkl"
    joblib.dump(gmm, gmm_path)
    print(f"Saved {gmm_path}")

    meta = {
        "k": k,
        "cov_type": "diag",
        "n_train": int(len(X_tr)),
        "d_model": int(X_tr.shape[1]),
        "gmm_seed": args.gmm_seed,
        "norm_threshold": thr,
        "gmm_path": str(gmm_path),
        "embeddings_dir": str(args.embeddings_dir),
    }

    if not args.no_composition_plot:
        X_te, y_te = filter_classes(*splits["test"], SM_INDICES)
        if thr > 0:
            X_te, y_te, _ = filter_low_norm(X_te, y_te, 0.0, threshold=thr)
        assign = gmm.predict(X_te)
        mat = composition_matrix(assign, y_te, k)
        plot_composition(mat, k, plots / "cluster_composition.pdf")
        meta["dominant_sm"] = [CLASS_NAMES[SM_INDICES[int(mat[i].argmax())]] for i in range(k)]

    save_json(out / "gmm_meta.json", meta)
    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
