#!/usr/bin/env python3
"""
XAI paper pipeline — Step 01: select GMM K (Appendix).

Fits diagonal GMMs over a K grid on SM train embeddings, scores BIC on a
held-out SM validation set, and measures partition stability (mean pairwise ARI
across random initialisations). Writes:

  bic_vs_k.pdf, ari_vs_k.pdf, k_selection.json

Usage:
    python scripts/xai/01_select_k.py \\
        --embeddings-dir /eos/.../embeddings \\
        --output-dir     /eos/.../xai_paper/select_k \\
        --k-values 4 6 8 10 12 \\
        [--ari-threshold 0.8]
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

_XAI = Path(__file__).resolve().parent
_ROOT = _XAI.parents[1]
sys.path.insert(0, str(_XAI))
sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from common.constants import SM_INDICES
from common.style import OI
from common.io_embeddings import filter_classes, filter_low_norm, load_train_val_test
from common.utils import save_json

COV_TYPE = "diag"
GMM_MAX_ITER = 300
DEFAULT_K = [4, 6, 8, 10, 12]


def fit_gmm(Z: np.ndarray, k: int, seed: int, n_init: int = 1) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=COV_TYPE,
        random_state=seed,
        max_iter=GMM_MAX_ITER,
        n_init=n_init,
        reg_covar=1e-6,
    )
    gmm.fit(Z)
    return gmm


def bic_on_val(gmm: GaussianMixture, Z_val: np.ndarray) -> float:
    # sklearn bic() expects the data used for the score; we evaluate on val.
    n_samples, _ = Z_val.shape
    score = gmm.score(Z_val)  # mean log-likelihood
    n_params = gmm._n_parameters()
    return float(-2 * score * n_samples + n_params * np.log(n_samples))


def mean_pairwise_ari(Z: np.ndarray, k: int, n_inits: int, base_seed: int) -> float:
    assigns = []
    for i in range(n_inits):
        gmm = fit_gmm(Z, k=k, seed=base_seed + i, n_init=1)
        assigns.append(gmm.predict(Z))
    scores = [
        adjusted_rand_score(assigns[a], assigns[b])
        for a, b in combinations(range(n_inits), 2)
    ]
    return float(np.mean(scores)) if scores else 1.0


def choose_k(ks, bic, ari, ari_threshold: float) -> dict:
    """Smallest K past BIC elbow with ARI >= threshold; fallback = best BIC with ARI ok."""
    bic = np.asarray(bic, dtype=float)
    ari = np.asarray(ari, dtype=float)
    ks = list(ks)

    # Discrete elbow: largest drop in BIC followed by flattening
    deltas = np.diff(bic)
    elbow_idx = int(np.argmin(deltas)) + 1 if len(deltas) else 0
    elbow_idx = min(elbow_idx, len(ks) - 1)

    chosen = None
    for i in range(elbow_idx, len(ks)):
        if ari[i] >= ari_threshold:
            chosen = ks[i]
            reason = f"elbow_idx={elbow_idx}, first K>=elbow with ARI>={ari_threshold}"
            break
    if chosen is None:
        ok = np.where(ari >= ari_threshold)[0]
        if len(ok):
            i = int(ok[np.argmin(bic[ok])])
            chosen = ks[i]
            reason = f"min BIC among ARI>={ari_threshold}"
        else:
            i = int(np.argmin(bic))
            chosen = ks[i]
            reason = "fallback_min_BIC (ARI threshold unmet)"

    return {
        "selected_K": int(chosen),
        "elbow_index": int(elbow_idx),
        "elbow_K": int(ks[elbow_idx]),
        "reason": reason,
        "ari_threshold": ari_threshold,
    }


def plot_bic(ks, bic, selected_k, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, bic, "o-", color=OI["sm"])
    ax.axvline(selected_k, color="crimson", ls="--", label=f"selected K={selected_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("BIC (SM validation)")
    ax.set_title("GMM BIC vs K")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_ari(ks, ari, selected_k, ari_threshold, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, ari, "s-", color=OI["latent"])
    ax.axhline(ari_threshold, color="gray", ls=":", label=f"ARI≥{ari_threshold}")
    ax.axvline(selected_k, color="crimson", ls="--", label=f"selected K={selected_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean pairwise ARI")
    ax.set_ylim(0, 1.05)
    ax.set_title("GMM assignment stability vs K")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K)
    p.add_argument("--ari-threshold", type=float, default=0.8)
    p.add_argument("--n-ari-inits", type=int, default=5, help="Random inits for ARI")
    p.add_argument("--gmm-seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=200_000, help="Subsample SM train for speed")
    p.add_argument("--max-val", type=int, default=100_000)
    p.add_argument("--filter-norm-percentile", type=float, default=0.0)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    splits = load_train_val_test(args.embeddings_dir)
    X_tr, y_tr = filter_classes(*splits["train"], SM_INDICES)
    if splits["val"] is not None:
        X_va, y_va = filter_classes(*splits["val"], SM_INDICES)
    else:
        print("[WARN] No val split — holding out 20% of SM train as validation")
        rng = np.random.default_rng(args.gmm_seed)
        idx = rng.permutation(len(X_tr))
        n_va = max(1, int(0.2 * len(X_tr)))
        X_va, y_va = X_tr[idx[:n_va]], y_tr[idx[:n_va]]
        X_tr, y_tr = X_tr[idx[n_va:]], y_tr[idx[n_va:]]

    X_tr, y_tr, thr = filter_low_norm(X_tr, y_tr, args.filter_norm_percentile)
    if thr > 0:
        X_va, y_va, _ = filter_low_norm(X_va, y_va, 0.0, threshold=thr)

    rng = np.random.default_rng(args.gmm_seed)
    if len(X_tr) > args.max_train:
        sel = rng.choice(len(X_tr), args.max_train, replace=False)
        X_tr, y_tr = X_tr[sel], y_tr[sel]
    if len(X_va) > args.max_val:
        sel = rng.choice(len(X_va), args.max_val, replace=False)
        X_va, y_va = X_va[sel], y_va[sel]

    print(f"SM train: {X_tr.shape}  |  SM val: {X_va.shape}  |  d={X_tr.shape[1]}")

    bic_list, ari_list = [], []
    for k in args.k_values:
        print(f"\n=== K={k} ===")
        gmm = fit_gmm(X_tr, k=k, seed=args.gmm_seed, n_init=3)
        bic = bic_on_val(gmm, X_va)
        ari = mean_pairwise_ari(X_tr, k=k, n_inits=args.n_ari_inits, base_seed=args.gmm_seed)
        print(f"  BIC(val)={bic:.1f}  mean_ARI={ari:.3f}  converged={gmm.converged_}")
        bic_list.append(bic)
        ari_list.append(ari)

    selection = choose_k(args.k_values, bic_list, ari_list, args.ari_threshold)
    print(f"\nSelected K={selection['selected_K']}  ({selection['reason']})")

    result = {
        "k_values": list(args.k_values),
        "bic_val": bic_list,
        "mean_ari": ari_list,
        "selection": selection,
        "cov_type": COV_TYPE,
        "embeddings_dir": str(args.embeddings_dir),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "d_model": int(X_tr.shape[1]),
        "norm_threshold": thr,
    }
    save_json(out / "k_selection.json", result)

    plot_bic(args.k_values, bic_list, selection["selected_K"], plots / "bic_vs_k.pdf")
    plot_ari(
        args.k_values,
        ari_list,
        selection["selected_K"],
        args.ari_threshold,
        plots / "ari_vs_k.pdf",
    )
    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
