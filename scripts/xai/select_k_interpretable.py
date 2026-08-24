#!/usr/bin/env python3
"""
Select K for the interpretability GMM: the smallest number of regions that is
reproducible and physically distinguishable.

Why not the criteria of step 01. Two measurements rule them out on this data:

  - BIC(val) decreases monotonically out to K=40 in every embedding dimension
    tested, with steps that do not shrink at the edge of the grid. The penalty
    grows as n_params*log(n) while the likelihood term grows as n, so at ~10^5-10^6
    events it is roughly an order of magnitude too weak to produce a minimum.
    Selecting on BIC alone therefore returns the largest K on the grid, whatever
    that happens to be.
  - Mean pairwise ARI decreases with K almost by construction (more components,
    more ways for two fits to disagree), so selecting on ARI alone returns the
    smallest K on the grid.

Each criterion alone degenerates to an edge of the grid. This script adds a third
measurement that opposes them for a reason specific to what the mixture is used
for here — interpretation, not density estimation:

  - Physical distinctiveness. Each component is summarised by the median of the
    high-level observables among the SM events it holds, standardised by each
    observable's global spread (the same scaling as the Wasserstein ranking). Two
    components whose profiles coincide describe the same physics and give the
    reader nothing to tell apart, however well they improve the likelihood. We
    report the minimum and median pairwise distance between occupied component
    profiles; the minimum collapses once K starts splitting regions that are not
    physically distinct.

Stability is measured at PRODUCTION settings, which step 01 does not do: it
evaluates ARI with n_init=1 while the fitted mixture uses n_init=5, so it reports
the fragility of a procedure that is never used. Measured correctly the same
configuration moves from 0.56 to 0.82 on d256 at full statistics.

Selection rule (transparent and configurable): among K values that are stable
(ARI >= --ari-threshold) and physically distinct (min profile distance >=
--min-profile-dist), take the SMALLEST. Preferring few regions is a deliberate
choice — the mixture exists to be read by a person — and it is stated rather than
smuggled in through a criterion that happens to favour it.

All curves are written out, so the rule can be re-applied at other thresholds
without refitting.

Usage:
    python scripts/xai/select_k_interpretable.py \\
        --embeddings-dir /eos/.../embeddings \\
        --matched-npz /eos/.../04_profile/matched_sm_hh4b.npz \\
        --output-dir /eos/.../k_selection_v2 \\
        --k-values 4 6 8 10 12 14 16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import combinations
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
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from common.constants import PHYSICS_VARS, SM_INDICES
from common.io_embeddings import filter_classes, load_train_val_test
from common.physics import load_matched_npz
from common.style import OI
from common.utils import save_json

C_STAB = OI["sm"]
C_DIST = OI["signal"]
C_BIC = OI["neutral"]


def fit(Z, k, seed, n_init, max_iter, cov_type="diag"):
    g = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=seed,
                        max_iter=max_iter, n_init=n_init, reg_covar=1e-6)
    g.fit(Z)
    return g


def bic_on_val(gmm, Z_val):
    n = len(Z_val)
    return float(-2 * gmm.score(Z_val) * n + gmm._n_parameters() * np.log(n))


def profile_matrix(gmm, Z, y, phys, phys_scale, min_frac):
    """Standardised median profile per occupied component, shape (n_occupied, n_vars)."""
    assign = gmm.predict(Z)
    mask_sm = np.isin(y, SM_INDICES)
    k = gmm.n_components
    n_sm = int(mask_sm.sum())
    rows, occ = [], []
    for ki in range(k):
        m = mask_sm & (assign == ki)
        if m.sum() < max(20, min_frac * n_sm):
            continue
        prof = []
        for var in PHYSICS_VARS:
            v = phys[var][m]
            v = v[np.isfinite(v)]
            sc = phys_scale[var]
            prof.append(np.median(v) / sc if len(v) and sc > 0 else np.nan)
        rows.append(prof)
        occ.append(ki)
    return np.array(rows, dtype=float), occ


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--matched-npz", type=Path, required=True,
                   help="From step 04 — supplies the physics used for distinctiveness")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--k-values", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14, 16])
    p.add_argument("--n-init", type=int, default=5, help="Production setting")
    p.add_argument("--n-restarts", type=int, default=6, help="Independent fits compared for ARI")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-train", type=int, default=0, help="0 = use all SM train")
    p.add_argument("--max-val", type=int, default=100_000)
    p.add_argument("--ari-threshold", type=float, default=0.80)
    p.add_argument("--min-profile-dist", type=float, default=0.25)
    p.add_argument("--min-frac", type=float, default=0.01,
                   help="Components below this share of SM are ignored as unoccupied")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pca-var", type=float, default=0.0,
        help="If >0, project onto the leading principal components, fitted on SM "
             "train alone. A value below 1 is read as a fraction of the variance to "
             "retain; a value of 1 or more as an explicit number of components. "
             "VCReg's variance term keeps every dimension high-variance, so the "
             "spectrum is nearly flat and a variance target barely reduces anything "
             "(0.99 leaves 209 of 256 dimensions, and measurably hurt the ARI). "
             "Reaching the regime where a mixture is actually better conditioned "
             "therefore needs an explicit count — 16 or 32 — which is what the "
             "component form is for. No whitening: rescaling the axes would change "
             "the geometry a diagonal covariance sees.",
    )
    p.add_argument(
        "--cov-type", default="diag", choices=["diag", "full"],
        help="VCReg's covariance term decorrelates the embedding dimensions "
             "globally, so a diagonal mixture matches the geometry the objective "
             "builds. That decorrelation is a batch statistic, not a per-component "
             "one, so 'full' tests whether local correlations remain — affordable "
             "only at small d_model (265k parameters at d=256 against 1.2M events).",
    )
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    splits = load_train_val_test(args.embeddings_dir)
    X_tr, _ = filter_classes(*splits["train"], SM_INDICES)
    X_va, _ = filter_classes(*splits["val"], SM_INDICES)
    rng = np.random.default_rng(args.seed)
    if args.max_train and len(X_tr) > args.max_train:
        X_tr = X_tr[rng.choice(len(X_tr), args.max_train, replace=False)]
    if len(X_va) > args.max_val:
        X_va = X_va[rng.choice(len(X_va), args.max_val, replace=False)]
    print(f"SM train {len(X_tr):,} | SM val {len(X_va):,} | dim {X_tr.shape[1]}")

    Z, y, phys = load_matched_npz(args.matched_npz)

    pca = None
    if args.pca_var > 0:
        # sklearn switches on the type, not the value: an int is a component count,
        # a float in (0, 1) a variance target. Passing 32.0 would be rejected.
        n_comp = int(args.pca_var) if args.pca_var >= 1 else args.pca_var
        pca = PCA(n_components=n_comp, svd_solver="full", whiten=False,
                  random_state=args.seed)
        # Fitted on SM train only: val and the matched arrays are projected with it,
        # never used to define it.
        pca.fit(X_tr)
        d0 = X_tr.shape[1]
        X_tr, X_va, Z = pca.transform(X_tr), pca.transform(X_va), pca.transform(Z)
        print(f"PCA: {d0} -> {X_tr.shape[1]} dims, "
              f"{pca.explained_variance_ratio_.sum():.4f} of the variance retained")

    phys_scale = {}
    for var in PHYSICS_VARS:
        v = phys[var][np.isfinite(phys[var])]
        phys_scale[var] = float(np.std(v)) if len(v) else float("nan")

    rows = []
    for k in args.k_values:
        t0 = time.time()
        gmms = [fit(X_tr, k, args.seed + i, args.n_init, args.max_iter, args.cov_type)
                for i in range(args.n_restarts)]
        assigns = [g.predict(X_tr) for g in gmms]
        pair = [adjusted_rand_score(assigns[a], assigns[b])
                for a, b in combinations(range(args.n_restarts), 2)]
        ari_mean, ari_std = float(np.mean(pair)), float(np.std(pair))

        best = min(gmms, key=lambda g: bic_on_val(g, X_va))
        bic = bic_on_val(best, X_va)

        prof, occ = profile_matrix(best, Z, y, phys, phys_scale, args.min_frac)
        if len(prof) >= 2:
            d = [np.linalg.norm(prof[a] - prof[b]) for a, b in combinations(range(len(prof)), 2)]
            dmin, dmed = float(np.nanmin(d)), float(np.nanmedian(d))
        else:
            dmin = dmed = float("nan")

        dt = time.time() - t0
        rows.append({"k": k, "bic_val": bic, "ari_mean": ari_mean, "ari_std": ari_std,
                     "n_occupied": len(occ), "profile_dist_min": dmin,
                     "profile_dist_median": dmed, "seconds": dt})
        print(f"K={k:3d}  ARI={ari_mean:.3f}±{ari_std:.3f}  BIC={bic/1e6:.3f}M  "
              f"occupied={len(occ)}/{k}  min-profile-dist={dmin:.3f}  ({dt/60:.1f} min)")
        joblib.dump(best, out / f"gmm_K{k}.pkl")

    stable = [r for r in rows if r["ari_mean"] >= args.ari_threshold]
    distinct = [r for r in stable
                if np.isfinite(r["profile_dist_min"]) and r["profile_dist_min"] >= args.min_profile_dist]
    if distinct:
        chosen, reason = min(distinct, key=lambda r: r["k"]), "smallest K that is stable and physically distinct"
    elif stable:
        chosen, reason = min(stable, key=lambda r: r["k"]), "smallest stable K (no K met the distinctiveness floor)"
    else:
        chosen = max(rows, key=lambda r: r["ari_mean"])
        reason = "no K met the stability threshold; most stable K reported instead"

    print(f"\nSelected K = {chosen['k']}  ({reason})")

    save_json(out / "k_selection_v2.json", {
        "k_values": args.k_values, "results": rows,
        "settings": {"n_init": args.n_init, "n_restarts": args.n_restarts,
                     "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "cov_type": args.cov_type,
                     "pca_var": args.pca_var,
                     "n_dims_used": int(X_tr.shape[1]),
                     "ari_threshold": args.ari_threshold,
                     "min_profile_dist": args.min_profile_dist},
        "selection": {"selected_K": chosen["k"], "reason": reason},
    })
    with open(out / "k_selection_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "ari_mean", "ari_std", "bic_val",
                                          "n_occupied", "profile_dist_min",
                                          "profile_dist_median", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    axes[0].errorbar(ks, [r["ari_mean"] for r in rows], yerr=[r["ari_std"] for r in rows],
                     marker="o", color=C_STAB, capsize=3)
    axes[0].axhline(args.ari_threshold, ls="--", color="0.35", lw=1)
    axes[0].set_ylabel("mean pairwise ARI")
    axes[0].set_title("Stability", fontsize=10)

    axes[1].plot(ks, [r["bic_val"] / 1e6 for r in rows], marker="o", color=C_BIC)
    axes[1].set_ylabel("BIC on validation [$10^6$]")
    axes[1].set_title("Density fit", fontsize=10)

    axes[2].plot(ks, [r["profile_dist_min"] for r in rows], marker="o", color=C_DIST)
    axes[2].axhline(args.min_profile_dist, ls="--", color="0.35", lw=1)
    axes[2].set_ylabel("min pairwise profile distance")
    axes[2].set_title("Physical distinctiveness", fontsize=10)

    for ax in axes:
        ax.set_xlabel("$K$")
        ax.axvline(chosen["k"], color=C_DIST, lw=1.0, alpha=0.35)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Selection of $K$ — selected $K$ = {chosen['k']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "k_selection_v2.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}/k_selection_v2.{{json,csv,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
