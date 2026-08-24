#!/usr/bin/env python3
"""
Diagnostic: is the low ARI in step 01 a data problem or an n_init problem?

Step 01 reports mean pairwise ARI around 0.6-0.7, never reaching its own 0.8
threshold, so K selection falls back to min-BIC. Two candidate causes, which
this script separates because they imply different fixes:

  1. Too little data. Step 01 subsamples SM train to 200k of the ~1.15M
     available (17%). With a poorly determined likelihood surface, EM lands in
     different local optima from different inits and ARI drops.

  2. An n_init mismatch. Step 01 measures ARI with n_init=1 (one random start
     per fit), while the production GMM in step 02 uses n_init=5 (best of five).
     The reported ARI therefore describes the reproducibility of a procedure
     that is never used, and understates the stability of the actual mixture.

Note this does NOT test the BIC elbow, and more data cannot fix that: the BIC
penalty grows as n_params*log(n) while the likelihood term grows as n, so
enlarging the sample makes BIC favour *larger* K. See wrapper_k_diagnostic.sh.

Writes ari_scaling.json / .csv: mean pairwise ARI for each (n_train, n_init).

Usage:
    python scripts/xai/diagnose_ari_scaling.py \\
        --embeddings-dir /eos/.../embeddings --output-dir /eos/.../ari_scaling \\
        --k 12 --n-train 200000 600000 1150000 --n-init 1 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import combinations
from pathlib import Path

_XAI = Path(__file__).resolve().parent
_ROOT = _XAI.parents[1]
sys.path.insert(0, str(_XAI))
sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from common.constants import SM_INDICES
from common.io_embeddings import filter_classes, load_train_val_test
from common.utils import save_json


def mean_pairwise_ari(Z: np.ndarray, k: int, n_restarts: int, n_init: int,
                      base_seed: int, max_iter: int) -> tuple[float, float, list]:
    """Fit `n_restarts` independent GMMs and average ARI over all pairs.

    `n_init` is the inner sklearn restart count each fit gets — the quantity
    under test. n_init=1 reproduces step 01; n_init=5 matches step 02's
    production fit.
    """
    assigns = []
    for i in range(n_restarts):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=base_seed + i,
            max_iter=max_iter,
            n_init=n_init,
            reg_covar=1e-6,
        )
        gmm.fit(Z)
        assigns.append(gmm.predict(Z))
    scores = [
        adjusted_rand_score(assigns[a], assigns[b])
        for a, b in combinations(range(n_restarts), 2)
    ]
    return float(np.mean(scores)), float(np.std(scores)), scores


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--n-train", type=int, nargs="+", default=[200_000, 600_000, 1_150_000])
    p.add_argument("--n-init", type=int, nargs="+", default=[1, 5])
    p.add_argument("--n-restarts", type=int, default=4, help="Independent fits to compare")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--gmm-seed", type=int, default=42)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    splits = load_train_val_test(args.embeddings_dir)
    X_all, _ = filter_classes(*splits["train"], SM_INDICES)
    print(f"SM train available: {len(X_all):,}  (dim {X_all.shape[1]})")

    rng = np.random.default_rng(args.gmm_seed)
    rows = []
    for n_train in args.n_train:
        n = min(n_train, len(X_all))
        idx = rng.choice(len(X_all), n, replace=False) if n < len(X_all) else np.arange(len(X_all))
        Z = X_all[idx]
        for n_init in args.n_init:
            t0 = time.time()
            mean, std, scores = mean_pairwise_ari(
                Z, args.k, args.n_restarts, n_init, args.gmm_seed, args.max_iter
            )
            dt = time.time() - t0
            print(f"n_train={n:>9,}  n_init={n_init}  ARI={mean:.4f} ± {std:.4f}   ({dt/60:.1f} min)")
            rows.append({
                "n_train": int(n),
                "n_init": int(n_init),
                "ari_mean": mean,
                "ari_std": std,
                "pairwise": scores,
                "seconds": dt,
            })

    save_json(out / "ari_scaling.json", {
        "k": args.k,
        "n_restarts": args.n_restarts,
        "max_iter": args.max_iter,
        "n_sm_available": int(len(X_all)),
        "d_model": int(X_all.shape[1]),
        "results": rows,
    })
    with open(out / "ari_scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_train", "n_init", "ari_mean", "ari_std", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if k.startswith("ari") else r[k])
                        for k in ["n_train", "n_init", "ari_mean", "ari_std", "seconds"]})
    print(f"\nSaved {out}/ari_scaling.{{json,csv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
