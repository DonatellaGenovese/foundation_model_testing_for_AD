#!/usr/bin/env python3
"""Number the components of a refitted mixture like the published one.

Component indices of a Gaussian mixture are arbitrary: two fits of the same partition
can list the same components in a different order. The paper's figures and tables name
components (C5, C4, C2), so a refit is only comparable once its components carry the
published numbers. This finds the permutation that best matches the means of the two
mixtures (Hungarian assignment on squared distance) and rewrites the refit with that
order. The permutation is written next to the output; the identity means the refit
already agreed.

Usage:
    python scripts/xai/align_mixture.py --gmm refit/gmm_K7.pkl \\
        --reference /eos/.../gmm_K7.pkl --out refit/gmm_K7.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment


def permute(g, order):
    for attr in ("weights_", "means_", "covariances_", "precisions_", "precisions_cholesky_"):
        if hasattr(g, attr):
            setattr(g, attr, getattr(g, attr)[order])
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmm", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    g, ref = joblib.load(a.gmm), joblib.load(a.reference)
    cost = ((ref.means_[:, None, :] - g.means_[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)                 # reference component r <- refit component c
    order = c[np.argsort(r)]
    identity = bool(np.array_equal(order, np.arange(len(order))))
    g = permute(g, order)
    joblib.dump(g, a.out)
    info = {
        "reference": str(a.reference), "refit": str(a.gmm), "order": order.tolist(), "identity": identity,
        "max_abs_weight_diff": float(np.abs(g.weights_ - ref.weights_).max()),
        "max_abs_mean_diff": float(np.abs(g.means_ - ref.means_).max()),
    }
    Path(str(a.out) + ".alignment.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
