#!/usr/bin/env python3
"""Per-class and macro AUROC of the CE baseline, aggregated over its encoder seeds.

CE is a classifier, so Tables 4 and 8 report its own test predictions rather than a
linear probe on its embeddings. Training writes those predictions through the
SaveLogits callback to <run-dir>/seed_N/test_predictions/test_logits_and_labels.npz.
Each class is scored one-vs-rest on the softmax probability of that class -- what
MultiClassROC logs per run during the test step -- and the seeds are summarised as mean
and sample standard deviation (ddof=1), the convention of the paper's tables.
MultiClassROC only logs per run; nothing else in the tree aggregates over seeds, which
is why this script exists. It reproduces the CE column of Table 8 to four decimals.

Usage:
    python scripts/aggregate_ce_auroc.py                    # the d256 run the paper reports
    python scripts/aggregate_ce_auroc.py --run-dir <dir> --output ce_auroc.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_RUN = Path("/eos/user/d/dgenoves/anomaly_pipeline/new_exp/"
                   "fm_testing_12class_nosparse_dmodel256_cern")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                    help="CE run directory holding seed_*/ (default: the d256 run)")
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON summary")
    a = ap.parse_args()

    files = sorted(a.run_dir.glob("seed_*/test_predictions/test_logits_and_labels.npz"))
    if len(files) < 2:
        print(f"Found {len(files)} prediction files under {a.run_dir}; need at least two "
              f"seeds for a standard deviation.")
        return 1

    names, per_seed = None, []
    for f in files:
        d = np.load(f, allow_pickle=True)
        logits, y = d["logits"], d["labels"]
        n = ([str(x) for x in d["class_names"]] if "class_names" in d
             else [str(i) for i in range(logits.shape[1])])
        if names is None:
            names = n
        elif n != names:
            # Labels are positions; averaging across seeds with different orders would
            # mix classes silently.
            print(f"Class order in {f} differs from the first seed; refusing to average.")
            return 1
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        per_seed.append([roc_auc_score(y == c, p[:, c]) for c in range(p.shape[1])])

    per_seed = np.array(per_seed)
    macro = per_seed.mean(axis=1)
    print(f"{len(files)} seeds from {a.run_dir}")
    rows = {}
    for c, n in enumerate(names):
        m, s = float(per_seed[:, c].mean()), float(per_seed[:, c].std(ddof=1))
        rows[n] = {"mean": m, "std": s}
        print(f"  {n:15s} {m:.4f} ± {s:.4f}")
    rows["macro"] = {"mean": float(macro.mean()), "std": float(macro.std(ddof=1))}
    print(f"  {'macro':15s} {macro.mean():.4f} ± {macro.std(ddof=1):.4f}")

    if a.output:
        a.output.write_text(json.dumps({"run_dir": str(a.run_dir), "n_seeds": len(files),
                                        "auroc": rows}, indent=2))
        print(f"Saved {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
