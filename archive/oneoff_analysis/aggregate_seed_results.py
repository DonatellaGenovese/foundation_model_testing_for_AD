    - "QCD_inclusive"
    - "DY_to_ll"
    - "Z_to_vv_jet"
    - "Z_to_qq_uds"
    - "Z_to_bb"
    - "Z_to_cc"
    - "W_to_lv"
    - "W_to_qq"
    - "gamma"
    - "QCD_bb"
    - "tt_all-hadr"
    - "tt_semi-lept"
    - "tt_all-lept"
    - "Minbias_Soft_QCD"
    - "Upsilon_to_Leptons"
    - "VBFHbb"
    - "HH_4b"
    - "ggHtautau""""
Aggregate metrics across seeds for each encoder type.

Reads metrics.json from each seed_N subdirectory under the pipeline output
and computes mean ± std across seeds.

Expected directory layout:
    <base_dir>/
        aug_supcon_12class_nosparse_dmodel128_cern/
            seed_0/metrics.json
            seed_1/metrics.json
            ...
        aug_supcon_12class_nosparse_selfsup_dmodel128_cern/
            seed_0/metrics.json
            ...

Usage:
    python scripts/aggregate_seed_results.py \
        --base-dir /eos/user/d/dgenoves/anomaly_pipeline/seeds

Outputs:
    logs/seed_results/summary.csv
    logs/seed_results/summary.txt
"""

import argparse
import csv
import json
import math
from pathlib import Path

ENCODERS = {
    "aug":     "aug_supcon_12class_nosparse_dmodel128_cern",
    "selfsup": "aug_supcon_12class_nosparse_selfsup_dmodel128_cern",
}

# Core AE metrics always present
METRICS = ["separation_ratio", "val_drift_metric", "drift_metric", "drift_fpr01", "drift_fpr05", "drift_fpr10"]

# Aggregate linear probe metrics always present
PROBE_METRICS = ["linear_probe_accuracy", "linear_probe_f1_macro", "linear_probe_auroc_macro"]

# 15 classes trained in Phase 1
CLASSES = [
    "QCD_inclusive", "DY_to_ll", "Z_to_vv_jet", "Z_to_qq_uds", "Z_to_bb",
    "Z_to_cc", "W_to_lv", "W_to_qq", "gamma", "QCD_bb",
    "tt_all-hadr", "tt_semi-lept", "tt_all-lept", "Minbias_Soft_QCD", "Upsilon_to_Leptons",
]
PER_CLASS_METRICS = (
    [f"linear_probe_f1_{c}"    for c in CLASSES] +
    [f"linear_probe_auroc_{c}" for c in CLASSES]
)

ALL_METRICS = METRICS + PROBE_METRICS + PER_CLASS_METRICS

OUTPUT_DIR = Path("logs/seed_results")


def mean_std(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, float("nan")
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(variance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", required=True,
        help="Root directory containing one subdir per encoder type"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for enc, subdir_name in ENCODERS.items():
        enc_dir = base_dir / subdir_name
        if not enc_dir.exists():
            print(f"WARNING: {enc_dir} not found, skipping {enc}")
            continue

        seed_dirs = sorted(enc_dir.glob("seed_*"))
        if not seed_dirs:
            print(f"WARNING: no seed_* dirs found in {enc_dir}, skipping {enc}")
            continue

        collected = {m: [] for m in ALL_METRICS}
        seeds_found = []

        for sd in seed_dirs:
            mpath = sd / "metrics.json"
            if not mpath.exists():
                print(f"  {enc} / {sd.name}: metrics.json not found, skipping")
                continue
            data = json.loads(mpath.read_text())
            seed = data.get("seed", sd.name)
            seeds_found.append(seed)
            for m in ALL_METRICS:
                val = data.get(m, float("nan"))
                if not math.isnan(val):
                    collected[m].append(val)

        n = len(seeds_found)
        print(f"\n{enc}: {n} seeds found  ({seeds_found})")

        row = {"encoder": enc, "n_seeds": n}
        for m in ALL_METRICS:
            mu, sigma = mean_std(collected[m])
            row[f"{m}_mean"] = mu
            row[f"{m}_std"] = sigma

        # Print AE metrics
        print("  -- AE metrics --")
        for m in METRICS:
            mu, sigma = row[f"{m}_mean"], row[f"{m}_std"]
            print(f"  {m:35s}: {mu:.4f} ± {sigma:.4f}  (n={len(collected[m])})")

        # Print aggregate probe metrics
        print("  -- Linear probe (aggregate) --")
        for m in PROBE_METRICS:
            mu, sigma = row[f"{m}_mean"], row[f"{m}_std"]
            print(f"  {m:35s}: {mu:.4f} ± {sigma:.4f}  (n={len(collected[m])})")

        # Print per-class F1
        print("  -- Linear probe F1 per class --")
        for c in CLASSES:
            m = f"linear_probe_f1_{c}"
            mu, sigma = row[f"{m}_mean"], row[f"{m}_std"]
            print(f"  {c:35s}: {mu:.4f} ± {sigma:.4f}  (n={len(collected[m])})")

        # Print per-class AUROC
        print("  -- Linear probe AUROC per class --")
        for c in CLASSES:
            m = f"linear_probe_auroc_{c}"
            mu, sigma = row[f"{m}_mean"], row[f"{m}_std"]
            print(f"  {c:35s}: {mu:.4f} ± {sigma:.4f}  (n={len(collected[m])})")

        rows.append(row)

    # --- CSV ---
    out_csv = OUTPUT_DIR / "summary.csv"
    fieldnames = ["encoder", "n_seeds"]
    for m in ALL_METRICS:
        fieldnames += [f"{m}_mean", f"{m}_std"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV to: {out_csv}")

    # --- Text summary (core metrics only for readability) ---
    out_txt = OUTPUT_DIR / "summary.txt"
    lines = []
    summary_metrics = METRICS + PROBE_METRICS
    header = f"{'encoder':<12}  {'n':>3}  " + "  ".join(
        f"{m:>28}" for m in summary_metrics
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        parts = [f"{r['encoder']:<12}  {r['n_seeds']:>3}"]
        for m in summary_metrics:
            mu = r[f"{m}_mean"]
            sd = r[f"{m}_std"]
            parts.append(f"  {mu:6.4f}±{sd:.4f}{'':>10}")
        lines.append("  ".join(parts))

    # Per-class table
    lines.append("")
    lines.append("Per-class F1 (mean ± std):")
    enc_names = [r["encoder"] for r in rows]
    col_w = 14
    lines.append(f"  {'class':<35}" + "".join(f"  {e:>{col_w}}" for e in enc_names))
    lines.append("  " + "-" * (35 + (col_w + 2) * len(enc_names)))
    for c in CLASSES:
        m = f"linear_probe_f1_{c}"
        vals = [f"{rows[i][f'{m}_mean']:.4f}±{rows[i][f'{m}_std']:.4f}" for i in range(len(rows))]
        lines.append(f"  {c:<35}" + "".join(f"  {v:>{col_w}}" for v in vals))

    lines.append("")
    lines.append("Per-class AUROC (mean ± std):")
    lines.append(f"  {'class':<35}" + "".join(f"  {e:>{col_w}}" for e in enc_names))
    lines.append("  " + "-" * (35 + (col_w + 2) * len(enc_names)))
    for c in CLASSES:
        m = f"linear_probe_auroc_{c}"
        vals = [f"{rows[i][f'{m}_mean']:.4f}±{rows[i][f'{m}_std']:.4f}" for i in range(len(rows))]
        lines.append(f"  {c:<35}" + "".join(f"  {v:>{col_w}}" for v in vals))

    txt = "\n".join(lines)
    print(f"\n{txt}")
    out_txt.write_text(txt + "\n")
    print(f"Saved text to: {out_txt}")


if __name__ == "__main__":
    main()
