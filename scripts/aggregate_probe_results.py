"""
Aggregate probe_results.json across seeds into mean ± std.

Usage:
    python scripts/aggregate_probe_results.py \
        --results-dir /eos/user/d/dgenoves/anomaly_pipeline/probe_results/augsupcon_d128 \
        --output /eos/user/d/dgenoves/anomaly_pipeline/probe_results/augsupcon_d128/aggregated_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Collect all seed results
    seed_results = []
    seed_dirs = sorted(results_dir.glob("seed_*"))
    for seed_dir in seed_dirs:
        # support both direct and probe_evaluation/ subdirectory layouts
        json_path = seed_dir / "probe_evaluation" / "probe_results.json"
        if not json_path.exists():
            json_path = seed_dir / "probe_results.json"
        if not json_path.exists():
            print(f"  WARNING: missing probe_results.json in {seed_dir}, skipping.")
            continue
        with open(json_path) as f:
            seed_results.append(json.load(f))

    if not seed_results:
        print("ERROR: no probe_results.json found.")
        return

    n_seeds = len(seed_results)
    print(f"Aggregating {n_seeds} seeds from {results_dir}")

    # Collect all metric keys
    all_keys = sorted(seed_results[0].keys())

    summary = {
        "n_seeds": n_seeds,
        "seed_dirs": [str(d) for d in seed_dirs if (d / "probe_evaluation" / "probe_results.json").exists() or (d / "probe_results.json").exists()],
    }

    for key in all_keys:
        vals = []
        for r in seed_results:
            if key in r and isinstance(r[key], (int, float)):
                vals.append(float(r[key]))
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"]  = float(np.std(vals))
            summary[f"{key}_vals"] = vals

    # Print summary table
    print(f"\n{'Metric':<50} {'Mean':>10} {'Std':>10}")
    print("-" * 72)
    for key in all_keys:
        if f"{key}_mean" in summary:
            print(f"{key:<50} {summary[f'{key}_mean']:>10.6f} {summary[f'{key}_std']:>10.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
