#!/usr/bin/env python3
"""
Aggregate n_heads ablation probe evaluation results.

Reads probe_results.json from each training run under:
  /eos/user/d/dgenoves/anomaly_pipeline/ablation/nheads/logs/train/runs/

Outputs per-run and mean±std summaries grouped by (model, n_heads).

Usage:
    python3 scripts/ablation/aggregate_probe_results_nheads.py
    python3 scripts/ablation/aggregate_probe_results_nheads.py --plot
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EOS_TRAIN_BASE = Path("/eos/user/d/dgenoves/anomaly_pipeline/ablation/nheads/logs/train/runs")
DEFAULT_OUTPUT = Path("/eos/user/d/dgenoves/anomaly_pipeline/ablation/nheads/results")

MODEL_SHORT = {
    "VCReg": "vcreg",
    "AugmentedSupCon": "aug_supcon",
    "AugmentedSelfSupCon": "selfsupcon",
    "VICReg": "vicreg",
}

KEY_METRICS = [
    "linear_probe_accuracy",
    "linear_probe_f1_macro",
    "linear_probe_auroc_macro",
]


def get_run_info(run_dir: Path):
    cfg = run_dir / ".hydra" / "config.yaml"
    probe = run_dir / "probe_evaluation" / "probe_results.json"
    if not cfg.exists() or not probe.exists():
        return None

    text = cfg.read_text()
    model = next(
        (l.split("_target_:")[-1].strip().split(".")[-1]
         .replace("COLLIDE2V", "").replace("LitModule", "")
         for l in text.split("\n") if "_target_:" in l and "LitModule" in l),
        None,
    )
    nheads = next(
        (int(l.split(":")[-1].strip())
         for l in text.split("\n") if re.match(r"\s+n_heads:", l)),
        None,
    )
    seed = next(
        (int(l.split(":")[-1].strip())
         for l in text.split("\n") if re.match(r"^seed:", l.strip())),
        None,
    )
    if seed is None:
        match = re.search(r"^seed:\s*(\d+)", text, re.MULTILINE)
        seed = int(match.group(1)) if match else None
    if not model or nheads is None or seed is None:
        return None

    with open(probe) as f:
        metrics = json.load(f)

    row = {
        "run_dir": str(run_dir),
        "model": model,
        "model_short": MODEL_SHORT.get(model, model),
        "n_heads": nheads,
        "seed": seed,
    }
    row.update(metrics)
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    metric_cols = sorted({k for r in rows for k in r if k.startswith("linear_probe_")})
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["n_heads"])].append(row)

    out = []
    for (model, n_heads) in sorted(groups):
        sub = groups[(model, n_heads)]
        agg_row = {
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "n_heads": int(n_heads),
            "n_seeds": len(sub),
        }
        for metric in metric_cols:
            vals = [float(r[metric]) for r in sub if metric in r and isinstance(r[metric], (int, float))]
            agg_row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            agg_row[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        out.append(agg_row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(agg: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    models = ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
    nheads_vals = sorted({row["n_heads"] for row in agg})

    for metric in KEY_METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_short in models:
            sub = sorted(
                [r for r in agg if r["model_short"] == model_short],
                key=lambda r: r["n_heads"],
            )
            if not sub:
                continue
            x = [r["n_heads"] for r in sub]
            y = [r[f"{metric}_mean"] for r in sub]
            yerr = [r[f"{metric}_std"] for r in sub]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=model_short)

        ax.set_xticks(nheads_vals)
        ax.set_xlabel("n_heads")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"n_heads ablation — {metric.replace('_', ' ')}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        slug = metric.replace("/", "_")
        fig.savefig(output_dir / f"nheads_{slug}.png", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / f"nheads_{slug}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {output_dir / f'nheads_{slug}.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-base", type=Path, default=EOS_TRAIN_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.train_base.iterdir()):
        if not run_dir.is_dir():
            continue
        info = get_run_info(run_dir)
        if info:
            rows.append(info)

    if not rows:
        print("ERROR: no probe_results.json found.")
        return

    rows.sort(key=lambda r: (r["model_short"], r["n_heads"], r["seed"]))
    agg = aggregate(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = args.output_dir / "probe_results_per_run.csv"
    agg_path = args.output_dir / "probe_results_aggregated.csv"
    json_path = args.output_dir / "probe_results_aggregated.json"

    write_csv(per_run_path, rows)
    write_csv(agg_path, agg)

    summary = {
        "n_runs": len(rows),
        "models": sorted({r["model_short"] for r in rows}),
        "n_heads_values": sorted({int(r["n_heads"]) for r in rows}),
        "aggregated": agg,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Collected {len(rows)} probe results")
    print(f"Saved per-run table:   {per_run_path}")
    print(f"Saved aggregated CSV:  {agg_path}")
    print(f"Saved aggregated JSON: {json_path}")

    print(f"\n{'Model':<14} {'n_heads':>7} {'N':>3}  {'Accuracy':>18}  {'AUROC macro':>18}")
    print("-" * 72)
    for row in agg:
        acc = f"{row['linear_probe_accuracy_mean']:.4f} ± {row['linear_probe_accuracy_std']:.4f}"
        auroc = f"{row['linear_probe_auroc_macro_mean']:.4f} ± {row['linear_probe_auroc_macro_std']:.4f}"
        print(f"{row['model_short']:<14} {int(row['n_heads']):>7} {int(row['n_seeds']):>3}  {acc:>18}  {auroc:>18}")

    if args.plot:
        print("\nGenerating plots...")
        plot_results(agg, args.output_dir / "plots")


if __name__ == "__main__":
    main()
