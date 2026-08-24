"""
Run anomaly detection strategies across multiple seeds and aggregate results.

For each seed:
  - Runs all 5 strategies for embedding AE (AugSupCon frozen) and raw AE
  - Seeds are re-applied before each strategy run for reproducibility
  - Per-seed summary saved to {output_dir}/seed{N}/strategies_summary.json

After all seeds:
  - Computes mean ± std across seeds for all metrics
  - Saves aggregated summary to {output_dir}/aggregated_summary.json

Supports resume: seeds whose summary.json already exists are skipped.

Usage:
    python scripts/run_anomaly_seeds.py \\
        --phase2-experiment anomaly_qcd_vs_higgs_embedding_augsupcon_best_cern \\
        --embeddings-dir /eos/user/d/dgenoves/anomaly_pipeline/besttrial2/aug_supcon_15class_best_cern/embeddings \\
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/strategies_v2/aug_supcon_seeds \\
        --raw-experiment anomaly_qcd_vs_higgs_raw_cern \\
        --seeds 0 1 2 3 4
"""

import argparse
import json
import gc
import math
import os
from collections import defaultdict
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import lightning as L

from src.train_full_anomaly_pipeline import run_phase2, run_raw_ae_baseline

STRATEGIES = [
    {
        "name": "mse_qcd",
        "description": "Pure unsupervised: monitor QCD val MSE (no signals in val)",
        "val_signal_classes": [],
        "monitor": "ae/val_loss_normal",
    },
    {
        "name": "auroc_all_signals",
        "description": "Semi-supervised: AUROC monitor, all 3 Higgs in val",
        "val_signal_classes": [15, 16, 17],
        "monitor": "ae/val_auroc",
    },
    {
        "name": "auroc_cls15",
        "description": "Semi-supervised: AUROC monitor, only VBFHbb (cls15) in val",
        "val_signal_classes": [15],
        "monitor": "ae/val_auroc",
    },
    {
        "name": "auroc_cls16",
        "description": "Semi-supervised: AUROC monitor, only HH4b (cls16) in val",
        "val_signal_classes": [16],
        "monitor": "ae/val_auroc",
    },
    {
        "name": "auroc_cls17",
        "description": "Semi-supervised: AUROC monitor, only ggHtautau (cls17) in val",
        "val_signal_classes": [17],
        "monitor": "ae/val_auroc",
    },
]


def _run_one(run_fn, strategy, output_dir, seed):
    """Run a single strategy. Re-seeds before training for reproducibility."""
    L.seed_everything(seed, workers=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    name = strategy["name"]
    try:
        metrics = run_fn(
            output_dir=output_dir,
            monitor=strategy["monitor"],
            val_signal_classes=strategy["val_signal_classes"],
        )
        ps = metrics.get("per_signal", {})
        result = {
            "description": strategy["description"],
            "monitor": strategy["monitor"],
            "val_signal_classes": strategy["val_signal_classes"],
            "status": "ok",
            "best_epoch": metrics.get("best_epoch"),
            "separation_ratio": metrics.get("separation_ratio", float("nan")),
            "drift_metric": metrics.get("drift_metric", float("nan")),
            "drift_fpr01": metrics.get("drift_fpr01", float("nan")),
            "drift_fpr05": metrics.get("drift_fpr05", float("nan")),
            "drift_fpr10": metrics.get("drift_fpr10", float("nan")),
            "per_signal": ps,
        }
        print(
            f"    [OK] {name}: epoch={metrics.get('best_epoch')}  "
            f"auroc_all={ps.get('auroc_all', float('nan')):.4f}  "
            f"drift={metrics.get('drift_metric', float('nan')):.4f}"
        )
        return result
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"    [FAIL] {name}: {exc}")
        return {
            "description": strategy["description"],
            "monitor": strategy["monitor"],
            "val_signal_classes": strategy["val_signal_classes"],
            "status": "error",
            "error": str(exc),
        }


def run_one_seed(seed, args, output_dir):
    """Run all strategies for a single seed. Returns the summary dict."""
    summary_path = output_dir / "strategies_summary.json"

    # Resume: skip if already done (all strategies ok)
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        emb_ok = all(
            existing.get("results", {}).get(s["name"], {}).get("status") == "ok"
            for s in STRATEGIES
        )
        raw_ok = (not args.raw_experiment) or all(
            existing.get("raw_results", {}).get(s["name"], {}).get("status") == "ok"
            for s in STRATEGIES
        )
        if emb_ok and raw_ok:
            print(f"  [seed {seed}] Already complete — skipping.")
            return existing

    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "seed": seed,
        "model_tag": args.model_tag,
        "phase2_experiment": args.phase2_experiment,
        "results": {},
        "raw_results": {},
    }

    embedding_dir = Path(args.embeddings_dir)

    # Embedding-based strategies
    print(f"\n  [seed {seed}] Embedding strategies ─────────────────────────────")
    for strategy in STRATEGIES:
        name = strategy["name"]
        strat_out = output_dir / name
        os.makedirs(strat_out, exist_ok=True)

        def _phase2(output_dir, monitor, val_signal_classes):
            return run_phase2(
                phase2_experiment=args.phase2_experiment,
                embedding_dir=embedding_dir,
                output_dir=output_dir,
                monitor=monitor,
                val_signal_classes=val_signal_classes,
            )

        summary["results"][name] = _run_one(_phase2, strategy, strat_out, seed)

    # Raw AE strategies
    if args.raw_experiment:
        print(f"\n  [seed {seed}] Raw AE strategies ────────────────────────────────")
        for strategy in STRATEGIES:
            name = strategy["name"]
            strat_out = output_dir / "raw_baseline" / name
            os.makedirs(strat_out, exist_ok=True)

            def _raw(output_dir, monitor, val_signal_classes):
                return run_raw_ae_baseline(
                    experiment=args.raw_experiment,
                    output_dir=output_dir,
                    monitor=monitor,
                    val_signal_classes=val_signal_classes,
                )

            summary["raw_results"][name] = _run_one(_raw, strategy, strat_out, seed)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ─── Aggregation ─────────────────────────────────────────────────────────────

def _mean_std(values):
    vals = [v for v in values if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(vals) / n
    if n == 1:
        return mu, float("nan")
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var)


def aggregate(seed_summaries, output_dir, args):
    """Aggregate per-seed summaries into mean ± std for all metrics."""

    def _collect(approach_key):
        # strategy → metric_key → [values across seeds]
        collected = defaultdict(lambda: defaultdict(list))
        for s in seed_summaries:
            for strat_name, res in s.get(approach_key, {}).items():
                if res.get("status") != "ok":
                    continue
                # Top-level scalar metrics
                for key in ("separation_ratio", "drift_metric",
                            "drift_fpr01", "drift_fpr05", "drift_fpr10", "best_epoch"):
                    val = res.get(key)
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        collected[strat_name][key].append(float(val))
                # per_signal metrics
                for k, v in res.get("per_signal", {}).items():
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                        collected[strat_name][f"per_signal.{k}"].append(float(v))
        return collected

    def _build_agg(collected):
        agg = {}
        for strat_name, metrics in collected.items():
            agg[strat_name] = {}
            for key, values in metrics.items():
                mu, std = _mean_std(values)
                agg[strat_name][key] = {"mean": mu, "std": std, "n": len(values), "values": values}
        return agg

    emb_collected = _collect("results")
    raw_collected  = _collect("raw_results")

    aggregated = {
        "model_tag": args.model_tag,
        "phase2_experiment": args.phase2_experiment,
        "n_seeds": len(seed_summaries),
        "seeds": [s["seed"] for s in seed_summaries],
        "results": _build_agg(emb_collected),
        "raw_results": _build_agg(raw_collected),
    }

    out_path = output_dir / "aggregated_summary.json"
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n  Aggregated summary saved to: {out_path}")

    _print_aggregated(aggregated)
    return aggregated


def _print_aggregated(agg):
    KEY_METRICS = [
        ("per_signal.auroc_cls15", "AUC_c15"),
        ("per_signal.auroc_cls16", "AUC_c16"),
        ("per_signal.auroc_cls17", "AUC_c17"),
        ("per_signal.auroc_all",   "AUC_all"),
        ("drift_metric",           "drift"),
        ("per_signal.mse_mean_cls0", "MSE_QCD"),
        ("best_epoch",             "epoch"),
    ]

    def _print_block(label, results):
        print(f"\n{label}  (n={agg['n_seeds']} seeds)")
        hdr = f"  {'Strategy':<22}" + "".join(f"  {tag:>18}" for _, tag in KEY_METRICS)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for strat_name, metrics in results.items():
            row = f"  {strat_name:<22}"
            for key, _ in KEY_METRICS:
                entry = metrics.get(key, {})
                mu  = entry.get("mean", float("nan"))
                std = entry.get("std",  float("nan"))
                if math.isnan(std):
                    cell = f"{mu:.4f}{'':>7}"
                else:
                    cell = f"{mu:.4f}±{std:.4f}"
                row += f"  {cell:>18}"
            print(row)

    _print_block(f"Embedding AE ({agg['model_tag']})", agg["results"])
    if agg.get("raw_results"):
        _print_block("Raw AE", agg["raw_results"])


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run anomaly strategies across multiple seeds and aggregate results"
    )
    parser.add_argument("--phase2-experiment", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-experiment", default=None)
    parser.add_argument("--model-tag", default="model")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="List of seeds to run (default: 0 1 2 3 4)",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    embedding_dir = Path(args.embeddings_dir)
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embeddings dir not found: {embedding_dir}")

    print(f"\n{'='*80}")
    print(f"  Multi-seed anomaly run  |  model: {args.model_tag}")
    print(f"  Seeds      : {args.seeds}")
    print(f"  Embeddings : {embedding_dir}")
    print(f"  Output     : {output_dir}")
    print(f"  Strategies : {[s['name'] for s in STRATEGIES]}")
    print(f"{'='*80}\n")

    seed_summaries = []
    for seed in args.seeds:
        print(f"\n{'─'*80}")
        print(f"  SEED {seed}")
        print(f"{'─'*80}")
        seed_dir = output_dir / f"seed{seed}"
        summary = run_one_seed(seed, args, seed_dir)
        seed_summaries.append(summary)

    print(f"\n{'='*80}")
    print(f"  All seeds done — aggregating...")
    print(f"{'='*80}")
    aggregate(seed_summaries, output_dir, args)


if __name__ == "__main__":
    main()
