"""
Run AE training under 5 validation strategies for anomaly detection.

Strategy A — mse_qcd (pure unsupervised):
    - val_signal_classes=[] → no signals during validation
    - Monitor: ae/val_loss_normal (mode=min)
    - Best checkpoint = lowest QCD reconstruction MSE on val

Strategy B — auroc_all_signals (semi-supervised, all signals):
    - val_signal_classes=[15, 16, 17] → all 3 Higgs signals visible in val
    - Monitor: ae/val_auroc (mode=max)

Strategy C1/C2/C3 — auroc_clsN (semi-supervised, one signal at a time):
    - val_signal_classes=[cls] for cls in {15, 16, 17}
    - Monitor: ae/val_auroc (mode=max)
    - Test still evaluates all 3 signals

All 5 sub-experiments share the same pre-extracted embeddings (Phase 1 frozen).

Saved per strategy:
    MSE (mean±std) for QCD, cls15, cls16, cls17
    Drift on test QCD at FPR 1%/5%/10% (threshold derived from val QCD)
    Cut τ at FPR 1%/5%/10%
    Measured FPR at each cut
    FNR (1-TPR) per signal at each cut
    AUROC per signal and overall
    Best epoch (0-indexed, from checkpoint filename)

Usage:
    python scripts/run_anomaly_strategies.py \\
        --phase2-experiment anomaly_qcd_vs_higgs_embedding_augsupcon_best_cern \\
        --embeddings-dir /eos/user/d/dgenoves/anomaly_pipeline/besttrial2/aug_supcon_15class_best_cern/embeddings \\
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/strategies_v2/aug_supcon \\
        --raw-experiment anomaly_qcd_vs_higgs_raw_cern \\
        --model-tag aug_supcon
"""

import argparse
import json
import os
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import gc
import torch
import lightning as L

from src.train_full_anomaly_pipeline import run_phase2, run_raw_ae_baseline

def _build_strategies(signal_classes):
    strategies = [
        {
            "name": "mse_qcd",
            "description": "Pure unsupervised: monitor QCD val MSE (no signals in val)",
            "val_signal_classes": [],
            "monitor": "ae/val_loss_normal",
            "score_classes": None,
        },
        {
            "name": "mse_allsm_qcdthresh",
            "description": "Allsm training, QCD-only threshold: monitor QCD MSE, threshold from QCD only",
            "val_signal_classes": [],
            "monitor": "ae/val_loss_score",
            "score_classes": [0],
        },
        {
            "name": "auroc_all_signals",
            "description": f"Semi-supervised: AUROC monitor, all signals {signal_classes} in val",
            "val_signal_classes": list(signal_classes),
            "monitor": "ae/val_auroc",
            "score_classes": None,
        },
    ]
    for cls in signal_classes:
        strategies.append({
            "name": f"auroc_cls{cls}",
            "description": f"Semi-supervised: AUROC monitor, only cls{cls} in val",
            "val_signal_classes": [cls],
            "monitor": "ae/val_auroc",
            "score_classes": None,
        })
    return strategies


STRATEGIES = _build_strategies([15, 16, 17])


def _run_strategy(run_fn, strategy, output_dir, **kwargs):
    """Run a single strategy and return the result dict or an error entry."""
    name = strategy["name"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        metrics = run_fn(
            output_dir=output_dir,
            monitor=strategy["monitor"],
            val_signal_classes=strategy["val_signal_classes"],
            **kwargs,
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
            f"  [OK] {name}: epoch={metrics.get('best_epoch')}  "
            f"auroc_all={ps.get('auroc_all', float('nan')):.4f}  "
            f"drift={metrics.get('drift_metric', float('nan')):.4f}"
        )
        return result
    except Exception as exc:
        import traceback
        print(f"  [FAIL] {name}: {exc}")
        traceback.print_exc()
        return {
            "description": strategy["description"],
            "monitor": strategy["monitor"],
            "val_signal_classes": strategy["val_signal_classes"],
            "status": "error",
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run AE training under 5 anomaly-detection strategies"
    )
    parser.add_argument(
        "--phase2-experiment",
        required=True,
        help="Hydra experiment name for Phase 2 (e.g. anomaly_qcd_vs_higgs_embedding_augsupcon_best_cern)",
    )
    parser.add_argument(
        "--embeddings-dir",
        required=False,
        default=None,
        help="Directory containing pre-extracted train/val/test_embeddings.npz",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for strategy sub-experiment outputs",
    )
    parser.add_argument(
        "--model-tag",
        default="model",
        help="Short tag for the encoder model used (for logging only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--signal-classes",
        nargs="+",
        type=int,
        default=[15, 16, 17],
        metavar="CLS",
        help="Class indices for anomaly signals (default: 15 16 17)",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        default=False,
        help="Skip embedding-based strategies (preserves existing results if summary exists)",
    )
    parser.add_argument(
        "--raw-experiment",
        default=None,
        help="If set, also run all strategies on raw input features (no encoder). "
             "e.g. anomaly_qcd_vs_higgs_raw_cern",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        default=False,
        help="Skip raw baseline strategies (preserves existing raw_results if summary exists)",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Subset of strategy names to run (default: all 5)",
    )
    parser.add_argument(
        "--score-classes",
        nargs="+",
        type=int,
        default=None,
        metavar="CLS",
        help="If set, override score_classes for ALL strategies (e.g. --score-classes 0 "
             "for QCD-only threshold/AUROC when AE is trained on all SM classes).",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)
    L.seed_everything(args.seed, workers=True)

    if args.embeddings_dir is None and not args.skip_embedding:
        raise ValueError("--embeddings-dir is required unless --skip-embedding is set")
    embedding_dir = Path(args.embeddings_dir) if args.embeddings_dir else Path("/dev/null")
    if not args.skip_embedding and not embedding_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embedding_dir}")

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_strategies = _build_strategies(args.signal_classes)

    # Override score_classes for all strategies when --score-classes is passed.
    # This forces threshold and AUROC to be computed on the specified classes only
    # (e.g. QCD-only when AE is trained on all SM classes).
    if args.score_classes is not None:
        for s in all_strategies:
            s["score_classes"] = args.score_classes
            # mse_qcd monitors normal-class MSE; when score_classes is set we
            # want to monitor only those classes, so switch to val_loss_score.
            if s["name"] == "mse_qcd":
                s["monitor"] = "ae/val_loss_score"

    strategies_to_run = all_strategies
    if args.strategies:
        allowed = set(args.strategies)
        strategies_to_run = [s for s in all_strategies if s["name"] in allowed]
        if not strategies_to_run:
            raise ValueError(f"No matching strategies for: {args.strategies}")

    print(f"\n{'='*80}")
    print(f"  Anomaly Strategy Comparison  |  model: {args.model_tag}")
    print(f"  Embeddings : {embedding_dir}")
    print(f"  Output     : {output_dir}")
    print(f"  Strategies : {[s['name'] for s in strategies_to_run]}")
    print(f"{'='*80}\n")

    # Load existing summary if present (to preserve results when skipping)
    summary_path = output_dir / "strategies_summary.json"
    summary = {
        "model_tag": args.model_tag,
        "phase2_experiment": args.phase2_experiment,
        "embeddings_dir": str(embedding_dir),
        "seed": args.seed,
        "results": {},
        "raw_results": {},
    }
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        summary["results"]     = existing.get("results", {})
        summary["raw_results"] = existing.get("raw_results", {})

    # ── Embedding-based strategies ────────────────────────────────────────────
    if args.skip_embedding:
        print("  [--skip-embedding] Skipping embedding-based strategies.")
    else:
        for strategy in strategies_to_run:
            name = strategy["name"]
            print(f"\n{'─'*80}")
            print(f"  Running: {name}  —  {strategy['description']}")
            print(f"{'─'*80}")
            strategy_output = output_dir / name
            os.makedirs(strategy_output, exist_ok=True)
            summary["results"][name] = _run_strategy(
                run_fn=lambda output_dir, monitor, val_signal_classes: run_phase2(
                    phase2_experiment=args.phase2_experiment,
                    embedding_dir=embedding_dir,
                    output_dir=output_dir,
                    monitor=monitor,
                    val_signal_classes=val_signal_classes,
                ),
                strategy=strategy,
                output_dir=strategy_output,
            )
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

    # ── Raw AE baseline ───────────────────────────────────────────────────────
    if args.raw_experiment:
        if args.skip_raw:
            print("  [--skip-raw] Skipping raw baseline strategies.")
        else:
            print(f"\n{'='*80}")
            print(f"  Raw AE Baseline  |  experiment: {args.raw_experiment}")
            print(f"{'='*80}\n")
            summary["raw_experiment"] = args.raw_experiment

            for strategy in strategies_to_run:
                name = strategy["name"]
                print(f"\n{'─'*80}")
                print(f"  [RAW] Running: {name}  —  {strategy['description']}")
                print(f"{'─'*80}")
                strategy_output = output_dir / "raw_baseline" / name
                os.makedirs(strategy_output, exist_ok=True)
                summary["raw_results"][name] = _run_strategy(
                    run_fn=lambda output_dir, monitor, val_signal_classes, score_classes=strategy.get("score_classes"): run_raw_ae_baseline(
                        experiment=args.raw_experiment,
                        output_dir=output_dir,
                        monitor=monitor,
                        val_signal_classes=val_signal_classes,
                        score_classes=score_classes,
                    ),
                    strategy=strategy,
                    output_dir=strategy_output,
                )
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  Strategy comparison saved to: {summary_path}")
    print(f"{'='*80}\n")

    _print_results(f"Embedding-based ({args.model_tag})", summary["results"], args.signal_classes)
    if args.raw_experiment:
        _print_results("Raw baseline", summary.get("raw_results", {}), args.signal_classes)


def _print_results(label: str, results: dict, signal_classes: list = None):
    if signal_classes is None:
        signal_classes = [15, 16, 17]
    cls_headers = " ".join(f"{'AUC_c' + str(c):>8}" for c in signal_classes)
    print(f"\n{label}")
    hdr = (f"{'Strategy':<22} {'Monitor':<22} {'epoch':>5} "
           f"{cls_headers} {'AUC_all':>8} "
           f"{'drift':>8} {'MSE_QCD':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, res in results.items():
        if res.get("status") != "ok":
            print(f"{name:<22} ERROR: {res.get('error', '?')}")
            continue
        ps = res.get("per_signal", {})
        cls_values = " ".join(f"{ps.get(f'auroc_cls{c}', float('nan')):>8.4f}" for c in signal_classes)
        print(
            f"{name:<22} {res['monitor']:<22} {str(res.get('best_epoch', '?')):>5} "
            f"{cls_values} "
            f"{ps.get('auroc_all',   float('nan')):>8.4f} "
            f"{res.get('drift_metric', float('nan')):>8.4f} "
            f"{ps.get('mse_mean_cls0', float('nan')):>8.4f}"
        )


if __name__ == "__main__":
    main()
