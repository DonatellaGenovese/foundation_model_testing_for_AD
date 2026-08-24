"""
Run AE anomaly detection across all encoder seeds.

For each encoder seed N:
  1. Find checkpoint in encoder_seeds_dir/seed_N/checkpoints/
  2. Load encoder model
  3. Extract embeddings to output_dir/encoder_seed_N/embeddings/
  4. Train AE on QCD embeddings, evaluate AUROC on BSM anomalies
  5. Save results to output_dir/encoder_seed_N/summary.json

Aggregates mean ± std across encoder seeds → output_dir/aggregated_summary.json

Supports resume: encoder seeds whose summary.json already exists are skipped.

Usage:
    python scripts/run_encoder_seeds_anomaly.py \\
        --phase2-experiment anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern \\
        --encoder-seeds-dir /eos/user/d/dgenoves/anomaly_pipeline/new_exp/vcreg_12class_nosparse_dmodel256_cern \\
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/ad_encoder_seeds/augsupcon_nosparse_dmodel128
"""

import argparse
import collections
import functools
import gc
import json
import math
import os
import typing
from importlib import import_module
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import omegaconf
import lightning as L

torch.serialization.add_safe_globals([
    functools.partial,
    torch.optim.AdamW, torch.optim.Adam,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    # OneCycleLR is what the vicreg encoders train with, and Lightning
    # pickles the scheduler object into the checkpoint. Without it here, loading
    # those encoders fails under PyTorch 2.6's weights_only default — which is
    # exactly what killed the post-training test step of that campaign.
    torch.optim.lr_scheduler.OneCycleLR,
    omegaconf.ListConfig, omegaconf.DictConfig, omegaconf.dictconfig.DictConfig,
    omegaconf.nodes.AnyNode, omegaconf.base.Metadata, omegaconf.base.ContainerMetadata,
    collections.defaultdict, typing.Any,
    list, dict, int,
])
torch.set_float32_matmul_precision('high')

from src.train_full_anomaly_pipeline import extract_and_save_embeddings, run_phase2, _compose_cfg
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def find_best_ckpt(seed_dir: Path) -> Path:
    """Return the best checkpoint (epoch_XXX.ckpt saved by ModelCheckpoint),
    falling back to last.ckpt only if no epoch checkpoint exists."""
    ckpt_dir = seed_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints dir: {ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt"))
    if ckpts:
        return ckpts[-1]   # ModelCheckpoint saves only the best (save_top_k=1)
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def load_encoder(ckpt_path: Path, model_class_str: str) -> torch.nn.Module:
    module_path, class_name = model_class_str.rsplit(".", 1)
    model_cls = getattr(import_module(module_path), class_name)
    model = model_cls.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    return model


def _mean_std(values):
    vals = [v for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan")
    mu = sum(vals) / len(vals)
    if len(vals) == 1:
        return mu, float("nan")
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return mu, math.sqrt(var)


def run_one_encoder_seed(
    encoder_seed: int,
    ckpt_path: Path,
    model_class_str: str,
    phase2_experiment: str,
    output_dir: Path,
    ae_seed: int,
    anomaly_classes: list,
    strategies_filter: list = None,
) -> dict:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"  [encoder_seed {encoder_seed}] Already done — loading {summary_path}")
        with open(summary_path) as f:
            return json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    all_strategies = [
        {"name": "mse_normal",        "val_signal_classes": [],              "monitor": "ae/val_loss_normal"},
        {"name": "auroc_all_signals", "val_signal_classes": anomaly_classes, "monitor": "ae/val_auroc"},
    ] + [
        {"name": f"auroc_cls{c}", "val_signal_classes": [c], "monitor": "ae/val_auroc"}
        for c in anomaly_classes
    ]
    if strategies_filter:
        allowed = set(strategies_filter)
        strategies = [s for s in all_strategies if s["name"] in allowed]
    else:
        strategies = all_strategies

    # Step 1 — extract embeddings
    embedding_dir = output_dir / "embeddings"
    if not (embedding_dir / "train_embeddings.npz").exists():
        print(f"  [encoder_seed {encoder_seed}] Loading encoder from {ckpt_path.name}")
        model = load_encoder(ckpt_path, model_class_str)
        extract_and_save_embeddings(
            model=model,
            phase2_experiment=phase2_experiment,
            output_dir=embedding_dir,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"  [encoder_seed {encoder_seed}] Embeddings already extracted")

    # Step 2 — run AE for each strategy
    results = {}
    for strategy in strategies:
        name = strategy["name"]
        strat_out = output_dir / name
        strat_result_path = strat_out / "result.json"

        if strat_result_path.exists():
            with open(strat_result_path) as f:
                results[name] = json.load(f)
            print(f"  [encoder_seed {encoder_seed}][{name}] Already done")
            continue

        os.makedirs(strat_out, exist_ok=True)
        print(f"  [encoder_seed {encoder_seed}][{name}] Training AE...")
        L.seed_everything(ae_seed, workers=True)
        try:
            res = run_phase2(
                phase2_experiment=phase2_experiment,
                embedding_dir=embedding_dir,
                output_dir=strat_out,
                monitor=strategy["monitor"],
                val_signal_classes=strategy["val_signal_classes"],
            )
            res["status"] = "ok"
        except Exception as e:
            print(f"  [encoder_seed {encoder_seed}][{name}] ERROR: {e}")
            res = {"status": "error", "error": str(e)}

        results[name] = res
        with open(strat_result_path, "w") as f:
            json.dump(res, f, indent=2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "encoder_seed": encoder_seed,
        "ckpt_path": str(ckpt_path),
        "ae_seed": ae_seed,
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def aggregate(summaries: list, output_dir: Path) -> dict:
    collected = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in summaries:
        for strat_name, res in s.get("results", {}).items():
            if res.get("status") != "ok":
                continue
            for k in ("separation_ratio", "drift_metric",
                      "drift_fpr01", "drift_fpr05", "drift_fpr10"):
                if k in res and res[k] is not None:
                    collected[strat_name][k].append(float(res[k]))
            for k, v in res.get("per_signal", {}).items():
                if isinstance(v, (int, float)):
                    collected[strat_name][f"per_signal.{k}"].append(float(v))

    agg = {}
    for strat_name, metrics in collected.items():
        agg[strat_name] = {
            key: {"mean": _mean_std(vals)[0], "std": _mean_std(vals)[1],
                  "n": len(vals), "values": vals}
            for key, vals in metrics.items()
        }

    result = {"aggregated": agg, "n_encoder_seeds": len(summaries)}
    with open(output_dir / "aggregated_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  Aggregated results — {len(summaries)} encoder seeds")
    print(f"{'='*80}")
    for strat_name, metrics in agg.items():
        print(f"\n  [{strat_name}]")
        for k, v in metrics.items():
            if "fpr" in k:
                continue
            mu, std = v["mean"], v["std"]
            std_str = f"±{std:.4f}" if not math.isnan(std) else ""
            print(f"    {k:40s}: {mu:.4f}{std_str}  (n={v['n']})")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-experiment", required=True,
                        help="Hydra experiment name for phase 2 AE config")
    parser.add_argument("--encoder-seeds-dir", required=True,
                        help="Dir containing seed_0/, seed_1/, ... with encoder checkpoints")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory; encoder_seed_N/ subdirs created here")
    parser.add_argument("--ae-seed", type=int, default=42,
                        help="Fixed seed for AE training (default: 42)")
    parser.add_argument("--encoder-seeds", nargs="+", type=int, default=list(range(10)),
                        help="Which encoder seeds to process (default: 0..9)")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategies to run (default: all). e.g. --strategies mse_normal")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    encoder_seeds_dir = Path(args.encoder_seeds_dir)

    cfg = _compose_cfg("anomaly_detection.yaml",
                       [f"experiment={args.phase2_experiment}"],
                       output_dir=output_dir)
    model_class_str = cfg.get("model_class")
    if not model_class_str:
        raise ValueError(f"Config '{args.phase2_experiment}' must define 'model_class'")
    anomaly_classes = list(cfg.get("anomaly_classes", []))

    print(f"\n{'='*80}")
    print(f"  Encoder-seeds anomaly pipeline")
    print(f"  Phase2 experiment : {args.phase2_experiment}")
    print(f"  Encoder seeds dir : {encoder_seeds_dir}")
    print(f"  Encoder seeds     : {args.encoder_seeds}")
    print(f"  AE seed           : {args.ae_seed}")
    print(f"  Model class       : {model_class_str}")
    print(f"  Anomaly classes   : {anomaly_classes}")
    print(f"  Output dir        : {output_dir}")
    print(f"{'='*80}\n")

    summaries = []
    for enc_seed in args.encoder_seeds:
        seed_input_dir = encoder_seeds_dir / f"seed_{enc_seed}"
        seed_output_dir = output_dir / f"encoder_seed_{enc_seed}"

        print(f"\n{'─'*80}")
        print(f"  ENCODER SEED {enc_seed}")
        print(f"{'─'*80}")

        try:
            ckpt_path = find_best_ckpt(seed_input_dir)
            print(f"  Checkpoint: {ckpt_path}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        summary = run_one_encoder_seed(
            encoder_seed=enc_seed,
            ckpt_path=ckpt_path,
            model_class_str=model_class_str,
            phase2_experiment=args.phase2_experiment,
            output_dir=seed_output_dir,
            ae_seed=args.ae_seed,
            anomaly_classes=anomaly_classes,
            strategies_filter=args.strategies,
        )
        summaries.append(summary)

    if summaries:
        aggregate(summaries, output_dir)

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
