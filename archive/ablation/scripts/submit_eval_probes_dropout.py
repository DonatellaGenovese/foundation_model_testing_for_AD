#!/usr/bin/env python3
"""
Submit linear probe evaluation jobs for all dropout ablation training runs.

Standard d=128 architecture (n_heads=8, n_layers=6, d_ff=512) for all 4 models.
dropout in {0.1, 0.2, 0.3} × 5 seeds = 60 runs.

Probe output: inside each training run at {run}/probe_evaluation/

Usage:
    python3 scripts/ablation/submit_eval_probes_dropout.py [--dry-run]
    python3 scripts/ablation/submit_eval_probes_dropout.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_eval_probes_dropout.py --dropouts 20 30
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

EOS_TRAIN_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/dropout/logs/train/runs"
PROJECT_DIR    = Path(__file__).resolve().parents[2]
LOG_DIR        = PROJECT_DIR / "logs/condor_logs/ablation/eval_probes_dropout"
SUB_DIR        = PROJECT_DIR / "logs/condor_subs/ablation"
WRAPPER        = PROJECT_DIR / "scripts/ablation/wrapper_eval_probe_ablation.sh"

SEEDS = [7, 42, 137, 1337, 31337]

# Map (model_short, dropout×10) -> experiment config name
EXPERIMENT_MAP = {
    ("VCReg",               10):  "ablation/training/vcreg_dropout_10",
    ("VCReg",               20):  "ablation/training/vcreg_dropout_20",
    ("VCReg",               30):  "ablation/training/vcreg_dropout_30",
    ("AugmentedSupCon",     10):  "ablation/training/aug_supcon_dropout_10",
    ("AugmentedSupCon",     20):  "ablation/training/aug_supcon_dropout_20",
    ("AugmentedSupCon",     30):  "ablation/training/aug_supcon_dropout_30",
    ("AugmentedSelfSupCon", 10):  "ablation/training/selfsupcon_dropout_10",
    ("AugmentedSelfSupCon", 20):  "ablation/training/selfsupcon_dropout_20",
    ("AugmentedSelfSupCon", 30):  "ablation/training/selfsupcon_dropout_30",
    ("VICReg",              10):  "ablation/training/vicreg_dropout_10",
    ("VICReg",              20):  "ablation/training/vicreg_dropout_20",
    ("VICReg",              30):  "ablation/training/vicreg_dropout_30",
}

CLI_TO_MODEL = {
    "vcreg":      "VCReg",
    "aug_supcon": "AugmentedSupCon",
    "selfsupcon": "AugmentedSelfSupCon",
    "vicreg":     "VICReg",
}


def get_run_info(run_dir: Path):
    cfg = run_dir / ".hydra" / "config.yaml"
    if not cfg.exists():
        return None
    try:
        text = cfg.read_text()
        model = next(
            (l.split("_target_:")[-1].strip().split(".")[-1]
             .replace("COLLIDE2V", "").replace("LitModule", "")
             for l in text.split("\n") if "_target_:" in l and "LitModule" in l),
            None,
        )
        dropout = next(
            (float(l.split("dropout:")[-1].strip())
             for l in text.split("\n") if "  dropout:" in l),
            None,
        )
        seed = next(
            (int(l.split("seed:")[-1].strip())
             for l in text.split("\n") if l.strip().startswith("seed:")),
            None,
        )
        if not model or dropout is None or seed is None:
            return None
        return {"model": model, "dropout10": round(dropout * 100), "seed": seed, "run_dir": run_dir}
    except Exception:
        return None


def find_best_checkpoint(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt")) if ckpt_dir.exists() else []
    return str(ckpts[-1]) if ckpts else None


def already_done(run_dir: Path) -> bool:
    return (run_dir / "probe_evaluation" / "probe_results.json").exists()


def make_sub(run_info: dict, experiment: str, ckpt_path: str, dry_run: bool) -> None:
    model_short = run_info["model"].replace("Augmented", "").lower()
    job_name = f"{model_short}_dropout{run_info['dropout10']}_seed{run_info['seed']}"
    output_dir = str(run_info["run_dir"])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = SUB_DIR / f"eval_probe_dropout_{job_name}.sub"
    sub_content = textwrap.dedent(f"""\
        executable = {WRAPPER}
        arguments  = $(ClusterId)

        output = {LOG_DIR}/{job_name}_$(ClusterId).out
        error  = {LOG_DIR}/{job_name}_$(ClusterId).err
        log    = {LOG_DIR}/{job_name}_$(ClusterId).log

        run_as_owner = True
        +JobFlavour  = "nextweek"
        getenv       = True
        request_cpus = 8
        request_gpus = 1
        Requirements = (TARGET.GPUs_GlobalMemoryMb >= 16000)

        environment = "EXPERIMENT={experiment} CKPT_PATH={ckpt_path} SEED={run_info['seed']} OUTPUT_DIR={output_dir}"

        queue
    """)

    sub_path.write_text(sub_content)
    print(f"  {'[DRY]' if dry_run else '[SUB]'} {job_name}  ckpt={Path(ckpt_path).name}")

    if not dry_run:
        result = subprocess.run(
            ["condor_submit", str(sub_path)],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr.strip()}")
        else:
            print(f"    → {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(CLI_TO_MODEL), default=None)
    parser.add_argument("--dropouts", nargs="+", type=int, choices=[10, 20, 30], default=None,
                        help="Dropout × 10: 10=0.1, 20=0.2, 30=0.3")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    filter_models   = {CLI_TO_MODEL[m] for m in args.models} if args.models else None
    filter_dropouts = set(args.dropouts) if args.dropouts else None
    filter_seeds    = set(args.seeds) if args.seeds else None

    train_base = Path(EOS_TRAIN_BASE)
    all_runs = [get_run_info(p) for p in sorted(train_base.iterdir()) if p.is_dir()]
    all_runs = [r for r in all_runs if r is not None]

    if filter_models:
        all_runs = [r for r in all_runs if r["model"] in filter_models]
    if filter_dropouts:
        all_runs = [r for r in all_runs if r["dropout10"] in filter_dropouts]
    if filter_seeds:
        all_runs = [r for r in all_runs if r["seed"] in filter_seeds]

    # For duplicate (model, dropout, seed) keep the one with the best checkpoint
    best: dict[tuple, dict] = {}
    for r in all_runs:
        key = (r["model"], r["dropout10"], r["seed"])
        ckpt = find_best_checkpoint(r["run_dir"])
        if ckpt and key not in best:
            r["ckpt"] = ckpt
            best[key] = r

    print(f"{'DRY RUN — ' if args.dry_run else ''}Found {len(best)} runs to evaluate\n")

    submitted = skipped = missing = 0
    for key in sorted(best):
        r = best[key]
        experiment = EXPERIMENT_MAP.get((r["model"], r["dropout10"]))
        if experiment is None:
            print(f"  [WARN] No experiment config for {r['model']} dropout={r['dropout10']/10}")
            missing += 1
            continue

        if already_done(r["run_dir"]):
            print(f"  [SKIP] {r['model']} dropout={r['dropout10']/10:.1f} seed={r['seed']} — already done")
            skipped += 1
            continue

        make_sub(r, experiment, r["ckpt"], dry_run=args.dry_run)
        submitted += 1

    print(f"\nDone: {submitted} submitted, {skipped} skipped, {missing} missing config.")


if __name__ == "__main__":
    main()
