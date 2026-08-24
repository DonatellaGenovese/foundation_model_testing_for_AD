#!/usr/bin/env python3
"""
Submit linear probe evaluation jobs for all d_model ablation training runs.

Covers d_model in {32, 64, 128, 256, 512} for VCReg, AugSupCon, SelfSupCon
(5 seeds each = 75 runs total). d=128 runs were copied from the loss ablation
with compatible hyperparameters (var=25/cov=1 for VCReg, cw050/t=0.1 for
AugSupCon, cw0/t=0.05 for SelfSupCon).

Eval logs:  ablation/d_model/logs/eval/runs/{timestamp}/
Probe output: inside each training run at {run}/probe_evaluation/

Usage:
    python3 scripts/ablation/submit_eval_probes_dmodel.py [--dry-run]
    python3 scripts/ablation/submit_eval_probes_dmodel.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_eval_probes_dmodel.py --dmodels 128 256
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

EOS_TRAIN_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs"
PROJECT_DIR    = Path(__file__).resolve().parents[2]
LOG_DIR        = PROJECT_DIR / "logs/condor_logs/ablation/eval_probes_dmodel"
SUB_DIR        = PROJECT_DIR / "logs/condor_subs/ablation"
WRAPPER        = PROJECT_DIR / "scripts/ablation/wrapper_eval_probe_ablation.sh"

SEEDS = [7, 42, 137, 1337, 31337]

# Map (model_short, d_model) -> experiment config name
EXPERIMENT_MAP = {
    ("VCReg",               32):  "ablation/training/vcreg_dmodel32",
    ("VCReg",               64):  "ablation/training/vcreg_dmodel64",
    ("VCReg",              128):  "ablation/training/vcreg_dmodel128",
    ("VCReg",              256):  "ablation/training/vcreg_dmodel256",
    ("VCReg",              512):  "ablation/training/vcreg_dmodel512",
    ("AugmentedSupCon",     32):  "ablation/training/aug_supcon_dmodel32",
    ("AugmentedSupCon",     64):  "ablation/training/aug_supcon_dmodel64",
    ("AugmentedSupCon",    128):  "ablation/training/aug_supcon_dmodel128",
    ("AugmentedSupCon",    256):  "ablation/training/aug_supcon_dmodel256",
    ("AugmentedSupCon",    512):  "ablation/training/aug_supcon_dmodel512",
    ("AugmentedSelfSupCon", 32):  "ablation/training/selfsupcon_dmodel32",
    ("AugmentedSelfSupCon", 64):  "ablation/training/selfsupcon_dmodel64",
    ("AugmentedSelfSupCon",128):  "ablation/training/selfsupcon_dmodel128",
    ("AugmentedSelfSupCon",256):  "ablation/training/selfsupcon_dmodel256",
    ("AugmentedSelfSupCon",512):  "ablation/training/selfsupcon_dmodel512",
    ("VICReg",               32):  "ablation/training/vicreg_dmodel32",
    ("VICReg",               64):  "ablation/training/vicreg_dmodel64",
    ("VICReg",              128):  "ablation/training/vicreg_dmodel128",
    ("VICReg",              256):  "ablation/training/vicreg_dmodel256",
    ("VICReg",              512):  "ablation/training/vicreg_dmodel512",
}

# Filter args map model cli names to internal names
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
        d_model = next(
            (int(l.split("d_model:")[-1].strip())
             for l in text.split("\n") if "  d_model:" in l),
            None,
        )
        seed = next(
            (int(l.split("seed:")[-1].strip())
             for l in text.split("\n") if l.strip().startswith("seed:")),
            None,
        )
        if not model or d_model is None or seed is None:
            return None
        return {"model": model, "d_model": d_model, "seed": seed, "run_dir": run_dir}
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
    job_name = f"{model_short}_d{run_info['d_model']}_seed{run_info['seed']}"
    output_dir = str(run_info["run_dir"])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = SUB_DIR / f"eval_probe_dmodel_{job_name}.sub"
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
    parser.add_argument("--dmodels", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    filter_models  = {CLI_TO_MODEL[m] for m in args.models} if args.models else None
    filter_dmodels = set(args.dmodels) if args.dmodels else None
    filter_seeds   = set(args.seeds) if args.seeds else None

    train_base = Path(EOS_TRAIN_BASE)
    all_runs = [get_run_info(p) for p in sorted(train_base.iterdir()) if p.is_dir()]
    all_runs = [r for r in all_runs if r is not None]

    if filter_models:
        all_runs = [r for r in all_runs if r["model"] in filter_models]
    if filter_dmodels:
        all_runs = [r for r in all_runs if r["d_model"] in filter_dmodels]
    if filter_seeds:
        all_runs = [r for r in all_runs if r["seed"] in filter_seeds]

    # For duplicate (model, d_model, seed) keep the one with the best checkpoint
    best: dict[tuple, dict] = {}
    for r in all_runs:
        key = (r["model"], r["d_model"], r["seed"])
        ckpt = find_best_checkpoint(r["run_dir"])
        if ckpt and key not in best:
            r["ckpt"] = ckpt
            best[key] = r

    print(f"{'DRY RUN — ' if args.dry_run else ''}Found {len(best)} runs to evaluate\n")

    submitted = skipped = missing = 0
    for key in sorted(best):
        r = best[key]
        experiment = EXPERIMENT_MAP.get((r["model"], r["d_model"]))
        if experiment is None:
            print(f"  [WARN] No experiment config for {r['model']} d={r['d_model']}")
            missing += 1
            continue

        if already_done(r["run_dir"]):
            print(f"  [SKIP] {r['model']} d={r['d_model']} seed={r['seed']} — already done")
            skipped += 1
            continue

        make_sub(r, experiment, r["ckpt"], dry_run=args.dry_run)
        submitted += 1

    print(f"\nDone: {submitted} submitted, {skipped} skipped, {missing} missing config.")


if __name__ == "__main__":
    main()
