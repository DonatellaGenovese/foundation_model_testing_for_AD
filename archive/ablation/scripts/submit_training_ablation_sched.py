#!/usr/bin/env python3
"""
Submit learning-rate scheduler ablation training jobs to HTCondor.

Standard d=128 architecture for all 4 models.
scheduler in {plateau, plateau_agg, cosine, none} × 5 seeds = 80 jobs.

Default scheduler: ReduceLROnPlateau (plateau) as in training backbones.

Runs saved to: /eos/user/d/dgenoves/anomaly_pipeline/ablation/sched/logs/

Usage:
    python3 scripts/ablation/submit_training_ablation_sched.py [--dry-run]
    python3 scripts/ablation/submit_training_ablation_sched.py --models vcreg --scheds cosine
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/training_sched"

SEEDS    = [7, 42, 137, 1337, 31337]
MODELS   = ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
SCHEDS   = ["plateau", "plateau_agg", "cosine", "none"]
DEFAULT_SCHED = "plateau"

BACKBONE = {
    "vcreg": "backbone_vcreg",
    "aug_supcon": "backbone_aug_supcon",
    "selfsupcon": "backbone_selfsupcon",
    "vicreg": "backbone_vicreg",
}

SCHEDULER_BLOCK = {
    "plateau": """\
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    _partial_: true
    mode: min
    factor: 0.2
    patience: 10
    threshold: 0.0001
    threshold_mode: rel
    T_max: null
    eta_min: null""",
    "plateau_agg": """\
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    _partial_: true
    mode: min
    factor: 0.2
    patience: 2
    threshold: 0.0001
    threshold_mode: rel
    T_max: null
    eta_min: null""",
    "cosine": """\
  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    _partial_: true
    T_max: ${trainer.max_epochs}
    eta_min: 0.0
    mode: null
    factor: null
    patience: null
    threshold: null
    threshold_mode: null""",
    "none": """\
  scheduler: null""",
}

EXPERIMENTS = {
    (model, sched): f"ablation/training/{model}_sched_{sched}"
    for model in MODELS
    for sched in SCHEDS
}


def write_experiment_config(model: str, sched: str) -> None:
    cfg_dir = PROJECT_DIR / "configs/experiment/ablation/training"
    path = cfg_dir / f"{model}_sched_{sched}.yaml"
    content = f"""# @package _global_
# {model} scheduler={sched} ablation — standard d=128 architecture.
defaults:
  - ablation/training/{BACKBONE[model]}
  - override /paths: ablation_sched_cern
  - _self_

tags: ["ablation", "sched", "{model}", "d128", "sched_{sched}"]

model:
{SCHEDULER_BLOCK[sched]}

logger:
  mlflow:
    experiment_name: "ablation_sched"
    run_name: "{model}_sched_{sched}"
"""
    path.write_text(content)


def ensure_configs() -> None:
    for model in MODELS:
        for sched in SCHEDS:
            write_experiment_config(model, sched)


def submit_job(experiment: str, seed: int, dry_run: bool = False, flavour: str = "tomorrow") -> None:
    exp_short = experiment.split("/")[-1]
    job_name  = f"{exp_short}_seed{seed}"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = LOG_DIR / f"{job_name}.sub"

    sub_content = f"""\
executable = {WRAPPER}

output = {LOG_DIR}/{job_name}.out
error  = {LOG_DIR}/{job_name}.err
log    = {LOG_DIR}/{job_name}.log

stream_output = True
stream_error  = True

run_as_owner = True
+JobFlavour  = "{flavour}"
getenv       = True
request_cpus = 6
request_gpus = 1

Requirements = (TARGET.GPUs_GlobalMemoryMb >= 16000)

environment = "EXPERIMENT={experiment} SEED={seed} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"

queue
"""
    sub_path.write_text(sub_content)

    if dry_run:
        print(f"  [DRY] {job_name}")
        return

    result = subprocess.run(["condor_submit", str(sub_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Submitted: {job_name}")
    else:
        print(f"  FAILED:    {job_name}\n{result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--scheds", nargs="+", choices=SCHEDS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--flavour", default="tomorrow")
    parser.add_argument("--write-configs-only", action="store_true")
    args = parser.parse_args()

    ensure_configs()
    if args.write_configs_only:
        print("Wrote scheduler ablation experiment configs.")
        return

    models = args.models or MODELS
    scheds = args.scheds or SCHEDS
    seeds  = args.seeds or SEEDS

    to_submit = [(m, s) for m in models for s in scheds]
    total = len(to_submit) * len(seeds)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(to_submit)} configs × {len(seeds)} seeds = {total} jobs")
    print(f"Models:   {models}")
    print(f"Scheds:   {scheds}")
    print(f"Seeds:    {seeds}")
    print(f"Flavour:  {args.flavour}")
    print(f"Default:  {DEFAULT_SCHED}")
    print(f"EOS logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/sched/logs/")
    print()

    for model, sched in to_submit:
        exp = EXPERIMENTS[(model, sched)]
        marker = " (default)" if sched == DEFAULT_SCHED else ""
        print(f"[{model} sched={sched}{marker}]")
        for seed in seeds:
            submit_job(exp, seed, dry_run=args.dry_run, flavour=args.flavour)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
