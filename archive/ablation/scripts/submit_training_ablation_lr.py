#!/usr/bin/env python3
"""
Submit learning-rate ablation training jobs to HTCondor.

Standard d=128 architecture (n_heads=8, n_layers=6, d_ff=512, dropout=0.1) for all 4 models.
lr in {1e-4, 5e-4, 1e-3, 3e-3} × 5 seeds = 80 jobs.

Default ablation lrs: VCReg/VICReg=1e-3, AugSupCon/SelfSupCon=1e-4.

Runs saved to: /eos/user/d/dgenoves/anomaly_pipeline/ablation/lr/logs/

Usage:
    python3 scripts/ablation/submit_training_ablation_lr.py [--dry-run]
    python3 scripts/ablation/submit_training_ablation_lr.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_training_ablation_lr.py --lrs 1e4 1e3
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/training_lr"

SEEDS  = [7, 42, 137, 1337, 31337]
MODELS = ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
LR_SLUGS = ["1e4", "5e4", "1e3", "3e3"]

LR_VALUES = {
    "1e4": 1e-4,
    "5e4": 5e-4,
    "1e3": 1e-3,
    "3e3": 3e-3,
}

DEFAULT_LR = {
    "vcreg": "1e3",
    "aug_supcon": "1e4",
    "selfsupcon": "1e4",
    "vicreg": "1e3",
}

EXPERIMENTS = {
    (model, slug): f"ablation/training/{model}_lr_{slug}"
    for model in MODELS
    for slug in LR_SLUGS
}


def submit_job(experiment: str, seed: int, dry_run: bool = False) -> None:
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
+JobFlavour  = "nextweek"
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
    parser.add_argument("--lrs", nargs="+", choices=LR_SLUGS, default=None,
                        help="LR slug: 1e4=1e-4, 5e4=5e-4, 1e3=1e-3, 3e3=3e-3")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    models = args.models or MODELS
    lrs    = args.lrs    or LR_SLUGS
    seeds  = args.seeds  or SEEDS

    to_submit = [(m, slug) for m in models for slug in lrs]
    total = len(to_submit) * len(seeds)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(to_submit)} configs × {len(seeds)} seeds = {total} jobs")
    print(f"Models: {models}")
    print(f"LRs:    {[LR_VALUES[s] for s in lrs]}")
    print(f"Seeds:  {seeds}")
    print(f"Defaults: {', '.join(f'{m}={LR_VALUES[DEFAULT_LR[m]]}' for m in models)}")
    print(f"EOS logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/lr/logs/")
    print()

    for model, slug in to_submit:
        exp = EXPERIMENTS[(model, slug)]
        marker = " (default)" if slug == DEFAULT_LR[model] else ""
        print(f"[{model} lr={LR_VALUES[slug]:g}{marker}]")
        for seed in seeds:
            submit_job(exp, seed, dry_run=args.dry_run)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
