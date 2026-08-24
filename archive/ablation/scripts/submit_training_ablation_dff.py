#!/usr/bin/env python3
"""
Submit d_ff ablation training jobs to HTCondor.

Standard architecture (d_model=128, num_layers=6, n_heads=8, dropout=0.1) for all 4 models.
d_ff in {256, 512, 1024} × 5 seeds = 60 jobs.

Runs saved to: /eos/user/d/dgenoves/anomaly_pipeline/ablation/dff/logs/

Usage:
    python3 scripts/ablation/submit_training_ablation_dff.py [--dry-run]
    python3 scripts/ablation/submit_training_ablation_dff.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_training_ablation_dff.py --dff 256 512
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/training_dff"

SEEDS  = [7, 42, 137, 1337, 31337]
MODELS = ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
DFFS   = [256, 512, 1024]

EXPERIMENTS = {
    (model, d): f"ablation/training/{model}_dff_{d}"
    for model in MODELS
    for d in DFFS
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
    parser.add_argument("--models",  nargs="+", choices=MODELS, default=None)
    parser.add_argument("--dff",     nargs="+", type=int, choices=DFFS, default=None)
    parser.add_argument("--seeds",   nargs="+", type=int, default=None)
    args = parser.parse_args()

    models = args.models or MODELS
    dffs   = args.dff    or DFFS
    seeds  = args.seeds  or SEEDS

    to_submit = [(m, d) for m in models for d in dffs]
    total = len(to_submit) * len(seeds)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(to_submit)} configs × {len(seeds)} seeds = {total} jobs")
    print(f"Models:  {models}")
    print(f"d_ff:    {dffs}")
    print(f"Seeds:   {seeds}")
    print(f"EOS logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/dff/logs/")
    print()

    for model, d in to_submit:
        exp = EXPERIMENTS[(model, d)]
        print(f"[{model} d_ff={d}]")
        for seed in seeds:
            submit_job(exp, seed, dry_run=args.dry_run)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
