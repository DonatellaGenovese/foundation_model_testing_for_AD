#!/usr/bin/env python3
"""
Submit n_layers ablation training jobs to HTCondor.

Standard architecture (d_model=128, n_heads=8, d_ff=512, dropout=0.1) for all 4 models.
num_layers in {2, 4, 6} × 5 seeds = 60 jobs.

Runs saved to: /eos/user/d/dgenoves/anomaly_pipeline/ablation/nlayers/logs/

Usage:
    python3 scripts/ablation/submit_training_ablation_nlayers.py [--dry-run]
    python3 scripts/ablation/submit_training_ablation_nlayers.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_training_ablation_nlayers.py --nlayers 2 4
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/training_nlayers"

SEEDS   = [7, 42, 137, 1337, 31337]
MODELS  = ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
NLAYERS = [2, 4, 6]

EXPERIMENTS = {
    (model, n): f"ablation/training/{model}_nlayers_{n}"
    for model in MODELS
    for n in NLAYERS
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
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--models",   nargs="+", choices=MODELS, default=None)
    parser.add_argument("--nlayers",  nargs="+", type=int, choices=NLAYERS, default=None)
    parser.add_argument("--seeds",    nargs="+", type=int, default=None)
    args = parser.parse_args()

    models  = args.models  or MODELS
    nlayers = args.nlayers or NLAYERS
    seeds   = args.seeds   or SEEDS

    to_submit = [(m, n) for m in models for n in nlayers]
    total = len(to_submit) * len(seeds)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(to_submit)} configs × {len(seeds)} seeds = {total} jobs")
    print(f"Models:   {models}")
    print(f"n_layers: {nlayers}")
    print(f"Seeds:    {seeds}")
    print(f"EOS logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/nlayers/logs/")
    print()

    for model, n in to_submit:
        exp = EXPERIMENTS[(model, n)]
        print(f"[{model} num_layers={n}]")
        for seed in seeds:
            submit_job(exp, seed, dry_run=args.dry_run)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
