#!/usr/bin/env python3
"""
Submit VCReg new_exp training jobs to HTCondor.

d_model in {32, 64, 128, 256}, d_ff = 4 * d_model.
Full dataset: 1M/100k/100k events per class.
Checkpoint saved on val/vcreg_loss (min).
10 seeds per config = 40 jobs total.

Runs saved to: /eos/user/d/dgenoves/anomaly_pipeline/new_exp/vcreg/logs/

Usage:
    python3 scripts/new_exp/submit_training_vcreg.py [--dry-run]
    python3 scripts/new_exp/submit_training_vcreg.py --dmodels 64 128
    python3 scripts/new_exp/submit_training_vcreg.py --seeds 7 42
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/new_exp/training_vcreg"

DMODELS = [32, 64, 128, 256]
# Original ablation seeds with 1337→12345, plus 5 new spread-out seeds
SEEDS = [7, 42, 100, 137, 1000, 10000, 12345, 31337, 100000, 999999]

EXPERIMENTS = {d: f"new_exp/vcreg_dmodel{d}" for d in DMODELS}


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
request_cpus = 12
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
    parser.add_argument("--dmodels",  nargs="+", type=int, choices=DMODELS, default=None)
    parser.add_argument("--seeds",    nargs="+", type=int, default=None)
    args = parser.parse_args()

    dmodels = args.dmodels or DMODELS
    seeds   = args.seeds   or SEEDS

    total = len(dmodels) * len(seeds)
    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(dmodels)} configs × {len(seeds)} seeds = {total} jobs")
    print(f"d_models: {dmodels}")
    print(f"seeds:    {seeds}")
    print(f"EOS logs: /eos/user/d/dgenoves/anomaly_pipeline/new_exp/vcreg/logs/")
    print()

    for d in dmodels:
        exp = EXPERIMENTS[d]
        print(f"[VCReg d_model={d}, d_ff={d*4}]")
        for seed in seeds:
            submit_job(exp, seed, dry_run=args.dry_run)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
