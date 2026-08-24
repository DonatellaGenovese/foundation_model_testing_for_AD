#!/usr/bin/env python3
"""
Submit VCReg loss ablation jobs to HTCondor.

6 configs × 5 seeds = 30 jobs.
Results saved to /eos/user/d/dgenoves/anomaly_pipeline/ablation/loss/logs/

Usage:
    python3 scripts/ablation/submit_loss_ablation.py [--dry-run]
    python3 scripts/ablation/submit_loss_ablation.py --experiments vcreg_var25 vcreg_var50
    python3 scripts/ablation/submit_loss_ablation.py --seeds 7 42
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/loss"

SEEDS = [7, 42, 137, 1337, 31337]

EXPERIMENTS = [
    "ablation/loss/vcreg_var5",
    "ablation/loss/vcreg_var10",
    "ablation/loss/vcreg_var25",
    "ablation/loss/vcreg_var50",
    "ablation/loss/vcreg_cov5",
    "ablation/loss/vcreg_cov10",
]


def submit_job(experiment: str, seed: int, dry_run: bool = False) -> None:
    exp_short = experiment.replace("/", "_")
    job_name  = f"{exp_short}_seed{seed}"

    log_out  = LOG_DIR / f"{job_name}.out"
    log_err  = LOG_DIR / f"{job_name}.err"
    log_log  = LOG_DIR / f"{job_name}.log"
    sub_path = LOG_DIR / f"{job_name}.sub"

    basename = experiment.split("/")[-1]
    run_name = f"{basename}_seed{seed}"

    sub_content = f"""\
executable = {WRAPPER}

output = {log_out}
error  = {log_err}
log    = {log_log}

stream_output = True
stream_error  = True

run_as_owner = True
+JobFlavour  = "nextweek"
getenv       = True
request_cpus = 6
request_gpus = 1

Requirements = (TARGET.GPUs_GlobalMemoryMb >= 40000)

environment = "EXPERIMENT={experiment} SEED={seed} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"

queue
"""

    sub_path.write_text(sub_content)

    if dry_run:
        print(f"  [DRY] {job_name}")
        return

    result = subprocess.run(
        ["condor_submit", str(sub_path)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  Submitted: {job_name}")
    else:
        print(f"  FAILED:    {job_name}\n{result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Submit VCReg loss ablation jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Subset of experiment basenames (e.g. vcreg_var25 vcreg_cov5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    seeds = args.seeds or SEEDS
    if args.experiments:
        experiments = [f"ablation/loss/{e}" for e in args.experiments]
    else:
        experiments = EXPERIMENTS

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total = len(experiments) * len(seeds)
    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(experiments)} experiments × {len(seeds)} seeds = {total} jobs")
    print(f"Seeds      : {seeds}")
    print(f"Condor logs: {LOG_DIR}")
    print(f"MLflow logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/loss/logs/")
    print()

    for experiment in experiments:
        print(f"[{experiment}]")
        for seed in seeds:
            submit_job(experiment, seed, dry_run=args.dry_run)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
