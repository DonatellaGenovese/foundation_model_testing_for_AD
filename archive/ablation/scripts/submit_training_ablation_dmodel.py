#!/usr/bin/env python3
"""
Submit d_model training ablation jobs to HTCondor.

12 configs × 5 seeds = 60 jobs.
d_model in {32, 64, 256, 512} for AugSupCon, SelfSupCon, VCReg.
d_model=128 reused from loss ablation (cw050, temp005, vcreg baseline).

Usage:
    python3 scripts/ablation/submit_training_ablation_dmodel.py [--dry-run]
    python3 scripts/ablation/submit_training_ablation_dmodel.py --models aug_supcon vcreg
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/training_dmodel"

SEEDS = [7, 42, 137, 1337, 31337]

MODELS  = ["aug_supcon", "selfsupcon", "vcreg"]
DMODELS = [32, 64, 256, 512]

EXPERIMENTS = [
    f"ablation/training/{model}_dmodel{dm}"
    for model in MODELS
    for dm in DMODELS
]


def submit_job(experiment: str, seed: int, dry_run: bool = False) -> None:
    exp_short = experiment.split("/")[-1]
    job_name  = f"{exp_short}_seed{seed}"

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

    sub_path = LOG_DIR / f"{job_name}.sub"
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
    parser = argparse.ArgumentParser(description="Submit d_model training ablation jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--dmodels", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    seeds   = args.seeds or SEEDS
    models  = args.models or MODELS
    dmodels = args.dmodels or DMODELS
    experiments = [f"ablation/training/{m}_dmodel{d}" for m in models for d in dmodels]

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total = len(experiments) * len(seeds)
    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(experiments)} experiments × {len(seeds)} seeds = {total} jobs")
    print(f"Models  : {models}")
    print(f"d_models: {dmodels}")
    print(f"Seeds   : {seeds}")
    print(f"EOS logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/training/logs/")
    print()

    for experiment in experiments:
        print(f"[{experiment}]")
        for seed in seeds:
            submit_job(experiment, seed, dry_run=args.dry_run)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
