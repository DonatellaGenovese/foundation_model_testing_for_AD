#!/usr/bin/env python3
"""
Submit batch_size ablation training jobs to HTCondor.

Standard d=128 architecture (n_heads=8, n_layers=6, d_ff=512, dropout=0.1) for all 4 models.
batch_size in {512, 1024, 2048} × 5 seeds = 60 jobs.

Default batch size in ablations: 1024.

Runs saved to: /eos/user/d/dgenoves/anomaly_pipeline/ablation/bs/logs/

Usage:
    python3 scripts/ablation/submit_training_ablation_bs.py [--dry-run]
    python3 scripts/ablation/submit_training_ablation_bs.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_training_ablation_bs.py --batch-sizes 512 1024 --flavour tomorrow
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/training_bs"

SEEDS       = [7, 42, 137, 1337, 31337]
MODELS      = ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
BATCH_SIZES = [512, 1024, 2048]
DEFAULT_BS  = 1024

EXPERIMENTS = {
    (model, bs): f"ablation/training/{model}_bs_{bs}"
    for model in MODELS
    for bs in BATCH_SIZES
}


def submit_job(experiment: str, seed: int, dry_run: bool = False, flavour: str = "nextweek") -> None:
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
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=BATCH_SIZES, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--flavour", default="tomorrow",
                        help="HTCondor JobFlavour (default: tomorrow)")
    args = parser.parse_args()

    models = args.models or MODELS
    batch_sizes = args.batch_sizes or BATCH_SIZES
    seeds = args.seeds or SEEDS

    to_submit = [(m, bs) for m in models for bs in batch_sizes]
    total = len(to_submit) * len(seeds)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(to_submit)} configs × {len(seeds)} seeds = {total} jobs")
    print(f"Models:      {models}")
    print(f"batch_size:  {batch_sizes}")
    print(f"Seeds:       {seeds}")
    print(f"Flavour:     {args.flavour}")
    print(f"Default bs:  {DEFAULT_BS}")
    print(f"EOS logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/bs/logs/")
    print()

    for model, bs in to_submit:
        exp = EXPERIMENTS[(model, bs)]
        marker = " (default)" if bs == DEFAULT_BS else ""
        print(f"[{model} batch_size={bs}{marker}]")
        for seed in seeds:
            submit_job(exp, seed, dry_run=args.dry_run, flavour=args.flavour)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
