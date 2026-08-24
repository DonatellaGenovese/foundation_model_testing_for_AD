"""
Submit augmentation ablation training jobs to HTCondor.

Submits one GPU job per (experiment, seed):
  - 9 experiments: 3 models × 3 augmentation types
  - 5 seeds: 7, 42, 137, 1337, 31337
  = 45 total jobs

Results saved via MLflow to:
  /eos/user/d/dgenoves/anomaly_pipeline/ablation/logs/

Condor logs saved to:
  {PROJECT_DIR}/logs/condor_logs/ablation/physics_aug/

Usage (on lxplus HOST, NOT inside apptainer):
    python scripts/ablation/submit_augmentation_ablation.py
    python scripts/ablation/submit_augmentation_ablation.py --dry-run
    python scripts/ablation/submit_augmentation_ablation.py --experiments aug_supcon_random_feature aug_supcon_physics
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/physics_aug"

SEEDS = [7, 42, 137, 1337, 31337]

EXPERIMENTS = [
    "ablation/aug_supcon_random_feature",
    "ablation/aug_supcon_random_particle",
    "ablation/aug_supcon_physics",
    "ablation/selfsupcon_random_feature",
    "ablation/selfsupcon_random_particle",
    "ablation/selfsupcon_physics",
    "ablation/vicreg_random_feature",
    "ablation/vicreg_random_particle",
    "ablation/vicreg_physics",
]


def submit_job(experiment: str, seed: int, dry_run: bool = False,
               extra_overrides: str = "", job_suffix: str = "") -> None:
    exp_short = experiment.replace("/", "_")
    suffix    = f"_{job_suffix}" if job_suffix else ""
    job_name  = f"{exp_short}_seed{seed}{suffix}"

    log_out  = LOG_DIR / f"{job_name}.out"
    log_err  = LOG_DIR / f"{job_name}.err"
    log_log  = LOG_DIR / f"{job_name}.log"
    sub_path = LOG_DIR / f"{job_name}.sub"

    env_line = f"EXPERIMENT={experiment} SEED={seed} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"
    if extra_overrides:
        env_line += f" EXTRA_OVERRIDES='{extra_overrides}'"

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

environment = "{env_line}"

queue
"""

    sub_path.write_text(sub_content)

    if dry_run:
        print(f"  [DRY-RUN] {job_name}")
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
    parser = argparse.ArgumentParser(description="Submit augmentation ablation jobs to Condor")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without submitting")
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Subset of experiment basenames to submit (e.g. aug_supcon_physics vicreg_random_feature)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Override seeds (default: 7 42 137 1337 31337)"
    )
    parser.add_argument(
        "--overrides", default="",
        help="Extra Hydra overrides passed to train.py (e.g. 'model.mask_probability=0.15')"
    )
    parser.add_argument(
        "--job-suffix", default="",
        help="Suffix appended to job name to avoid overwriting existing .sub/.log files"
    )
    args = parser.parse_args()

    seeds = args.seeds or SEEDS
    if args.experiments:
        experiments = [f"ablation/{e}" for e in args.experiments]
    else:
        experiments = EXPERIMENTS

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total = len(experiments) * len(seeds)
    print(f"Submitting {len(experiments)} experiments × {len(seeds)} seeds = {total} jobs")
    print(f"Seeds      : {seeds}")
    print(f"Condor logs: {LOG_DIR}")
    print(f"MLflow logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/logs/")
    if args.overrides:
        print(f"Overrides  : {args.overrides}")
    print()

    for experiment in experiments:
        print(f"[{experiment}]")
        for seed in seeds:
            submit_job(experiment, seed, dry_run=args.dry_run,
                       extra_overrides=args.overrides, job_suffix=args.job_suffix)

    suffix = "(dry-run)" if args.dry_run else f"{total} jobs submitted"
    print(f"\nDone. {suffix}")


if __name__ == "__main__":
    main()
