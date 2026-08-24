#!/usr/bin/env python3
"""
Submit probe evaluation jobs for all 15 VICReg augmentation ablation runs.

Usage:
    python3 scripts/ablation/submit_eval_probes_vicreg.py [--dry-run]
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

EOS_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/logs/train/runs"

RUNS = [
    ("ablation/vicreg_random_feature",  7,     "2026-06-20_05-37-08", "epoch_009.ckpt"),
    ("ablation/vicreg_random_feature",  42,    "2026-06-20_05-40-54", "epoch_048.ckpt"),
    ("ablation/vicreg_random_feature",  137,   "2026-06-20_05-52-42", "epoch_004.ckpt"),
    ("ablation/vicreg_random_feature",  1337,  "2026-06-20_05-53-59", "epoch_005.ckpt"),
    ("ablation/vicreg_random_feature",  31337, "2026-06-20_05-53-37", "epoch_047.ckpt"),
    ("ablation/vicreg_random_particle", 7,     "2026-06-17_20-51-43", "epoch_044.ckpt"),
    ("ablation/vicreg_random_particle", 42,    "2026-06-17_21-07-13", "epoch_045.ckpt"),
    ("ablation/vicreg_random_particle", 137,   "2026-06-17_21-20-28", "epoch_049.ckpt"),
    ("ablation/vicreg_random_particle", 1337,  "2026-06-17_21-22-14", "epoch_049.ckpt"),
    ("ablation/vicreg_random_particle", 31337, "2026-06-17_21-31-47", "epoch_049.ckpt"),
    ("ablation/vicreg_physics",         7,     "2026-06-17_21-31-45", "epoch_049.ckpt"),
    ("ablation/vicreg_physics",         42,    "2026-06-17_21-33-51", "epoch_049.ckpt"),
    ("ablation/vicreg_physics",         137,   "2026-06-18_14-31-11", "epoch_048.ckpt"),
    ("ablation/vicreg_physics",         1337,  "2026-06-17_21-57-42", "epoch_043.ckpt"),
    ("ablation/vicreg_physics",         31337, "2026-06-18_18-42-48", "epoch_049.ckpt"),
]

PROJECT_DIR = "/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD"
LOG_DIR     = Path(PROJECT_DIR) / "logs/condor_logs/ablation/eval_probes_vicreg"
WRAPPER     = Path(PROJECT_DIR) / "scripts/ablation/wrapper_eval_probe_ablation.sh"
SUB_DIR     = Path(PROJECT_DIR) / "logs/condor_subs/ablation"


def make_sub(experiment: str, seed: int, run_ts: str, ckpt_name: str,
             dry_run: bool) -> None:
    exp_short  = experiment.replace("ablation/", "").replace("/", "_")
    job_name   = f"{exp_short}_seed{seed}"
    ckpt_path  = f"{EOS_BASE}/{run_ts}/checkpoints/{ckpt_name}"
    output_dir = f"{EOS_BASE}/{run_ts}"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = SUB_DIR / f"eval_probe_{job_name}.sub"
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

        environment = "EXPERIMENT={experiment} CKPT_PATH={ckpt_path} SEED={seed} OUTPUT_DIR={output_dir}"

        queue
    """)

    sub_path.write_text(sub_content)
    print(f"  {'[DRY]' if dry_run else '[SUB]'} {job_name}  ckpt={ckpt_name}")

    if not dry_run:
        result = subprocess.run(
            ["condor_submit", str(sub_path)],
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr.strip()}")
        else:
            print(f"    → {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(RUNS)} VICReg probe eval jobs\n")

    submitted = skipped = 0
    for experiment, seed, run_ts, ckpt_name in RUNS:
        output_dir = f"{EOS_BASE}/{run_ts}"
        check = subprocess.run(
            ["eos", "root://eosuser.cern.ch", "ls", f"{output_dir}/probe_evaluation/"],
            capture_output=True, text=True
        )
        if "probe_results.json" in check.stdout:
            exp_short = experiment.replace("ablation/", "").replace("/", "_")
            print(f"  [SKIP] {exp_short}_seed{seed} — already done")
            skipped += 1
            continue

        make_sub(experiment, seed, run_ts, ckpt_name, dry_run=args.dry_run)
        submitted += 1

    print(f"\nDone: {submitted} submitted, {skipped} skipped.")


if __name__ == "__main__":
    main()
