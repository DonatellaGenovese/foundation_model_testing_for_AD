#!/usr/bin/env python3
"""
Submit probe evaluation jobs for all 25 VICReg loss ablation runs.

Usage:
    python3 scripts/ablation/submit_eval_probes_vicreg_loss.py [--dry-run]
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

EOS_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/loss_contrastive/logs/train/runs"

RUNS = [
    ("ablation/loss/vicreg_inv0",  7,     "2026-06-21_10-35-24", "epoch_000.ckpt"),
    ("ablation/loss/vicreg_inv0",  42,    "2026-06-21_11-18-07", "epoch_003.ckpt"),
    ("ablation/loss/vicreg_inv0",  137,   "2026-06-21_11-45-07", "epoch_000.ckpt"),
    ("ablation/loss/vicreg_inv0",  1337,  "2026-06-21_12-03-05", "epoch_000.ckpt"),
    ("ablation/loss/vicreg_inv0",  31337, "2026-06-21_12-09-06", "epoch_001.ckpt"),
    ("ablation/loss/vicreg_inv5",  7,     "2026-06-21_12-09-28", "epoch_040.ckpt"),
    ("ablation/loss/vicreg_inv5",  42,    "2026-06-21_12-28-08", "epoch_046.ckpt"),
    ("ablation/loss/vicreg_inv5",  137,   "2026-06-21_12-36-05", "epoch_042.ckpt"),
    ("ablation/loss/vicreg_inv5",  1337,  "2026-06-21_13-22-53", "epoch_040.ckpt"),
    ("ablation/loss/vicreg_inv5",  31337, "2026-06-21_13-50-59", "epoch_048.ckpt"),
    ("ablation/loss/vicreg_inv10", 7,     "2026-06-21_14-00-17", "epoch_042.ckpt"),
    ("ablation/loss/vicreg_inv10", 42,    "2026-06-21_14-16-38", "epoch_042.ckpt"),
    ("ablation/loss/vicreg_inv10", 137,   "2026-06-21_14-48-33", "epoch_046.ckpt"),
    ("ablation/loss/vicreg_inv10", 1337,  "2026-06-21_14-54-43", "epoch_048.ckpt"),
    ("ablation/loss/vicreg_inv10", 31337, "2026-06-21_15-10-14", "epoch_047.ckpt"),
    ("ablation/loss/vicreg_inv25", 7,     "2026-06-21_15-34-17", "epoch_046.ckpt"),
    ("ablation/loss/vicreg_inv25", 42,    "2026-06-21_15-47-18", "epoch_048.ckpt"),
    ("ablation/loss/vicreg_inv25", 137,   "2026-06-21_15-49-16", "epoch_048.ckpt"),
    ("ablation/loss/vicreg_inv25", 1337,  "2026-06-21_16-03-39", "epoch_043.ckpt"),
    ("ablation/loss/vicreg_inv25", 31337, "2026-06-21_16-08-05", "epoch_049.ckpt"),
    ("ablation/loss/vicreg_inv50", 7,     "2026-06-21_16-32-04", "epoch_049.ckpt"),
    ("ablation/loss/vicreg_inv50", 42,    "2026-06-21_16-43-37", "epoch_037.ckpt"),
    ("ablation/loss/vicreg_inv50", 137,   "2026-06-21_16-44-01", "epoch_047.ckpt"),
    ("ablation/loss/vicreg_inv50", 1337,  "2026-06-21_17-05-27", "epoch_049.ckpt"),
    ("ablation/loss/vicreg_inv50", 31337, "2026-06-21_18-09-31", "epoch_046.ckpt"),
]

PROJECT_DIR = "/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD"
LOG_DIR     = Path(PROJECT_DIR) / "logs/condor_logs/ablation/eval_probes_vicreg_loss"
WRAPPER     = Path(PROJECT_DIR) / "scripts/ablation/wrapper_eval_probe_ablation.sh"
SUB_DIR     = Path(PROJECT_DIR) / "logs/condor_subs/ablation"


def make_sub(experiment: str, seed: int, run_ts: str, ckpt_name: str, dry_run: bool) -> None:
    exp_short  = experiment.replace("ablation/loss/", "").replace("/", "_")
    job_name   = f"{exp_short}_seed{seed}"
    ckpt_path  = f"{EOS_BASE}/{run_ts}/checkpoints/{ckpt_name}"
    output_dir = f"{EOS_BASE}/{run_ts}"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = SUB_DIR / f"eval_probe_vicreg_loss_{job_name}.sub"
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

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(RUNS)} VICReg loss probe eval jobs\n")

    submitted = skipped = 0
    for experiment, seed, run_ts, ckpt_name in RUNS:
        output_dir = f"{EOS_BASE}/{run_ts}"
        check = subprocess.run(
            ["eos", "root://eosuser.cern.ch", "ls", f"{output_dir}/probe_evaluation/"],
            capture_output=True, text=True
        )
        if "probe_results.json" in check.stdout:
            exp_short = experiment.replace("ablation/loss/", "")
            print(f"  [SKIP] {exp_short}_seed{seed} — already done")
            skipped += 1
            continue
        make_sub(experiment, seed, run_ts, ckpt_name, dry_run=args.dry_run)
        submitted += 1

    print(f"\nDone: {submitted} submitted, {skipped} skipped.")


if __name__ == "__main__":
    main()
