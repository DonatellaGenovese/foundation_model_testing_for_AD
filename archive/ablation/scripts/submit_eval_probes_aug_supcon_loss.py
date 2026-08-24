#!/usr/bin/env python3
"""
Submit probe evaluation jobs for all 50 AugSupCon loss ablation runs.

Usage:
    python3 scripts/ablation/submit_eval_probes_aug_supcon_loss.py [--dry-run]
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

EOS_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/loss_contrastive/logs/train/runs"

RUNS = [
    ("ablation/loss/aug_supcon_temp005", 7,     "2026-06-19_16-38-08", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_temp005", 42,    "2026-06-19_17-47-52", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_temp005", 137,   "2026-06-19_19-12-12", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp005", 1337,  "2026-06-19_19-23-21", "epoch_048.ckpt"),
    ("ablation/loss/aug_supcon_temp005", 31337, "2026-06-19_19-26-29", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp007", 7,     "2026-06-19_20-16-34", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_temp007", 42,    "2026-06-19_20-51-06", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp007", 137,   "2026-06-19_22-14-51", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp007", 1337,  "2026-06-19_22-22-11", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp007", 31337, "2026-06-19_22-22-12", "epoch_046.ckpt"),
    ("ablation/loss/aug_supcon_temp010", 7,     "2026-06-19_22-26-13", "epoch_046.ckpt"),
    ("ablation/loss/aug_supcon_temp010", 42,    "2026-06-19_22-39-37", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_temp010", 137,   "2026-06-19_22-40-23", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp010", 1337,  "2026-06-19_22-45-32", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp010", 31337, "2026-06-19_22-55-11", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp015", 7,     "2026-06-19_23-33-04", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_temp015", 42,    "2026-06-19_23-36-13", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_temp015", 137,   "2026-06-19_23-38-31", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp015", 1337,  "2026-06-19_23-46-59", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp015", 31337, "2026-06-19_23-48-25", "epoch_046.ckpt"),
    ("ablation/loss/aug_supcon_temp020", 7,     "2026-06-19_23-53-18", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_temp020", 42,    "2026-06-19_23-54-41", "epoch_048.ckpt"),
    ("ablation/loss/aug_supcon_temp020", 137,   "2026-06-20_00-00-02", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp020", 1337,  "2026-06-20_00-00-24", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_temp020", 31337, "2026-06-20_00-00-56", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw000",   7,     "2026-06-20_00-05-07", "epoch_044.ckpt"),
    ("ablation/loss/aug_supcon_cw000",   42,    "2026-06-20_00-26-49", "epoch_046.ckpt"),
    ("ablation/loss/aug_supcon_cw000",   137,   "2026-06-20_00-29-09", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw000",   1337,  "2026-06-20_00-46-18", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw000",   31337, "2026-06-20_00-44-59", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw005",   7,     "2026-06-20_00-50-50", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_cw005",   42,    "2026-06-20_01-03-52", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw005",   137,   "2026-06-20_01-16-55", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw005",   1337,  "2026-06-20_01-17-58", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw005",   31337, "2026-06-20_01-21-35", "epoch_045.ckpt"),
    ("ablation/loss/aug_supcon_cw010",   7,     "2026-06-20_01-21-06", "epoch_046.ckpt"),
    ("ablation/loss/aug_supcon_cw010",   42,    "2026-06-20_01-29-31", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_cw010",   137,   "2026-06-20_02-15-41", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw010",   1337,  "2026-06-20_02-27-38", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw010",   31337, "2026-06-20_02-30-40", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw020",   7,     "2026-06-20_02-33-31", "epoch_048.ckpt"),
    ("ablation/loss/aug_supcon_cw020",   42,    "2026-06-20_02-46-52", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw020",   137,   "2026-06-20_02-46-13", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw020",   1337,  "2026-06-20_02-47-43", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw020",   31337, "2026-06-20_02-58-33", "epoch_046.ckpt"),
    ("ablation/loss/aug_supcon_cw050",   7,     "2026-06-20_03-20-35", "epoch_047.ckpt"),
    ("ablation/loss/aug_supcon_cw050",   42,    "2026-06-20_03-21-58", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw050",   137,   "2026-06-20_03-34-44", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw050",   1337,  "2026-06-20_03-38-32", "epoch_049.ckpt"),
    ("ablation/loss/aug_supcon_cw050",   31337, "2026-06-20_03-37-29", "epoch_049.ckpt"),
]

PROJECT_DIR = "/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD"
LOG_DIR     = Path(PROJECT_DIR) / "logs/condor_logs/ablation/eval_probes_aug_supcon_loss"
WRAPPER     = Path(PROJECT_DIR) / "scripts/ablation/wrapper_eval_probe_ablation.sh"
SUB_DIR     = Path(PROJECT_DIR) / "logs/condor_subs/ablation"


def make_sub(experiment: str, seed: int, run_ts: str, ckpt_name: str, dry_run: bool) -> None:
    exp_short  = experiment.replace("ablation/loss/", "").replace("/", "_")
    job_name   = f"{exp_short}_seed{seed}"
    ckpt_path  = f"{EOS_BASE}/{run_ts}/checkpoints/{ckpt_name}"
    output_dir = f"{EOS_BASE}/{run_ts}"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = SUB_DIR / f"eval_probe_loss_{job_name}.sub"
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

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(RUNS)} AugSupCon loss probe eval jobs\n")

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
