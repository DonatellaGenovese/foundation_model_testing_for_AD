#!/usr/bin/env python3
"""
Submit probe evaluation jobs for all 25 SelfSupCon loss ablation runs.

Usage:
    python3 scripts/ablation/submit_eval_probes_selfsupcon_loss.py [--dry-run]
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

EOS_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/loss_contrastive/logs/train/runs"

RUNS = [
    ("ablation/loss/selfsupcon_temp005", 7,     "2026-06-20_03-47-32", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp005", 42,    "2026-06-20_03-50-33", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp005", 137,   "2026-06-20_04-07-14", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp005", 1337,  "2026-06-20_04-08-43", "epoch_048.ckpt"),
    ("ablation/loss/selfsupcon_temp005", 31337, "2026-06-20_04-11-55", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp007", 7,     "2026-06-20_04-15-07", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp007", 42,    "2026-06-20_04-16-33", "epoch_047.ckpt"),
    ("ablation/loss/selfsupcon_temp007", 137,   "2026-06-20_04-25-50", "epoch_047.ckpt"),
    ("ablation/loss/selfsupcon_temp007", 1337,  "2026-06-20_04-25-30", "epoch_048.ckpt"),
    ("ablation/loss/selfsupcon_temp007", 31337, "2026-06-20_04-26-05", "epoch_048.ckpt"),
    ("ablation/loss/selfsupcon_temp010", 7,     "2026-06-20_04-27-16", "epoch_045.ckpt"),
    ("ablation/loss/selfsupcon_temp010", 42,    "2026-06-20_04-26-32", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp010", 137,   "2026-06-20_04-28-12", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp010", 1337,  "2026-06-20_04-35-14", "epoch_048.ckpt"),
    ("ablation/loss/selfsupcon_temp010", 31337, "2026-06-20_04-37-30", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp015", 7,     "2026-06-20_04-38-37", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp015", 42,    "2026-06-20_04-37-52", "epoch_048.ckpt"),
    ("ablation/loss/selfsupcon_temp015", 137,   "2026-06-20_04-44-05", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp015", 1337,  "2026-06-20_04-43-41", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp015", 31337, "2026-06-20_05-04-46", "epoch_046.ckpt"),
    ("ablation/loss/selfsupcon_temp020", 7,     "2026-06-20_05-07-06", "epoch_048.ckpt"),
    ("ablation/loss/selfsupcon_temp020", 42,    "2026-06-20_05-14-16", "epoch_047.ckpt"),
    ("ablation/loss/selfsupcon_temp020", 137,   "2026-06-20_05-15-27", "epoch_049.ckpt"),
    ("ablation/loss/selfsupcon_temp020", 1337,  "2026-06-20_05-18-58", "epoch_047.ckpt"),
    ("ablation/loss/selfsupcon_temp020", 31337, "2026-06-20_05-34-50", "epoch_049.ckpt"),
]

PROJECT_DIR = "/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD"
LOG_DIR     = Path(PROJECT_DIR) / "logs/condor_logs/ablation/eval_probes_selfsupcon_loss"
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

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(RUNS)} SelfSupCon loss probe eval jobs\n")

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
