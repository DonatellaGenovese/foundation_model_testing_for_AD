#!/usr/bin/env python3
"""
Submit probe evaluation jobs for all 30 VCReg loss ablation runs.

Usage:
    python3 scripts/ablation/submit_eval_probes_vcreg_loss.py [--dry-run]
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

EOS_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/loss/logs/train/runs"

RUNS = [
    ("ablation/loss/vcreg_var5",  7,     "2026-06-18_19-01-50"),
    ("ablation/loss/vcreg_var5",  42,    "2026-06-18_19-08-20"),
    ("ablation/loss/vcreg_var5",  137,   "2026-06-18_19-31-53"),
    ("ablation/loss/vcreg_var5",  1337,  "2026-06-18_19-40-40"),
    ("ablation/loss/vcreg_var5",  31337, "2026-06-18_19-44-17"),
    ("ablation/loss/vcreg_var10", 7,     "2026-06-18_20-17-27"),
    ("ablation/loss/vcreg_var10", 42,    "2026-06-18_20-18-48"),
    ("ablation/loss/vcreg_var10", 137,   "2026-06-18_20-19-30"),
    ("ablation/loss/vcreg_var10", 1337,  "2026-06-18_20-54-55"),
    ("ablation/loss/vcreg_var10", 31337, "2026-06-18_20-57-49"),
    ("ablation/loss/vcreg_var25", 7,     "2026-06-18_21-13-17"),
    ("ablation/loss/vcreg_var25", 42,    "2026-06-18_21-15-17"),
    ("ablation/loss/vcreg_var25", 137,   "2026-06-18_21-15-16"),
    ("ablation/loss/vcreg_var25", 1337,  "2026-06-18_21-18-04"),
    ("ablation/loss/vcreg_var25", 31337, "2026-06-18_21-18-03"),
    ("ablation/loss/vcreg_var50", 7,     "2026-06-18_21-18-39"),
    ("ablation/loss/vcreg_var50", 42,    "2026-06-18_21-38-52"),
    ("ablation/loss/vcreg_var50", 137,   "2026-06-18_21-37-28"),
    ("ablation/loss/vcreg_var50", 1337,  "2026-06-18_21-41-38"),
    ("ablation/loss/vcreg_var50", 31337, "2026-06-18_21-41-43"),
    ("ablation/loss/vcreg_cov5",  7,     "2026-06-18_21-52-51"),
    ("ablation/loss/vcreg_cov5",  42,    "2026-06-18_21-56-39"),
    ("ablation/loss/vcreg_cov5",  137,   "2026-06-18_22-12-34"),
    ("ablation/loss/vcreg_cov5",  1337,  "2026-06-18_22-15-42"),
    ("ablation/loss/vcreg_cov5",  31337, "2026-06-18_22-16-13"),
    ("ablation/loss/vcreg_cov10", 7,     "2026-06-18_22-15-23"),
    ("ablation/loss/vcreg_cov10", 42,    "2026-06-18_22-16-06"),
    ("ablation/loss/vcreg_cov10", 137,   "2026-06-18_22-17-53"),
    ("ablation/loss/vcreg_cov10", 1337,  "2026-06-18_22-19-08"),
    ("ablation/loss/vcreg_cov10", 31337, "2026-06-18_22-42-06"),
]

PROJECT_DIR = "/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD"
LOG_DIR     = Path(PROJECT_DIR) / "logs/condor_logs/ablation/eval_probes_vcreg_loss"
WRAPPER     = Path(PROJECT_DIR) / "scripts/ablation/wrapper_eval_probe_ablation.sh"
SUB_DIR     = Path(PROJECT_DIR) / "logs/condor_subs/ablation"


def find_best_checkpoint(run_timestamp: str) -> str:
    ckpt_dir = f"{EOS_BASE}/{run_timestamp}/checkpoints"
    result = subprocess.run(
        ["eos", "root://eosuser.cern.ch", "ls", ckpt_dir],
        capture_output=True, text=True
    )
    ckpts = sorted(
        f for f in result.stdout.splitlines()
        if f.startswith("epoch_") and f.endswith(".ckpt")
    )
    if not ckpts:
        raise FileNotFoundError(f"No epoch_*.ckpt in {ckpt_dir}")
    return f"{ckpt_dir}/{ckpts[-1]}"


def make_sub(experiment: str, seed: int, run_timestamp: str,
             ckpt_path: str, dry_run: bool) -> None:
    exp_short  = experiment.replace("ablation/loss/", "").replace("/", "_")
    job_name   = f"{exp_short}_seed{seed}"
    output_dir = f"{EOS_BASE}/{run_timestamp}"

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
    print(f"  {'[DRY]' if dry_run else '[SUB]'} {job_name}  ckpt={ckpt_path.split('/')[-1]}")

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

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(RUNS)} probe eval jobs\n")

    submitted = skipped = 0
    for experiment, seed, run_ts in RUNS:
        output_dir = f"{EOS_BASE}/{run_ts}"
        check = subprocess.run(
            ["eos", "root://eosuser.cern.ch", "ls", f"{output_dir}/probe_evaluation/"],
            capture_output=True, text=True
        )
        if "probe_results.json" in check.stdout:
            print(f"  [SKIP] {experiment.split('/')[-1]}_seed{seed} — already done")
            skipped += 1
            continue

        try:
            ckpt_path = find_best_checkpoint(run_ts)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
            continue

        make_sub(experiment, seed, run_ts, ckpt_path, dry_run=args.dry_run)
        submitted += 1

    print(f"\nDone: {submitted} submitted, {skipped} skipped.")


if __name__ == "__main__":
    main()
