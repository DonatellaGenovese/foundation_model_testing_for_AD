#!/usr/bin/env python3
"""
Submit Condor probe-evaluation jobs for all 15 AugSupCon ablation runs.

Each job runs eval_probes.py on one checkpoint and writes probe_results.json
back into the same run directory (alongside train.log, checkpoints/).

Usage:
    python3 scripts/ablation/submit_eval_probes_aug_supcon.py [--dry-run]
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

# ── Mapping: (experiment, seed) → run timestamp ──────────────────────────────
# Derived from config_tree.log run_name field in each run directory.

EOS_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/logs/train/runs"

RUNS = [
    # experiment                              seed   run_timestamp
    ("ablation/aug_supcon_random_feature",    7,     "2026-06-16_20-53-33"),
    ("ablation/aug_supcon_random_feature",    42,    "2026-06-16_21-12-47"),
    ("ablation/aug_supcon_random_feature",    137,   "2026-06-16_21-19-33"),
    ("ablation/aug_supcon_random_feature",    1337,  "2026-06-16_21-37-06"),
    ("ablation/aug_supcon_random_feature",    31337, "2026-06-16_22-34-12"),
    ("ablation/aug_supcon_random_particle",   7,     "2026-06-16_22-53-50"),
    ("ablation/aug_supcon_random_particle",   42,    "2026-06-16_23-01-30"),
    ("ablation/aug_supcon_random_particle",   137,   "2026-06-16_23-09-23"),
    ("ablation/aug_supcon_random_particle",   1337,  "2026-06-16_23-09-24"),
    ("ablation/aug_supcon_random_particle",   31337, "2026-06-16_23-13-51"),
    ("ablation/aug_supcon_physics",           7,     "2026-06-16_23-37-24"),
    ("ablation/aug_supcon_physics",           42,    "2026-06-16_23-41-51"),
    ("ablation/aug_supcon_physics",           137,   "2026-06-16_23-44-20"),
    ("ablation/aug_supcon_physics",           1337,  "2026-06-16_23-46-28"),
    ("ablation/aug_supcon_physics",           31337, "2026-06-16_23-46-57"),
]

PROJECT_DIR = "/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD"
LOG_DIR = Path(PROJECT_DIR) / "logs/condor_logs/ablation/eval_probes_aug_supcon"
WRAPPER = Path(PROJECT_DIR) / "scripts/ablation/wrapper_eval_probe_ablation.sh"
SUB_DIR = Path(PROJECT_DIR) / "logs/condor_subs/ablation"


def find_best_checkpoint(run_timestamp: str) -> str:
    """Return the epoch_*.ckpt path (highest epoch, not last.ckpt)."""
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
    best = ckpts[-1]  # highest epoch
    return f"{ckpt_dir}/{best}"


def make_sub(experiment: str, seed: int, run_timestamp: str, ckpt_path: str,
             dry_run: bool) -> str:
    """Generate and optionally submit a Condor .sub file. Returns job description string."""
    exp_short = experiment.replace("ablation/", "").replace("/", "_")
    job_name = f"{exp_short}_seed{seed}"
    output_dir = f"{EOS_BASE}/{run_timestamp}"

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

    return job_name


def main():
    parser = argparse.ArgumentParser(description="Submit AugSupCon probe eval jobs")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without submitting")
    args = parser.parse_args()

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(RUNS)} probe eval jobs\n")

    submitted = 0
    skipped = 0
    for experiment, seed, run_ts in RUNS:
        output_dir = f"{EOS_BASE}/{run_ts}"
        probe_results = f"{output_dir}/probe_evaluation/probe_results.json"

        # Skip if already done
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
