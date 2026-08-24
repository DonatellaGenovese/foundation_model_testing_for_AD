#!/usr/bin/env python3
"""
Submit linear probe evaluation jobs for all scheduler ablation training runs.

Standard d=128 architecture for all 4 models.
scheduler in {plateau, plateau_agg, cosine, none} × 5 seeds = 80 runs.

Probe output: inside each training run at {run}/probe_evaluation/

Usage:
    python3 scripts/ablation/submit_eval_probes_sched.py [--dry-run]
    python3 scripts/ablation/submit_eval_probes_sched.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_eval_probes_sched.py --scheds cosine --force
"""

import argparse
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

EOS_TRAIN_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/sched/logs/train/runs"
PROJECT_DIR    = Path(__file__).resolve().parents[2]
LOG_DIR        = PROJECT_DIR / "logs/condor_logs/ablation/eval_probes_sched"
SUB_DIR        = PROJECT_DIR / "logs/condor_subs/ablation"
WRAPPER        = PROJECT_DIR / "scripts/ablation/wrapper_eval_probe_ablation.sh"

SEEDS = [7, 42, 137, 1337, 31337]

SCHEDS = ["plateau", "plateau_agg", "cosine", "none"]

EXPERIMENT_MAP = {
    (model, sched): f"ablation/training/{model}_sched_{sched}"
    for model in ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
    for sched in SCHEDS
}

CLI_TO_MODEL = {
    "vcreg":      "VCReg",
    "aug_supcon": "AugmentedSupCon",
    "selfsupcon": "AugmentedSelfSupCon",
    "vicreg":     "VICReg",
}

MODEL_TO_SHORT = {v: k for k, v in CLI_TO_MODEL.items()}


def get_run_info(run_dir: Path) -> Optional[dict]:
    cfg = run_dir / ".hydra" / "config.yaml"
    if not cfg.exists():
        return None
    try:
        text = cfg.read_text()
        run_name_m = re.search(r"run_name:\s*(\S+)", text)
        if not run_name_m:
            return None
        m = re.match(r"(.+)_sched_(plateau_agg|plateau|cosine|none)_seed(\d+)", run_name_m.group(1))
        if not m:
            return None

        model_short = m.group(1)
        if model_short not in CLI_TO_MODEL:
            return None

        return {
            "model": CLI_TO_MODEL[model_short],
            "model_short": model_short,
            "sched": m.group(2),
            "seed": int(m.group(3)),
            "run_dir": run_dir,
        }
    except Exception:
        return None


def find_best_checkpoint(run_dir: Path) -> Optional[str]:
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt")) if ckpt_dir.exists() else []
    return str(ckpts[-1]) if ckpts else None


def already_done(run_dir: Path) -> bool:
    return (run_dir / "probe_evaluation" / "probe_results.json").exists()


def make_sub(run_info: dict, experiment: str, ckpt_path: str, dry_run: bool, flavour: str) -> None:
    job_name = f"{run_info['model_short']}_sched{run_info['sched']}_seed{run_info['seed']}"
    output_dir = str(run_info["run_dir"])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = SUB_DIR / f"eval_probe_sched_{job_name}.sub"
    sub_content = textwrap.dedent(f"""\
        executable = {WRAPPER}

        output = {LOG_DIR}/{job_name}_$(ClusterId).out
        error  = {LOG_DIR}/{job_name}_$(ClusterId).err
        log    = {LOG_DIR}/{job_name}_$(ClusterId).log

        run_as_owner = True
        +JobFlavour  = "{flavour}"
        getenv       = True
        request_cpus = 8
        request_gpus = 1
        Requirements = (TARGET.GPUs_GlobalMemoryMb >= 16000)

        environment = "EXPERIMENT={experiment} CKPT_PATH={ckpt_path} SEED={run_info['seed']} OUTPUT_DIR={output_dir}"

        queue
    """)

    sub_path.write_text(sub_content)
    print(f"  {'[DRY]' if dry_run else '[SUB]'} {job_name}  ckpt={Path(ckpt_path).name}")

    if not dry_run:
        result = subprocess.run(
            ["condor_submit", str(sub_path)],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr.strip()}")
        else:
            print(f"    → {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-submit even if probe_results.json already exists")
    parser.add_argument("--models", nargs="+", choices=list(CLI_TO_MODEL), default=None)
    parser.add_argument("--scheds", nargs="+", choices=SCHEDS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--flavour", default="tomorrow",
                        help="HTCondor JobFlavour (default: tomorrow)")
    args = parser.parse_args()

    filter_models = {CLI_TO_MODEL[m] for m in args.models} if args.models else None
    filter_scheds = set(args.scheds) if args.scheds else None
    filter_seeds  = set(args.seeds) if args.seeds else None

    train_base = Path(EOS_TRAIN_BASE)
    if not train_base.exists():
        print(f"ERROR: train base not found: {train_base}")
        return

    all_runs = [get_run_info(p) for p in sorted(train_base.iterdir()) if p.is_dir()]
    all_runs = [r for r in all_runs if r is not None]

    if filter_models:
        all_runs = [r for r in all_runs if r["model"] in filter_models]
    if filter_scheds:
        all_runs = [r for r in all_runs if r["sched"] in filter_scheds]
    if filter_seeds:
        all_runs = [r for r in all_runs if r["seed"] in filter_seeds]

    best: dict[tuple, dict] = {}
    for r in all_runs:
        key = (r["model"], r["sched"], r["seed"])
        ckpt = find_best_checkpoint(r["run_dir"])
        if not ckpt:
            continue
        r["ckpt"] = ckpt
        r["mtime"] = r["run_dir"].stat().st_mtime
        if key not in best or r["mtime"] > best[key]["mtime"]:
            best[key] = r

    print(f"{'DRY RUN — ' if args.dry_run else ''}Found {len(best)} runs to evaluate\n")

    submitted = skipped = missing = 0
    for key in sorted(best):
        r = best[key]
        experiment = EXPERIMENT_MAP.get((r["model_short"], r["sched"]))
        if experiment is None:
            print(f"  [WARN] No experiment config for {r['model']} sched={r['sched']}")
            missing += 1
            continue

        if already_done(r["run_dir"]) and not args.force:
            print(f"  [SKIP] {r['model']} sched={r['sched']} seed={r['seed']} — already done ({r['run_dir'].name})")
            skipped += 1
            continue

        print(f"  run_dir={r['run_dir'].name}")
        make_sub(r, experiment, r["ckpt"], dry_run=args.dry_run, flavour=args.flavour)
        submitted += 1

    print(f"\nDone: {submitted} submitted, {skipped} skipped, {missing} missing config.")


if __name__ == "__main__":
    main()
