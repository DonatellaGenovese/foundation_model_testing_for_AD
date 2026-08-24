#!/usr/bin/env python3
"""
Submit linear probe evaluation jobs for all lr ablation training runs.

Standard d=128 architecture for all 4 models.
lr in {1e-4, 5e-4, 1e-3, 3e-3} × 5 seeds = 80 runs.

Probe output: inside each training run at {run}/probe_evaluation/

Usage:
    python3 scripts/ablation/submit_eval_probes_lr.py [--dry-run]
    python3 scripts/ablation/submit_eval_probes_lr.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_eval_probes_lr.py --lrs 1e4 1e3 --force
"""

import argparse
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

EOS_TRAIN_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/lr/logs/train/runs"
PROJECT_DIR    = Path(__file__).resolve().parents[2]
LOG_DIR        = PROJECT_DIR / "logs/condor_logs/ablation/eval_probes_lr"
SUB_DIR        = PROJECT_DIR / "logs/condor_subs/ablation"
WRAPPER        = PROJECT_DIR / "scripts/ablation/wrapper_eval_probe_ablation.sh"

SEEDS = [7, 42, 137, 1337, 31337]

LR_SLUGS = ["1e4", "5e4", "1e3", "3e3"]

EXPERIMENT_MAP = {
    (model, slug): f"ablation/training/{model}_lr_{slug}"
    for model in ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
    for slug in LR_SLUGS
}

CLI_TO_MODEL = {
    "vcreg":      "VCReg",
    "aug_supcon": "AugmentedSupCon",
    "selfsupcon": "AugmentedSelfSupCon",
    "vicreg":     "VICReg",
}

MODEL_TO_SHORT = {v: k for k, v in CLI_TO_MODEL.items()}

LR_TO_SLUG = {
    1e-4: "1e4",
    5e-4: "5e4",
    1e-3: "1e3",
    3e-3: "3e3",
}


def lr_to_slug(lr: float) -> Optional[str]:
    for val, slug in LR_TO_SLUG.items():
        if abs(lr - val) / val < 1e-3:
            return slug
    return None


def get_run_info(run_dir: Path):
    cfg = run_dir / ".hydra" / "config.yaml"
    if not cfg.exists():
        return None
    try:
        text = cfg.read_text()
        model = next(
            (l.split("_target_:")[-1].strip().split(".")[-1]
             .replace("COLLIDE2V", "").replace("LitModule", "")
             for l in text.split("\n") if "_target_:" in l and "LitModule" in l),
            None,
        )
        lr = next(
            (float(l.split(":")[-1].strip())
             for l in text.split("\n") if re.match(r"\s{4}lr:", l)),
            None,
        )
        seed = next(
            (int(l.split(":")[-1].strip())
             for l in text.split("\n") if re.match(r"^seed:", l.strip())),
            None,
        )
        if seed is None:
            match = re.search(r"^seed:\s*(\d+)", text, re.MULTILINE)
            seed = int(match.group(1)) if match else None
        slug = lr_to_slug(lr) if lr is not None else None
        if not model or slug is None or seed is None:
            return None
        return {"model": model, "lr_slug": slug, "seed": seed, "run_dir": run_dir}
    except Exception:
        return None


def find_best_checkpoint(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt")) if ckpt_dir.exists() else []
    return str(ckpts[-1]) if ckpts else None


def already_done(run_dir: Path) -> bool:
    return (run_dir / "probe_evaluation" / "probe_results.json").exists()


def make_sub(run_info: dict, experiment: str, ckpt_path: str, dry_run: bool) -> None:
    model_short = MODEL_TO_SHORT[run_info["model"]]
    job_name = f"{model_short}_lr{run_info['lr_slug']}_seed{run_info['seed']}"
    output_dir = str(run_info["run_dir"])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = SUB_DIR / f"eval_probe_lr_{job_name}.sub"
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
    parser.add_argument("--lrs", nargs="+", choices=LR_SLUGS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    filter_models = {CLI_TO_MODEL[m] for m in args.models} if args.models else None
    filter_lrs    = set(args.lrs) if args.lrs else None
    filter_seeds  = set(args.seeds) if args.seeds else None

    train_base = Path(EOS_TRAIN_BASE)
    if not train_base.exists():
        print(f"ERROR: train base not found: {train_base}")
        return

    all_runs = [get_run_info(p) for p in sorted(train_base.iterdir()) if p.is_dir()]
    all_runs = [r for r in all_runs if r is not None]

    if filter_models:
        all_runs = [r for r in all_runs if r["model"] in filter_models]
    if filter_lrs:
        all_runs = [r for r in all_runs if r["lr_slug"] in filter_lrs]
    if filter_seeds:
        all_runs = [r for r in all_runs if r["seed"] in filter_seeds]

    best: dict[tuple, dict] = {}
    for r in all_runs:
        key = (r["model"], r["lr_slug"], r["seed"])
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
        model_short = MODEL_TO_SHORT[r["model"]]
        experiment = EXPERIMENT_MAP.get((model_short, r["lr_slug"]))
        if experiment is None:
            print(f"  [WARN] No experiment config for {r['model']} lr={r['lr_slug']}")
            missing += 1
            continue

        if already_done(r["run_dir"]) and not args.force:
            print(f"  [SKIP] {r['model']} lr={r['lr_slug']} seed={r['seed']} — already done ({r['run_dir'].name})")
            skipped += 1
            continue

        print(f"  run_dir={r['run_dir'].name}")
        make_sub(r, experiment, r["ckpt"], dry_run=args.dry_run)
        submitted += 1

    print(f"\nDone: {submitted} submitted, {skipped} skipped, {missing} missing config.")


if __name__ == "__main__":
    main()
