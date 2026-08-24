#!/usr/bin/env python3
"""
Submit weight_decay ablation training jobs to HTCondor.

Standard d=128 architecture (n_heads=8, n_layers=6, d_ff=512, dropout=0.1) for all 4 models.
weight_decay in {0, 1e-5, 1e-3, 1e-2} × 5 seeds = 80 jobs.

Default weight decay: VCReg/VICReg=1e-5, AugSupCon/SelfSupCon=1e-2.
Learning rates fixed at defaults: VCReg/VICReg=1e-3, AugSupCon/SelfSupCon=1e-4.

Runs saved to: /eos/user/d/dgenoves/anomaly_pipeline/ablation/wd/logs/

Usage:
    python3 scripts/ablation/submit_training_ablation_wd.py [--dry-run]
    python3 scripts/ablation/submit_training_ablation_wd.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_training_ablation_wd.py --wds 0 1e2
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/training_wd"

SEEDS    = [7, 42, 137, 1337, 31337]
MODELS   = ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
WD_SLUGS = ["0", "1e5", "1e3", "1e2"]

WD_VALUES = {
    "0":   0.0,
    "1e5": 1e-5,
    "1e3": 1e-3,
    "1e2": 1e-2,
}

DEFAULT_WD = {
    "vcreg":      "1e5",
    "aug_supcon": "1e2",
    "selfsupcon": "1e2",
    "vicreg":     "1e5",
}

EXPERIMENTS = {
    (model, slug): f"ablation/training/{model}_wd_{slug}"
    for model in MODELS
    for slug in WD_SLUGS
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
    parser.add_argument("--wds", nargs="+", choices=WD_SLUGS, default=None,
                        help="Weight decay slug: 0, 1e5=1e-5, 1e3=1e-3, 1e2=1e-2")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--flavour", default="nextweek",
                        help="HTCondor JobFlavour (e.g. tomorrow, longlunch, nextweek)")
    args = parser.parse_args()

    models = args.models or MODELS
    wds    = args.wds    or WD_SLUGS
    seeds  = args.seeds  or SEEDS

    to_submit = [(m, slug) for m in models for slug in wds]
    total = len(to_submit) * len(seeds)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(to_submit)} configs × {len(seeds)} seeds = {total} jobs")
    print(f"Models: {models}")
    print(f"WDs:    {[WD_VALUES[s] for s in wds]}")
    print(f"Seeds:  {seeds}")
    print(f"Flavour: {args.flavour}")
    print(f"Defaults: {', '.join(f'{m}={WD_VALUES[DEFAULT_WD[m]]}' for m in models)}")
    print(f"EOS logs: /eos/user/d/dgenoves/anomaly_pipeline/ablation/wd/logs/")
    print()

    for model, slug in to_submit:
        exp = EXPERIMENTS[(model, slug)]
        marker = " (default)" if slug == DEFAULT_WD[model] else ""
        print(f"[{model} wd={WD_VALUES[slug]:g}{marker}]")
        for seed in seeds:
            submit_job(exp, seed, dry_run=args.dry_run, flavour=args.flavour)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
