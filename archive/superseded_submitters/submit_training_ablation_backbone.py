#!/usr/bin/env python3
"""
Submit backbone architecture ablation training jobs to HTCondor.

Ablates n_layers, n_heads, d_ff, dropout at d_model=128 for 4 models:
  VCReg, AugSupCon, SelfSupCon, VICReg.

41 configs × 5 seeds = 205 jobs total.

Usage:
    python3 scripts/ablation/submit_training_ablation_backbone.py [--dry-run]
    python3 scripts/ablation/submit_training_ablation_backbone.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_training_ablation_backbone.py --params layers dropout
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/ablation/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/ablation/training_backbone"

SEEDS = [7, 42, 137, 1337, 31337]

# All experiment configs grouped by (model, param, value)
EXPERIMENTS = {
    "vcreg": {
        "base":        "ablation/training/vcreg_arch_base",
        "layers_2":    "ablation/training/vcreg_arch_layers_2",
        "layers_3":    "ablation/training/vcreg_arch_layers_3",
        "layers_6":    "ablation/training/vcreg_arch_layers_6",
        "layers_8":    "ablation/training/vcreg_arch_layers_8",
        "heads_4":     "ablation/training/vcreg_arch_heads_4",
        "dff_512":     "ablation/training/vcreg_arch_dff_512",
        "dff_2048":    "ablation/training/vcreg_arch_dff_2048",
        "dropout_0":   "ablation/training/vcreg_arch_dropout_0",
        "dropout_10":  "ablation/training/vcreg_arch_dropout_10",
        "dropout_30":  "ablation/training/vcreg_arch_dropout_30",
    },
    "aug_supcon": {
        "base":        "ablation/training/aug_supcon_arch_base",
        "layers_3":    "ablation/training/aug_supcon_arch_layers_3",
        "layers_4":    "ablation/training/aug_supcon_arch_layers_4",
        "layers_8":    "ablation/training/aug_supcon_arch_layers_8",
        "heads_4":     "ablation/training/aug_supcon_arch_heads_4",
        "dff_512":     "ablation/training/aug_supcon_arch_dff_512",
        "dff_2048":    "ablation/training/aug_supcon_arch_dff_2048",
        "dropout_0":   "ablation/training/aug_supcon_arch_dropout_0",
        "dropout_10":  "ablation/training/aug_supcon_arch_dropout_10",
        "dropout_30":  "ablation/training/aug_supcon_arch_dropout_30",
    },
    "selfsupcon": {
        "base":        "ablation/training/selfsupcon_arch_base",
        "layers_4":    "ablation/training/selfsupcon_arch_layers_4",
        "layers_6":    "ablation/training/selfsupcon_arch_layers_6",
        "heads_4":     "ablation/training/selfsupcon_arch_heads_4",
        "dff_512":     "ablation/training/selfsupcon_arch_dff_512",
        "dff_2048":    "ablation/training/selfsupcon_arch_dff_2048",
        "dropout_0":   "ablation/training/selfsupcon_arch_dropout_0",
        "dropout_10":  "ablation/training/selfsupcon_arch_dropout_10",
        "dropout_30":  "ablation/training/selfsupcon_arch_dropout_30",
    },
    "vicreg": {
        "base":        "ablation/training/vicreg_arch_base",
        "layers_2":    "ablation/training/vicreg_arch_layers_2",
        "layers_3":    "ablation/training/vicreg_arch_layers_3",
        "layers_6":    "ablation/training/vicreg_arch_layers_6",
        "layers_8":    "ablation/training/vicreg_arch_layers_8",
        "heads_8":     "ablation/training/vicreg_arch_heads_8",
        "dff_256":     "ablation/training/vicreg_arch_dff_256",
        "dff_1024":    "ablation/training/vicreg_arch_dff_1024",
        "dropout_0":   "ablation/training/vicreg_arch_dropout_0",
        "dropout_10":  "ablation/training/vicreg_arch_dropout_10",
        "dropout_30":  "ablation/training/vicreg_arch_dropout_30",
    },
}

PARAM_GROUPS = {
    "base":    ["base"],
    "layers":  ["layers_2", "layers_3", "layers_4", "layers_6", "layers_8"],
    "heads":   ["heads_4", "heads_8"],
    "dff":     ["dff_256", "dff_512", "dff_1024", "dff_2048"],
    "dropout": ["dropout_0", "dropout_10", "dropout_20", "dropout_23", "dropout_30"],
}


def submit_job(experiment: str, seed: int, dry_run: bool = False) -> None:
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
+JobFlavour  = "nextweek"
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
    parser.add_argument("--models",  nargs="+", choices=list(EXPERIMENTS), default=None)
    parser.add_argument("--params",  nargs="+", choices=list(PARAM_GROUPS), default=None)
    parser.add_argument("--seeds",   nargs="+", type=int, default=None)
    args = parser.parse_args()

    models = args.models or list(EXPERIMENTS)
    seeds  = args.seeds  or SEEDS

    # Build list of experiments to submit
    to_submit = []
    for model in models:
        for variant, exp_path in EXPERIMENTS[model].items():
            if args.params:
                # Keep if variant matches any requested param group
                if not any(variant.startswith(p.rstrip("s")) or variant == p
                           for p in args.params):
                    # Check if variant key starts with any param group key
                    match = False
                    for pg in args.params:
                        group_keys = PARAM_GROUPS.get(pg, [])
                        if variant in group_keys or variant == pg:
                            match = True
                            break
                    if not match:
                        continue
            to_submit.append((model, variant, exp_path))

    total = len(to_submit) * len(seeds)
    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(to_submit)} configs × {len(seeds)} seeds = {total} jobs")
    print(f"Models : {models}")
    print(f"Seeds  : {seeds}")
    print()

    for model, variant, exp_path in to_submit:
        print(f"[{model}/{variant}]")
        for seed in seeds:
            submit_job(exp_path, seed, dry_run=args.dry_run)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
