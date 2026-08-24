#!/usr/bin/env python3
"""
Submit linear probe evaluation jobs for all n_layers ablation training runs.

Standard architecture (d_model=128, n_heads=8, d_ff=512, dropout=0.1) for all 4 models.
num_layers in {2, 4, 6} × 5 seeds = 60 runs.

Probe output: inside each training run at {run}/probe_evaluation/

Usage:
    python3 scripts/ablation/submit_eval_probes_nlayers.py [--dry-run]
    python3 scripts/ablation/submit_eval_probes_nlayers.py --models vcreg aug_supcon
    python3 scripts/ablation/submit_eval_probes_nlayers.py --nlayers 2 4
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

EOS_TRAIN_BASE = "/eos/user/d/dgenoves/anomaly_pipeline/ablation/nlayers/logs/train/runs"
PROJECT_DIR    = Path(__file__).resolve().parents[2]
LOG_DIR        = PROJECT_DIR / "logs/condor_logs/ablation/eval_probes_nlayers"
SUB_DIR        = PROJECT_DIR / "logs/condor_subs/ablation"
WRAPPER        = PROJECT_DIR / "scripts/ablation/wrapper_eval_probe_ablation.sh"

SEEDS = [7, 42, 137, 1337, 31337]

EXPERIMENT_MAP = {
    ("VCReg",               2): "ablation/training/vcreg_nlayers_2",
    ("VCReg",               4): "ablation/training/vcreg_nlayers_4",
    ("VCReg",               6): "ablation/training/vcreg_nlayers_6",
    ("AugmentedSupCon",     2): "ablation/training/aug_supcon_nlayers_2",
    ("AugmentedSupCon",     4): "ablation/training/aug_supcon_nlayers_4",
    ("AugmentedSupCon",     6): "ablation/training/aug_supcon_nlayers_6",
    ("AugmentedSelfSupCon", 2): "ablation/training/selfsupcon_nlayers_2",
    ("AugmentedSelfSupCon", 4): "ablation/training/selfsupcon_nlayers_4",
    ("AugmentedSelfSupCon", 6): "ablation/training/selfsupcon_nlayers_6",
    ("VICReg",              2): "ablation/training/vicreg_nlayers_2",
    ("VICReg",              4): "ablation/training/vicreg_nlayers_4",
    ("VICReg",              6): "ablation/training/vicreg_nlayers_6",
}

CLI_TO_MODEL = {
    "vcreg":      "VCReg",
    "aug_supcon": "AugmentedSupCon",
    "selfsupcon": "AugmentedSelfSupCon",
    "vicreg":     "VICReg",
}


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
        nlayers = next(
            (int(l.split("num_layers:")[-1].strip())
             for l in text.split("\n") if "  num_layers:" in l),
            None,
        )
        seed = next(
            (int(l.split("seed:")[-1].strip())
             for l in text.split("\n") if l.strip().startswith("seed:")),
            None,
        )
        if not model or nlayers is None or seed is None:
            return None
        return {"model": model, "nlayers": nlayers, "seed": seed, "run_dir": run_dir}
    except Exception:
        return None


def find_best_checkpoint(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt")) if ckpt_dir.exists() else []
    return str(ckpts[-1]) if ckpts else None


def already_done(run_dir: Path) -> bool:
    return (run_dir / "probe_evaluation" / "probe_results.json").exists()


def make_sub(run_info: dict, experiment: str, ckpt_path: str, dry_run: bool) -> None:
    model_short = run_info["model"].replace("Augmented", "").lower()
    job_name = f"{model_short}_nlayers{run_info['nlayers']}_seed{run_info['seed']}"
    output_dir = str(run_info["run_dir"])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = SUB_DIR / f"eval_probe_nlayers_{job_name}.sub"
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
    parser.add_argument("--models", nargs="+", choices=list(CLI_TO_MODEL), default=None)
    parser.add_argument("--nlayers", nargs="+", type=int, choices=[2, 4, 6], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    filter_models  = {CLI_TO_MODEL[m] for m in args.models} if args.models else None
    filter_nlayers = set(args.nlayers) if args.nlayers else None
    filter_seeds   = set(args.seeds) if args.seeds else None

    train_base = Path(EOS_TRAIN_BASE)
    all_runs = [get_run_info(p) for p in sorted(train_base.iterdir()) if p.is_dir()]
    all_runs = [r for r in all_runs if r is not None]

    if filter_models:
        all_runs = [r for r in all_runs if r["model"] in filter_models]
    if filter_nlayers:
        all_runs = [r for r in all_runs if r["nlayers"] in filter_nlayers]
    if filter_seeds:
        all_runs = [r for r in all_runs if r["seed"] in filter_seeds]

    best: dict[tuple, dict] = {}
    for r in all_runs:
        key = (r["model"], r["nlayers"], r["seed"])
        ckpt = find_best_checkpoint(r["run_dir"])
        if ckpt and key not in best:
            r["ckpt"] = ckpt
            best[key] = r

    print(f"{'DRY RUN — ' if args.dry_run else ''}Found {len(best)} runs to evaluate\n")

    submitted = skipped = missing = 0
    for key in sorted(best):
        r = best[key]
        experiment = EXPERIMENT_MAP.get((r["model"], r["nlayers"]))
        if experiment is None:
            print(f"  [WARN] No experiment config for {r['model']} nlayers={r['nlayers']}")
            missing += 1
            continue

        if already_done(r["run_dir"]):
            print(f"  [SKIP] {r['model']} nlayers={r['nlayers']} seed={r['seed']} — already done")
            skipped += 1
            continue

        make_sub(r, experiment, r["ckpt"], dry_run=args.dry_run)
        submitted += 1

    print(f"\nDone: {submitted} submitted, {skipped} skipped, {missing} missing config.")


if __name__ == "__main__":
    main()
