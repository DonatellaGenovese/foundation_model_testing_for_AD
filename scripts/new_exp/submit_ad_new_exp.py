#!/usr/bin/env python3
"""
Submit anomaly-detection (encoder + AE on embeddings) jobs for the new_exp encoders.

One Condor job per (model, d_model); the wrapper loops over all 5 encoder seeds
internally (strategy: mse_normal, QCD-only AE) and aggregates. Results under:
    /eos/user/d/dgenoves/anomaly_pipeline/new_exp/ad_results/<run_dir>/
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "src/wrapper_encoder_seeds_anomaly.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/new_exp/ad"
EOS_NEW     = "/eos/user/d/dgenoves/anomaly_pipeline/new_exp"

# VCReg-aligned seed set (12345 replaces 137)
SEEDS   = "7,42,12345,1337,31337"
AE_SEED = 42
DMODELS = [32, 64, 128, 256]

# model -> (phase2 experiment, encoder run-dir stem)
MODELS = {
    "simclr": {
        "phase2":  "new_exp/anomaly_embedding_simclr",
        "run_dir": "simclr_12class_nosparse_dmodel{dm}_cern",
    },
    "vicreg_physics": {
        "phase2":  "new_exp/anomaly_embedding_vicreg",
        "run_dir": "vicreg_physics_12class_nosparse_dmodel{dm}_cern",
    },
    # The VICReg row of the paper. Trained with the reproducible-convergence config
    # (lr 3e-4, OneCycleLR warmup+cosine, early stopping off): all 20 runs reached
    # epoch 91-99 of the 100-epoch schedule, against a previous campaign whose seeds
    # peaked anywhere from epoch 0 to 83. Supersedes the augmentation-ablation rows
    # below, whose encoders converged about half the time.
    "vicreg": {
        "phase2":  "new_exp/anomaly_embedding_vicreg",
        "run_dir": "vicreg_12class_nosparse_dmodel{dm}_cern",
    },
    # VICReg augmentation ablation, d128 only. The three strategies differ in how
    # the positive pair is built: physics smears eta/pt, random_particle masks whole
    # particles, random_feature masks individual features.
    #
    # The `_warmup5` run dirs are NOT a warmup arm: the LR-warmup override was lost
    # to Condor's environment parsing (fixed in submit_training_new_exp.py), so all
    # five dirs ran with warmup_epochs=0 and config_tree.log confirms it. That makes
    # `physics_warmup5` simply the physics arm — the only physics encoder that exists
    # — and turns the other two into exact replicas of their base arm, which is worth
    # scoring precisely because it measures the noise floor of the AD chain itself.
    "vicreg_physics_d128": {
        "phase2":  "new_exp/anomaly_embedding_vicreg",
        "run_dir": "vicreg_physics_warmup5_12class_nosparse_dmodel128_cern",
    },
    "vicreg_random_particle_d128": {
        "phase2":  "new_exp/anomaly_embedding_vicreg",
        "run_dir": "vicreg_random_particle_12class_nosparse_dmodel128_cern",
    },
    "vicreg_random_particle_d128_rep": {
        "phase2":  "new_exp/anomaly_embedding_vicreg",
        "run_dir": "vicreg_random_particle_warmup5_12class_nosparse_dmodel128_cern",
    },
    "vicreg_random_feature_d128": {
        "phase2":  "new_exp/anomaly_embedding_vicreg",
        "run_dir": "vicreg_random_feature_12class_nosparse_dmodel128_cern",
    },
    "vicreg_random_feature_d128_rep": {
        "phase2":  "new_exp/anomaly_embedding_vicreg",
        "run_dir": "vicreg_random_feature_warmup5_12class_nosparse_dmodel128_cern",
    },
    "supcon": {
        "phase2":  "new_exp/anomaly_embedding_supcon",
        "run_dir": "supcon_12class_nosparse_dmodel{dm}_cern",
    },
    "vcreg": {
        "phase2":  "new_exp/anomaly_embedding_vcreg",
        "run_dir": "vcreg_12class_nosparse_dmodel{dm}_cern",
    },
    # Pilot: lr=1.5e-4, ckpt val/acc (d128 only — ignore --dmodels for other dims)
    "vcreg_lr1p5e4_acc": {
        "phase2":  "new_exp/anomaly_embedding_vcreg",
        "run_dir": "vcreg_12class_nosparse_dmodel128_cern_lr1p5e4_acc",
    },
    # d_ff=1024 ablation of the pilot (seed 137 first; pass --seeds)
    "vcreg_lr1p5e4_acc_dff1024": {
        "phase2":  "new_exp/anomaly_embedding_vcreg",
        "run_dir": "vcreg_12class_nosparse_dmodel128_cern_lr1p5e4_acc_dff1024",
    },
    "vcreg_d256_lr1p5e4_acc": {
        "phase2":  "new_exp/anomaly_embedding_vcreg",
        "run_dir": "vcreg_12class_nosparse_dmodel256_cern_lr1p5e4_acc",
    },
    "vcreg_d64_lr1p5e4_acc": {
        "phase2":  "new_exp/anomaly_embedding_vcreg",
        "run_dir": "vcreg_12class_nosparse_dmodel64_cern_lr1p5e4_acc",
    },
    "vcreg_d256_old_arch": {
        "phase2":  "new_exp/anomaly_embedding_vcreg",
        "run_dir": "vcreg_12class_nosparse_dmodel256_cern_old_arch",
    },
}


def submit_job(model: str, dm: int, seeds: str, dry_run: bool = False, smnorm: bool = False) -> None:
    phase2  = MODELS[model]["phase2"] + ("_smnorm" if smnorm else "")
    # Some pilot run_dirs are fixed (no {dm} placeholder)
    stem = MODELS[model]["run_dir"]
    run_dir = stem.format(dm=dm) if "{dm}" in stem else stem
    enc_dir = f"{EOS_NEW}/{run_dir}"
    out_dir = f"{EOS_NEW}/ad_results/{run_dir}"

    job_name = f"ad_{run_dir}"
    if seeds != SEEDS:
        job_name = f"{job_name}_seeds{seeds.replace(',', '-')}"

    sub_content = f"""\
executable = {WRAPPER}

output = {LOG_DIR}/{job_name}.out
error  = {LOG_DIR}/{job_name}.err
log    = {LOG_DIR}/{job_name}.log

# No stream_output/stream_error: the CERN schedd rejects the submission outright
# since Nov 2025. Logs land in the files above once the job ends.

run_as_owner = True
+JobFlavour  = "nextweek"
getenv       = True
request_cpus = 8
request_gpus = 1

Requirements = (TARGET.GPUs_GlobalMemoryMb >= 16000)

environment = "PHASE2_EXPERIMENT={phase2} ENCODER_SEEDS_DIR={enc_dir} OUTPUT_DIR={out_dir} AE_SEED={AE_SEED} ENCODER_SEEDS={seeds} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"

queue
"""

    sub_path = LOG_DIR / f"{job_name}.sub"
    sub_path.write_text(sub_content)

    if dry_run:
        print(f"  [DRY] {job_name}")
        print(f"        phase2   = {phase2}")
        print(f"        encoder  = {enc_dir}")
        print(f"        output   = {out_dir}")
        print(f"        seeds    = {seeds}")
        return

    result = subprocess.run(["condor_submit", str(sub_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Submitted: {job_name}  →  {result.stdout.strip()}")
    else:
        print(f"  FAILED:    {job_name}\n{result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Submit new_exp anomaly-detection jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=None)
    parser.add_argument("--dmodels", nargs="+", type=int, default=None)
    parser.add_argument(
        "--seeds",
        default=SEEDS,
        help="Comma-separated encoder seeds (default: all five)",
    )
    parser.add_argument(
        "--smnorm",
        action="store_true",
        help="Use the SM-only-normalised dataset (appends _smnorm to the phase2 experiment)",
    )
    args = parser.parse_args()

    models  = args.models or list(MODELS)
    dmodels = args.dmodels or DMODELS
    # Fixed run_dir pilots: only one "dmodel" slot needed
    fixed_run_dir = {
        m for m in models if "{dm}" not in MODELS[m]["run_dir"]
    }
    if fixed_run_dir and args.dmodels is None:
        # avoid 4 duplicate submits for fixed run_dirs
        dmodels_by_model = {m: ([128] if m in fixed_run_dir else dmodels) for m in models}
    else:
        dmodels_by_model = {m: dmodels for m in models}

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total = sum(len(dmodels_by_model[m]) for m in models)
    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {total} AD jobs")
    print(f"Models  : {models}")
    print(f"Seeds   : {args.seeds}  (AE seed fixed: {AE_SEED})")
    print(f"Output  : {EOS_NEW}/ad_results/<run_dir>/\n")

    for model in models:
        for dm in dmodels_by_model[model]:
            submit_job(model, dm, seeds=args.seeds, dry_run=args.dry_run, smnorm=args.smnorm)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
