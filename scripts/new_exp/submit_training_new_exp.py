#!/usr/bin/env python3
"""
Submit production encoder training jobs (d_model scan, 5 seeds) to HTCondor.

Models × 4 d_model {32,64,128,256} × 5 seeds.
Currently: simclr, supcon, vcreg, vicreg, ce.
Config from ablation-selected hyperparameters, scaled to the large split
(1M/100k/100k). Outputs are written in encoder_seeds style:

    /eos/user/d/dgenoves/anomaly_pipeline/new_exp/<run_dir>/seed_<seed>/
        checkpoints/epoch_XXX.ckpt   (best only, save_last=false)
        config_tree.log
        tags.log

Reuses scripts/new_exp/wrapper_train.sh (EXPERIMENT / SEED / EXTRA_OVERRIDES).

Usage:
    python3 scripts/new_exp/submit_training_new_exp.py --dry-run
    python3 scripts/new_exp/submit_training_new_exp.py
    python3 scripts/new_exp/submit_training_new_exp.py --models simclr --dmodels 128
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/new_exp/wrapper_train.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/new_exp/training"
EOS_BASE    = "/eos/user/d/dgenoves/anomaly_pipeline/new_exp"

DMODELS = [32, 64, 128, 256]

# The five published campaigns do NOT share a seed set, and the difference is not
# cosmetic: passing the wrong set makes the submitter train encoders that no result
# in the paper was computed from. Each entry below is the set actually present under
# <EOS_BASE>/<run_dir>/seed_*, cross-checked against the seeds the AD aggregated
# (<EOS_BASE>/ad_results/<run_dir>/encoder_seed_*).
#
#   supcon, simclr  ->  137    (the sweep predates the VICReg rerun)
#   vicreg          ->  12345  (137 was replaced when the campaign was redone)
#   vcreg, ce       ->  0..4   (trained before the seed convention changed)
SEEDS         = [7, 42, 12345, 1337, 31337]      # fallback
SEEDS_137     = [7, 42, 137, 1337, 31337]
SEEDS_LEGACY  = [0, 1, 2, 3, 4]

# model -> (experiment stem, output run-dir stem)
MODELS = {
    "simclr": {
        "experiment": "new_exp/simclr_dmodel{dm}",
        "run_dir":    "simclr_12class_nosparse_dmodel{dm}_cern",
        "seeds":      SEEDS_137,
    },
    "vicreg_physics": {
        "experiment": "new_exp/vicreg_physics_dmodel{dm}",
        "run_dir":    "vicreg_physics_12class_nosparse_dmodel{dm}_cern",
    },
    "supcon": {
        "experiment": "new_exp/supcon_dmodel{dm}",
        "run_dir":    "supcon_12class_nosparse_dmodel{dm}_cern",
        "seeds":      SEEDS_137,
    },
    # The VCReg row of the paper, and the encoder the interpretability stage runs on.
    #
    # ITS CONFIG IS THE PRE-new_exp ONE, ON PURPOSE. The published encoders under
    # new_exp/vcreg_12class_nosparse_dmodel*_cern/seed_{0..4} are byte-identical
    # copies of anomaly_pipeline/encoder_seeds/ (md5 checked on all four d_model),
    # and their config_tree.log records lr=1.5e-4, wd=5e-4, 50 epochs, checkpoint on
    # val/acc — the settings of vcreg_12class_nosparse_dmodel{dm}_cern.
    #
    # new_exp/vcreg_dmodel{dm} is a *different* training (lr 1e-3, wd 1e-5, 40
    # epochs, checkpoint on val/vcreg_loss). It was run and then set aside; those
    # encoders live in new_exp/archive_vcreg_20260728/encoders/. Pointing this entry
    # at it would silently reproduce an encoder no published number came from, so
    # the config was archived to archive/configs/experiment_new_exp/.
    "vcreg": {
        "experiment": "vcreg_12class_nosparse_dmodel{dm}_cern",
        "run_dir":    "vcreg_12class_nosparse_dmodel{dm}_cern",
        "seeds":      SEEDS_LEGACY,
    },
    # Supervised cross-entropy baseline. Same story as vcreg: its configs predate
    # new_exp/ and are the ones that produced the published checkpoints, which is
    # why they carry the older `fm_testing_` naming. No linear probe is run on it:
    # being a classifier, the per-class AUROC of Table 8 comes from its own test
    # predictions.
    "ce": {
        "experiment": "fm_testing_12class_nosparse_dmodel{dm}_cern",
        "run_dir":    "fm_testing_12class_nosparse_dmodel{dm}_cern",
        "seeds":      SEEDS_LEGACY,
    },
    # The five VCReg pilots that once lived here (lr1p5e4_acc, _dff1024, d256/d64
    # variants, old_arch) were attempts to reach the published VCReg quality from the
    # new_exp recipe. None of them did; their encoders sit in
    # new_exp/archive_vcreg_20260728/encoders/ and their configs in
    # archive/configs/experiment_new_exp/. The published VCReg is the "vcreg" entry
    # above, trained from the pre-new_exp config.
    #
    # --- VICReg stability sweep (Step 1): augmentation x warmup, d128 only ---
    # Production arch (new_exp/_final_vicreg) held fixed; only augmentation_type
    # and warmup_epochs vary. physics is the existing baseline (see "vicreg_physics").
    "vicreg_random_particle": {
        "experiment": "new_exp/vicreg_random_particle_dmodel128",
        "run_dir":    "vicreg_random_particle_12class_nosparse_dmodel128_cern",
    },
    "vicreg_random_feature": {
        "experiment": "new_exp/vicreg_random_feature_dmodel128",
        "run_dir":    "vicreg_random_feature_12class_nosparse_dmodel128_cern",
    },
    # Reproducible-convergence rerun of the VICReg d_model scan: lr 3e-4,
    # OneCycleLR (warmup + cosine) instead of ReduceLROnPlateau, early stopping
    # off. Supersedes `vicreg_physics`, whose encoders no longer exist on disk and
    # whose seeds converged only about half the time. See
    # configs/experiment/new_exp/_vicreg.yaml for the evidence per change.
    "vicreg": {
        "experiment": "new_exp/vicreg_dmodel{dm}",
        "run_dir":    "vicreg_12class_nosparse_dmodel{dm}_cern",
    },
    "vicreg_physics_warmup5": {
        "experiment": "new_exp/vicreg_physics_dmodel128",
        "run_dir":    "vicreg_physics_warmup5_12class_nosparse_dmodel128_cern",
        "extra_overrides": "model.warmup_epochs=5",
    },
    "vicreg_random_particle_warmup5": {
        "experiment": "new_exp/vicreg_random_particle_dmodel128",
        "run_dir":    "vicreg_random_particle_warmup5_12class_nosparse_dmodel128_cern",
        "extra_overrides": "model.warmup_epochs=5",
    },
    "vicreg_random_feature_warmup5": {
        "experiment": "new_exp/vicreg_random_feature_dmodel128",
        "run_dir":    "vicreg_random_feature_warmup5_12class_nosparse_dmodel128_cern",
        "extra_overrides": "model.warmup_epochs=5",
    },
}


def submit_job(model: str, dm: int, seed: int, dry_run: bool = False) -> None:
    spec       = MODELS[model]
    experiment = spec["experiment"].format(dm=dm)
    run_dir    = spec["run_dir"].format(dm=dm)
    out_dir    = f"{EOS_BASE}/{run_dir}/seed_{seed}"

    job_name = f"{run_dir}_seed{seed}"
    # Multiple overrides are space-separated, so the value MUST be single-quoted in
    # the `environment` line below: Condor also uses spaces to separate variables,
    # and would otherwise read everything after the first space as a new variable.
    # This silently dropped `model.warmup_epochs=5` from the warmup5 arms.
    overrides = f"hydra.run.dir={out_dir}"
    if spec.get("extra_overrides"):
        overrides += f" {spec['extra_overrides']}"

    sub_content = f"""\
executable = {WRAPPER}

output = {LOG_DIR}/{job_name}.out
error  = {LOG_DIR}/{job_name}.err
log    = {LOG_DIR}/{job_name}.log

# No stream_output/stream_error: the CERN schedd rejects the submission outright
# since Nov 2025. Logs land in the files above once the job ends.

run_as_owner = True
+JobFlavour  = "tomorrow"
getenv       = True
request_cpus = 8
request_gpus = 1

Requirements = (TARGET.GPUs_GlobalMemoryMb >= 16000)

environment = "EXPERIMENT={experiment} SEED={seed} EXTRA_OVERRIDES='{overrides}' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"

queue
"""

    sub_path = LOG_DIR / f"{job_name}.sub"
    sub_path.write_text(sub_content)

    if dry_run:
        print(f"  [DRY] {job_name}")
        print(f"        experiment = {experiment}")
        print(f"        out_dir    = {out_dir}")
        return

    result = subprocess.run(["condor_submit", str(sub_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Submitted: {job_name}")
    else:
        print(f"  FAILED:    {job_name}\n{result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Submit production encoder training jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=None)
    parser.add_argument("--dmodels", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    models  = args.models or list(MODELS)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # An explicit --seeds wins; otherwise take the set that produced this model's
    # published encoders, never a single global default.
    def seeds_for(model: str):
        return args.seeds or MODELS[model].get("seeds", SEEDS)

    # Fixed run_dirs (no {dm}): submit once, ignore multi-dmodel unless user forced --dmodels
    def dmodels_for(model: str):
        if "{dm}" not in MODELS[model]["experiment"]:
            return args.dmodels or [128]
        return args.dmodels or DMODELS

    total = sum(len(dmodels_for(m)) * len(seeds_for(m)) for m in models)
    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {total} jobs")
    print(f"Models  : {models}")
    print(f"Seeds   : {'(--seeds) ' + str(args.seeds) if args.seeds else 'per model'}")
    print(f"Output  : {EOS_BASE}/<run_dir>/seed_<seed>/")
    print()

    for model in models:
        for dm in dmodels_for(model):
            seeds = seeds_for(model)
            print(f"[{model} d_model={dm}]  seeds={seeds}")
            for seed in seeds:
                submit_job(model, dm, seed, dry_run=args.dry_run)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
