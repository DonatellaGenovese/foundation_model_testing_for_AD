#!/usr/bin/env python3
"""
Submit linear-probe evaluation jobs for the new_exp encoders.

One Condor job per (model, d_model); the wrapper loops over all 5 seeds
internally and aggregates. Results saved under:
    /eos/user/d/dgenoves/anomaly_pipeline/new_exp/probe_results/<run_dir>/
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
WRAPPER     = PROJECT_DIR / "scripts/new_exp/wrapper_eval_probes_new_exp.sh"
LOG_DIR     = PROJECT_DIR / "logs/condor_logs/new_exp/probe_eval"
EOS_NEW     = "/eos/user/d/dgenoves/anomaly_pipeline/new_exp"

DMODELS = [32, 64, 128, 256]

# Seed sets differ per campaign — see the same table in submit_training_new_exp.py.
# These match probe_results/<run_dir>/seed_* on EOS exactly.
SEEDS         = "7,42,12345,1337,31337"      # fallback
SEEDS_137     = "7,42,137,1337,31337"
SEEDS_LEGACY  = "0,1,2,3,4"

# model -> (hydra experiment stem, encoder run-dir stem)
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
    # The VICReg row of the paper. Must be probed from the same encoders the AD
    # table reports (ad_results/vicreg_*), or Table 8 and Tables 9-10 would
    # describe different models.
    "vicreg": {
        "experiment": "new_exp/vicreg_dmodel{dm}",
        "run_dir":    "vicreg_12class_nosparse_dmodel{dm}_cern",
    },
    "supcon": {
        "experiment": "new_exp/supcon_dmodel{dm}",
        "run_dir":    "supcon_12class_nosparse_dmodel{dm}_cern",
        "seeds":      SEEDS_137,
    },
    # The experiment here selects the probe *protocol* only — split, label, paths.
    # The encoder architecture comes from the checkpoint (load_from_checkpoint), so
    # this entry does not have to name the config that trained VCReg.
    #
    # It used to name new_exp/vcreg_dmodel{dm}, now archived. The replacement below
    # composes to the same label (v2_12class_nosparse_highlevel), the same
    # 1M/100k/100k split and the same 12 classes, so the protocol is unchanged.
    "vcreg": {
        "experiment": "vcreg_12class_nosparse_dmodel{dm}_cern",
        "run_dir":    "vcreg_12class_nosparse_dmodel{dm}_cern",
        "seeds":      SEEDS_LEGACY,
    },
    # vcreg_d256_old_arch removed: its config is in archive/configs/experiment_new_exp/
    # and no published probe came from it.
}


def submit_job(model: str, dm: int, dry_run: bool = False, seeds: str = SEEDS) -> None:
    experiment = MODELS[model]["experiment"].format(dm=dm)
    run_dir    = MODELS[model]["run_dir"].format(dm=dm)
    enc_dir    = f"{EOS_NEW}/{run_dir}"
    out_dir    = f"{EOS_NEW}/probe_results/{run_dir}"

    job_name = f"probe_{run_dir}"

    sub_content = f"""\
executable = {WRAPPER}

output = {LOG_DIR}/{job_name}.out
error  = {LOG_DIR}/{job_name}.err
log    = {LOG_DIR}/{job_name}.log

# No stream_output/stream_error: the CERN schedd rejects the submission outright
# since Nov 2025.

run_as_owner = True
+JobFlavour  = "nextweek"
getenv       = True
request_cpus = 16
request_gpus = 1
request_memory = 42000

Requirements = (TARGET.GPUs_GlobalMemoryMb >= 16000)

environment = "EXPERIMENT={experiment} ENCODER_DIR={enc_dir} OUTPUT_DIR={out_dir} SEEDS='{seeds}' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"

queue
"""

    sub_path = LOG_DIR / f"{job_name}.sub"
    sub_path.write_text(sub_content)

    if dry_run:
        print(f"  [DRY] {job_name}")
        print(f"        experiment = {experiment}")
        print(f"        encoder    = {enc_dir}")
        print(f"        output     = {out_dir}")
        return

    result = subprocess.run(["condor_submit", str(sub_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Submitted: {job_name}")
    else:
        print(f"  FAILED:    {job_name}\n{result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Submit new_exp probe-eval jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=None)
    parser.add_argument("--dmodels", nargs="+", type=int, default=None)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated encoder seeds. Default: the set that produced each "
             "model's published encoders (they differ per model).",
    )
    args = parser.parse_args()

    models  = args.models or list(MODELS)
    dmodels = args.dmodels or DMODELS

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total = len(models) * len(dmodels)
    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(models)} models × "
          f"{len(dmodels)} d_model = {total} jobs (5 seeds each, looped in wrapper)")
    print(f"Models  : {models}")
    print(f"d_models: {dmodels}")
    print(f"Seeds   : {args.seeds if args.seeds else 'per model'}")
    print(f"Output  : {EOS_NEW}/probe_results/<run_dir>/\n")

    for model in models:
        seeds = args.seeds or MODELS[model].get("seeds", SEEDS)
        print(f"[{model}] seeds={seeds}")
        for dm in dmodels:
            submit_job(model, dm, dry_run=args.dry_run, seeds=seeds)

    print(f"\nDone. {total} jobs {'(dry-run)' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
