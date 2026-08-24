#!/usr/bin/env python3
"""
submit_vectorization_jobs.py

Use this script to batch submit vectorization jobs to Condor.

Reads Hydra configs, builds a global split manifest, splits it into chunks of ≤ MAX_FILES_PER_JOB,
and directly submits each chunk to Condor to run `vectorize_to_local` inside the Apptainer container.
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os
import json
import subprocess
import hydra
from math import ceil
from pathlib import Path
from omegaconf import DictConfig

from src.data.utils import load_global_filelist, make_split_manifest

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MAX_FILES_PER_JOB = 50
JOB_FLAVOUR = "tomorrow"  # ≈ 1 day runtime per job

PROJECT_DIR = Path(__file__).resolve().parents[1]
WRAPPER_SCRIPT = PROJECT_DIR / "src/data/wrapper_vectorize.sh"
LOG_DIR = PROJECT_DIR / "logs/condor_logs/vectorization"

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
@hydra.main(
    config_path="../configs",
    config_name="vectorize_preprocess.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):

    print("🟢 Hydra config composed successfully")
    # -----------------------------------------------------------------------------
    # LOAD CONFIGS
    # -----------------------------------------------------------------------------
    print("🟢 Loading Hydra configs...")

    data_cfg = cfg.data
    paths_cfg = cfg.paths

    LABEL = data_cfg.label
    TMP_VEC_DIR = Path(paths_cfg.tmp_data_dir) / LABEL / "vectorized"
    EOS_VEC_DIR = Path(paths_cfg.eos_data_dir) / LABEL / "vectorized"
    DATASET_DIR = Path(paths_cfg.dataset_dir)

    split_counts = data_cfg.train_val_test_split_per_class
    include_folders = [data_cfg.process_to_folder[c] for c in data_cfg.to_classify]

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------------
    # BUILD GLOBAL MANIFEST
    # -----------------------------------------------------------------------------
    print("\n🟢 Building global manifest...")
    global_filelist = load_global_filelist()

    manifest = make_split_manifest(
        global_filelist=global_filelist,
        split_counts=split_counts,
        include_folders=include_folders,
        seed=42,
    )

    manifest_path = EOS_VEC_DIR / "split_manifest_global.json"
    os.makedirs(EOS_VEC_DIR, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Wrote global manifest → {manifest_path}")

    # -----------------------------------------------------------------------------
    # FLATTEN MANIFEST INTO ENTRIES
    # -----------------------------------------------------------------------------
    entries = []
    for folder, splits in manifest.items():
        for split_name, files in splits.items():
            for fname in files:
                target_path = EOS_VEC_DIR / split_name / folder / fname.replace(".parquet", "_x.npy")
                if not target_path.exists():
                    entries.append((folder, split_name, fname))

    n_jobs = ceil(len(entries) / MAX_FILES_PER_JOB)
    print(f"\n🟡 Total files: {len(entries)} → {n_jobs} jobs (≤{MAX_FILES_PER_JOB} files/job)")

    # Extract experiment name from Hydra overrides (passed as experiment=<name>)
    import sys as _sys
    experiment_name = next(
        (a.split("=", 1)[1] for a in _sys.argv[1:] if a.startswith("experiment=")),
        "fm_testing_18class_highlevel",
    )

    # -----------------------------------------------------------------------------
    # SUBMIT ALL JOBS
    # -----------------------------------------------------------------------------
    for i in range(n_jobs):
        start = i * MAX_FILES_PER_JOB
        end = start + MAX_FILES_PER_JOB
        chunk = entries[start:end]
        submit_job(i, chunk, experiment_name)

    print(f"\n✅ All {n_jobs} jobs submitted to Condor.")

# -----------------------------------------------------------------------------
# FUNCTION TO SUBMIT A SINGLE JOB
# -----------------------------------------------------------------------------
def submit_job(job_idx, chunk, experiment_name="fm_testing_18class_highlevel"):
    # Build submanifest dict
    submanifest = {}
    for folder, split_name, fname in chunk:
        submanifest.setdefault(folder, {"train": [], "val": [], "test": []})
        submanifest[folder][split_name].append(fname)

    # Save submanifest to a temporary JSON file
    manifest_path = LOG_DIR / f"manifest_{job_idx:04d}.json"
    with open(manifest_path, "w") as f:
        json.dump(submanifest, f, indent=2)

    log_out = LOG_DIR / f"job_{job_idx:04d}.out"
    log_err = LOG_DIR / f"job_{job_idx:04d}.err"
    log_log = LOG_DIR / f"job_{job_idx:04d}.log"

    # Run the wrapper with the manifest path as argument
    submit_content = f"""\
executable = {WRAPPER_SCRIPT}
arguments  = {manifest_path} {PROJECT_DIR} {experiment_name}
initialdir = {LOG_DIR}

output = {log_out}
error  = {log_err}
log    = {log_log}

# No stream_output/stream_error: the CERN schedd refuses the whole submission
# ("stream_out and stream_err are no longer supported ... Failed to commit job
# submission into the queue"). The rejection happens at commit time, not while the
# file is parsed, so condor_submit -dry-run accepts it and only a real submit fails.
# Logs land in the files above once the job ends.

run_as_owner = True
+JobFlavour = "{JOB_FLAVOUR}"
getenv = True
request_cpus = 4
environment = "WRAPPER_DEBUG=1; OMP_NUM_THREADS=4; MKL_NUM_THREADS=4"

queue
"""

    sub_path = LOG_DIR / f"vectorize_job_{job_idx:04d}.sub"
    with open(sub_path, "w") as f:
        f.write(submit_content)

    # check=True, not False: a rejected submission used to be ignored here, so the
    # script printed "Submitted job ..." for every chunk and then "All N jobs
    # submitted" while the queue stayed empty. Fail on the first refusal instead.
    subprocess.run(["condor_submit", str(sub_path)], check=True)
    print(f"🚀 Submitted job {job_idx:04d} ({len(chunk)} files)")

if __name__ == "__main__":
    main()
