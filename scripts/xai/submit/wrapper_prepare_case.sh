#!/bin/bash
# HTCondor wrapper — build the CASE held-out-signal dataset (vectorise + apply the
# SM-only normalisation). See scripts/prepare_case_smnorm.py for why the split
# manifest is built by hand rather than by make_split_manifest.
#
# Runs on a batch node rather than interactively because it reads roughly 25 GB of
# parquet: nine QCD files at ~2 GB plus seven signal files at ~1 GB.
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "[$(date)] preparing CASE signals — host $(hostname)"

cd ${PROJECT_DIR}

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python3 -u scripts/prepare_case_smnorm.py
"

echo "[$(date)] done"
