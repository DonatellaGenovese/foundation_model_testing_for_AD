#!/bin/bash
# HTCondor wrapper — build the smnorm dataset (SM-only normalisation, all 15 classes)
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
echo "[$(date)] preprocess smnorm — host $(hostname)"
apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python3 scripts/preprocess_smnorm.py
"
echo "[$(date)] done"
