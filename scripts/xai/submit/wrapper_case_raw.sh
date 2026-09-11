#!/bin/bash
# Raw-feature AE baseline on a held-out signal set, all five seeds looped inside.
#
#   DATASET  case | newsig   (default case) -- the set infer_new_signals_raw.py scores.
#            case_raw.sub and newsig_raw.sub each set it explicitly.
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
SEEDS="${SEEDS:-7 42 137 1337 31337}"
DATASET="${DATASET:-case}"

echo "[$(date)] raw AE on ${DATASET} signals — seeds: ${SEEDS}, host $(hostname)"
cd ${PROJECT_DIR}

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -uo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  rc=0
  for S in ${SEEDS}; do
    echo \"===== raw seed \${S} =====\"
    python3 -u scripts/infer_new_signals_raw.py --dataset ${DATASET} --seed \${S} || {
      echo \"seed \${S} FAILED\"; rc=1; }
  done
  exit \${rc}
"
echo "[$(date)] done"
