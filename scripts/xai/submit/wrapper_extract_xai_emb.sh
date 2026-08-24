#!/bin/bash
# HTCondor wrapper — full-SM embedding extraction for the XAI pipeline.
#   SEED    encoder seed   (default 3)
#   DMODEL  embedding dim  (default 256)
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
SEED="${SEED:-3}"
DMODEL="${DMODEL:-256}"

echo "[$(date)] XAI embedding extraction — seed ${SEED}, d=${DMODEL} — host $(hostname)"
apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python3 scripts/xai/extract_xai_embeddings.py --seed ${SEED} --dmodel ${DMODEL}
"
echo "[$(date)] done"
