#!/bin/bash
# HTCondor wrapper — ARI vs (sample size, n_init) diagnostic.
#   DMODEL  embedding dimension (default 256)
#   SEED    encoder seed        (default 3)
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

DMODEL="${DMODEL:-256}"
SEED="${SEED:-3}"

NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
RUN=vcreg_12class_nosparse_dmodel${DMODEL}_cern
EMB=${NE}/xai_embeddings_smnorm/${RUN}/encoder_seed_${SEED}/embeddings
OUT=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/k_diagnostic/ari_scaling_d${DMODEL}_seed${SEED}

echo "[$(date)] ARI scaling — d=${DMODEL}, seed ${SEED}, host $(hostname)"
[ -e "${EMB}/train_embeddings.npz" ] || { echo "MISSING: ${EMB}/train_embeddings.npz"; exit 1; }

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python scripts/xai/diagnose_ari_scaling.py \
    --embeddings-dir ${EMB} \
    --output-dir ${OUT} \
    --k 12 \
    --n-train 200000 600000 1150000 \
    --n-init 1 5 \
    --n-restarts 4
"

echo "[$(date)] done"
