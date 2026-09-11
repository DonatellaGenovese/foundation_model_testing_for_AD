#!/bin/bash
# HTCondor wrapper — XAI step 01 (edit paths as needed)
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
EMBEDDINGS_DIR=/eos/user/d/dgenoves/anomaly_pipeline/gmm_encoder_seeds/vcreg_nosparse_dmodel128/encoder_seed_0/embeddings
OUTPUT_DIR=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d128_seed0/01_select_k

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python scripts/xai/01_select_k.py \
    --embeddings-dir ${EMBEDDINGS_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --k-values 4 6 8 10 12 \
    --ari-threshold 0.8
"
