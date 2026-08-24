#!/bin/bash
# HTCondor wrapper: GMM Step 1 — Latent Space Structure of SM
#
# Environment variables:
#   EMBEDDING_DIR  - path to encoder embeddings (train/val/test_embeddings.npz)
#   OUTPUT_DIR     - root output dir for step1 artifacts

set -euo pipefail

echo "[`date`] Starting GMM Step 1 on $(hostname)"
echo "Working directory: $(pwd)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

EMBEDDING_DIR="${EMBEDDING_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/xai_gmm_mocktest/ad_results/seed_7/encoder_seed_7/embeddings}"
OUTPUT_DIR="${OUTPUT_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/xai_gmm_mocktest/gmm_analysis/seed_7/step1_pca64}"
PCA_N="${PCA_N:-64}"

APPTAINER_FLAGS="--bind /afs:/afs --bind /eos:/eos --writable-tmpfs"

PYTHON_CMD="python scripts/xai_mocktest/gmm_step1_latent_structure.py \
    --embedding-dir ${EMBEDDING_DIR} \
    --output-dir    ${OUTPUT_DIR} \
    --pca-n         ${PCA_N}"

echo "EMBEDDING_DIR : ${EMBEDDING_DIR}"
echo "OUTPUT_DIR    : ${OUTPUT_DIR}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    ${PYTHON_CMD}
"

EXIT_CODE=$?
echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
