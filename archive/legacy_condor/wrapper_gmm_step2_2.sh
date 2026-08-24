#!/bin/bash
# HTCondor wrapper: GMM Step 2.2 — Physical Profiles per Cluster
#
# Environment variables:
#   STEP1_DIR      - step1 output dir (has PCA + GMM models)
#   CKPT_PATH      - encoder checkpoint
#   VECTORIZED_DIR - vectorized test data root
#   OUTPUT_DIR     - output dir for step2.2
#   PCA_N          - PCA dimension (0 = no PCA)

set -euo pipefail
echo "[$(date)] Starting GMM Step 2.2 on $(hostname)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
EOS_BASE=/eos/user/d/dgenoves/anomaly_pipeline/xai_gmm_mocktest
VEC_DIR=/eos/user/d/dgenoves/foundation_model_testing_data/v2_nosparse_higgs_allsm_highlevel/vectorized/test
CKPT=/eos/user/d/dgenoves/anomaly_pipeline/xai_gmm_mocktest/logs/train/runs/2026-06-23_10-04-54/checkpoints/epoch_048.ckpt

STEP1_DIR="${STEP1_DIR:-${EOS_BASE}/gmm_analysis/seed_7/step1_pca64}"
OUTPUT_DIR="${OUTPUT_DIR:-${EOS_BASE}/gmm_analysis/seed_7/step2_2_pca64}"
CKPT_PATH="${CKPT_PATH:-${CKPT}}"
PCA_N="${PCA_N:-64}"

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs"

PYTHON_CMD="python scripts/xai_mocktest/gmm_step2_2_physical_profiles.py \
    --step1-dir      ${STEP1_DIR} \
    --ckpt-path      ${CKPT_PATH} \
    --vectorized-dir ${VEC_DIR} \
    --output-dir     ${OUTPUT_DIR} \
    --pca-n          ${PCA_N} \
    --max-per-class  20000"

echo "STEP1_DIR  : ${STEP1_DIR}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "PCA_N      : ${PCA_N}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    ${PYTHON_CMD}
"
EXIT_CODE=$?
echo "[$(date)] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
