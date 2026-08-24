#!/bin/bash
# =====================================================
#  HTCondor wrapper: GMM anomaly explanation
#
#  Environment variables (set in the .sub file):
#    CKPT_PATH        - Path to Phase 1 encoder checkpoint
#    SM_EXPERIMENT    - Phase 1 Hydra experiment config (15-class SM)
#    BSM_EXPERIMENT   - Phase 2 Hydra experiment config (BSM classes)
#    OUTPUT_DIR       - Root directory for outputs
#    K_VALUES         - Comma-separated K values to sweep (default: 10,15,20,25)
#    COV_TYPES        - Comma-separated covariance types (default: diag,full)
#    MAX_GMM_SAMPLES  - Max SM samples used for GMM fitting (default: 500000)
#    SEED             - Random seed (default: 42)
# =====================================================

set -euo pipefail

echo "[`date`] Starting Condor job on $(hostname)"
echo "[`date`] Running as $(whoami)"
echo "Working directory: $(pwd)"

# --- Paths ---
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

# --- Config from environment (with defaults) ---
CKPT_PATH="${CKPT_PATH}"
SM_EXPERIMENT="${SM_EXPERIMENT:-aug_supcon_12class_nosparse_dmodel128_cern}"
BSM_EXPERIMENT="${BSM_EXPERIMENT:-anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern}"
OUTPUT_DIR="${OUTPUT_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/gmm_explain}"
K_VALUES="${K_VALUES:-6,10,12,15,20}"
COV_TYPES="${COV_TYPES:-diag,full}"
MAX_GMM_SAMPLES="${MAX_GMM_SAMPLES:-500000}"
MAX_EVENTS="${MAX_EVENTS:-}"
SEED="${SEED:-42}"

echo "[wrapper] Checkpoint      : ${CKPT_PATH}"
echo "[wrapper] SM experiment   : ${SM_EXPERIMENT}"
echo "[wrapper] BSM experiment  : ${BSM_EXPERIMENT}"
echo "[wrapper] Output dir      : ${OUTPUT_DIR}"
echo "[wrapper] K values        : ${K_VALUES}"
echo "[wrapper] Cov types       : ${COV_TYPES}"
echo "[wrapper] Max GMM samples : ${MAX_GMM_SAMPLES}"
echo "[wrapper] Max events      : ${MAX_EVENTS:-unlimited}"
echo "[wrapper] Seed            : ${SEED}"

# --- Go to project directory ---
cd ${PROJECT_DIR}

# --- Apptainer flags ---
APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

# --- Build python command ---
PYTHON_CMD="python src/gmm_explain.py \
    --ckpt-path ${CKPT_PATH} \
    --sm-experiment ${SM_EXPERIMENT} \
    --bsm-experiment ${BSM_EXPERIMENT} \
    --output-dir ${OUTPUT_DIR} \
    --k-values ${K_VALUES} \
    --cov-types ${COV_TYPES} \
    --max-gmm-samples ${MAX_GMM_SAMPLES} \
    --seed ${SEED}"
if [ -n "${MAX_EVENTS}" ]; then
    PYTHON_CMD="${PYTHON_CMD} --max-events ${MAX_EVENTS}"
fi

echo "[wrapper] Running: ${PYTHON_CMD}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    ${PYTHON_CMD}
"

EXIT_CODE=$?
echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
