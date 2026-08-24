#!/bin/bash
# =====================================================
#  HTCondor wrapper: GMM anomaly detection across all encoder seeds
#
#  Environment variables (set in .sub file):
#    PHASE2_EXPERIMENT   - Hydra config name (allsm variant for GMM)
#    ENCODER_SEEDS_DIR   - Dir with seed_0/, seed_1/, ... encoder checkpoints
#    OUTPUT_DIR          - Root output directory
#    GMM_SEED            - Fixed seed for GMM fitting (default: 42)
#    ENCODER_SEEDS       - Comma-separated list (default: 0,1,2,3,4,5,6,7,8,9)
# =====================================================

set -euo pipefail

echo "[`date`] Starting GMM encoder-seeds anomaly job on $(hostname)"
echo "[`date`] Running as $(whoami)"
echo "Working directory: $(pwd)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

PHASE2_EXPERIMENT="${PHASE2_EXPERIMENT:-anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_allsm_dmodel128_cern}"
ENCODER_SEEDS_DIR="${ENCODER_SEEDS_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/aug_supcon_12class_nosparse_dmodel128_cern}"
OUTPUT_DIR="${OUTPUT_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/gmm_encoder_seeds/augsupcon_nosparse_dmodel128}"
GMM_SEED="${GMM_SEED:-42}"
ENCODER_SEEDS="${ENCODER_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
ENCODER_SEEDS="${ENCODER_SEEDS//,/ }"   # normalise comma-separated to space-separated

echo "[wrapper] Phase2 experiment  : ${PHASE2_EXPERIMENT}"
echo "[wrapper] Encoder seeds dir  : ${ENCODER_SEEDS_DIR}"
echo "[wrapper] Output dir         : ${OUTPUT_DIR}"
echo "[wrapper] GMM seed           : ${GMM_SEED}"
echo "[wrapper] Encoder seeds      : ${ENCODER_SEEDS}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

PYTHON_CMD="python scripts/run_gmm_encoder_seeds.py \
    --phase2-experiment ${PHASE2_EXPERIMENT} \
    --encoder-seeds-dir ${ENCODER_SEEDS_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --gmm-seed ${GMM_SEED} \
    --encoder-seeds ${ENCODER_SEEDS}"

echo "[wrapper] Running: ${PYTHON_CMD}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    ${PYTHON_CMD}
"

EXIT_CODE=$?
echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
