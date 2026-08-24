#!/bin/bash
# =====================================================
#  HTCondor wrapper for ablation training jobs
#  Environment vars (set in .sub file):
#    EXPERIMENT  — e.g. ablation/aug_supcon_random_feature
#    SEED        — integer seed
# =====================================================

set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "[$(date)] Starting ablation job"
echo "[$(date)] Host: $(hostname) | User: $(whoami)"
echo "[$(date)] Experiment: ${EXPERIMENT} | Seed: ${SEED}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv \
    --bind /afs:/afs \
    --bind /eos:/eos \
    --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

# Include seed in MLflow run_name for easy filtering
BASENAME=$(basename ${EXPERIMENT})
RUN_NAME="${BASENAME}_seed${SEED}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    python src/train.py \
        experiment=${EXPERIMENT} \
        seed=${SEED} \
        logger.mlflow.run_name=${RUN_NAME} \
        ${EXTRA_OVERRIDES:-}
"

EXIT_CODE=$?
echo "[$(date)] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
