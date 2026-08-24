#!/bin/bash
# One-off: run SelfSup AugSupCon d256 seed_9 only

set -euo pipefail

echo "[`date`] Starting SelfSup d256 seed_9 on $(hostname)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
EXPERIMENT="aug_supcon_12class_nosparse_selfsup_dmodel256_cern"
OUTPUT_DIR="/eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/${EXPERIMENT}"
SEED=9
SEED_OUT="${OUTPUT_DIR}/seed_${SEED}"

mkdir -p "${SEED_OUT}"

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

cd ${PROJECT_DIR}

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    python src/train.py \
        experiment=${EXPERIMENT} \
        seed=${SEED} \
        paths.output_dir=${SEED_OUT} \
        callbacks.model_checkpoint.dirpath=${SEED_OUT}/checkpoints \
        logger.mlflow.run_name=${EXPERIMENT}_seed${SEED}
"

echo "[`date`] seed_9 done. Output: ${SEED_OUT}"
exit 0
