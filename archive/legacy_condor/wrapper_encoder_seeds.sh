#!/bin/bash
# =====================================================
#  HTCondor wrapper: encoder training × N seeds (probe evaluation only, no AD)
#
#  Runs src/train.py for each seed 0..N_SEEDS-1.
#
#  Environment variables (set in .sub file):
#    EXPERIMENT   - Hydra experiment name (e.g. vicreg_15class_dmodel256_cern)
#    N_SEEDS      - Number of seeds (default: 10)
#    OUTPUT_DIR   - Root directory; seed_N subdirs created here
# =====================================================

set -euo pipefail

echo "[`date`] Starting encoder-seeds job on $(hostname)"
echo "[`date`] Running as $(whoami)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

EXPERIMENT="${EXPERIMENT:-vicreg_15class_dmodel256_cern}"
N_SEEDS="${N_SEEDS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/${EXPERIMENT}}"

echo "[wrapper] Experiment : ${EXPERIMENT}"
echo "[wrapper] N seeds    : ${N_SEEDS}"
echo "[wrapper] Output dir : ${OUTPUT_DIR}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

for SEED in $(seq 0 $((N_SEEDS - 1))); do
    SEED_OUT="${OUTPUT_DIR}/seed_${SEED}"
    echo ""
    echo "[`date`] ===== Seed ${SEED} / $((N_SEEDS - 1)) ====="
    echo "[wrapper] Output: ${SEED_OUT}"

    mkdir -p "${SEED_OUT}"

    PYTHON_CMD="python src/train.py \
        experiment=${EXPERIMENT} \
        seed=${SEED} \
        paths.output_dir=${SEED_OUT} \
        callbacks.model_checkpoint.dirpath=${SEED_OUT}/checkpoints \
        logger.mlflow.run_name=${EXPERIMENT}_seed${SEED}"

    echo "[wrapper] Running: ${PYTHON_CMD}"

    apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        ${PYTHON_CMD}
    "

    echo "[`date`] Seed ${SEED} done"
done

echo ""
echo "[`date`] All ${N_SEEDS} seeds finished. Results in: ${OUTPUT_DIR}"
exit 0
