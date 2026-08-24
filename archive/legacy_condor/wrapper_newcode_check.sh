#!/bin/bash
# =====================================================
#  Sanity check: train one seed with new code and run
#  eval_probes, to verify results are consistent with
#  existing d128 checkpoints trained with old code.
#
#  Model: VCReg d128, seed 0
#  Output: /eos/user/d/dgenoves/anomaly_pipeline/newcode_check/vcreg_d128_seed0/
# =====================================================

set -euo pipefail

echo "[`date`] Starting new-code sanity check on $(hostname)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

EXPERIMENT="vcreg_12class_nosparse_dmodel128_cern"
OUTPUT_DIR="/eos/user/d/dgenoves/anomaly_pipeline/newcode_check/vcreg_d128_seed0"
SEED=0

mkdir -p "${OUTPUT_DIR}/checkpoints"

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

cd ${PROJECT_DIR}

# Step 1: Train encoder
echo "[`date`] Step 1: Training encoder (seed=${SEED})..."
apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    python src/train.py \
        experiment=${EXPERIMENT} \
        seed=${SEED} \
        paths.output_dir=${OUTPUT_DIR} \
        callbacks.model_checkpoint.dirpath=${OUTPUT_DIR}/checkpoints \
        logger.mlflow.run_name=${EXPERIMENT}_newcode_seed${SEED}
"

# Step 2: Find best checkpoint and run eval_probes
CKPT=$(find "${OUTPUT_DIR}/checkpoints" -name "epoch_*.ckpt" | sort | tail -1)
echo "[`date`] Step 2: Running eval_probes on ${CKPT}..."

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    python src/eval_probes.py \
        experiment=${EXPERIMENT} \
        ckpt_path=${CKPT} \
        seed=${SEED} \
        paths.output_dir=${OUTPUT_DIR}/probe_eval
"

echo "[`date`] Done. Results in: ${OUTPUT_DIR}"
exit 0
