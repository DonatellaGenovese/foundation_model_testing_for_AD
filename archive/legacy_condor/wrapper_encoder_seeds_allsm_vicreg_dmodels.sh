#!/bin/bash
# =====================================================
#  HTCondor wrapper: VICReg encoder training (10 seeds) + AE allsm
#  for a given d_model variant of 12-class nosparse
#
#  Phase 1: train encoder from scratch for each seed
#  Phase 2: run AE anomaly detection with allsm config
#
#  Environment variables (set in .sub file):
#    PHASE1_EXPERIMENT  - Hydra encoder training config
#    PHASE2_EXPERIMENT  - Hydra AE allsm config
#    ENCODER_SEEDS_DIR  - Where to save encoder checkpoints
#    OUTPUT_DIR         - Where to save AE results
#    N_SEEDS            - Number of seeds (default: 10)
# =====================================================

set -euo pipefail

echo "[`date`] Starting VICReg encoder+AE allsm job on $(hostname)"
echo "[`date`] Running as $(whoami)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

PHASE1_EXPERIMENT="${PHASE1_EXPERIMENT:-vicreg_12class_nosparse_dmodel64_cern}"
PHASE2_EXPERIMENT="${PHASE2_EXPERIMENT:-anomaly_qcd_vs_higgs_embedding_vicreg_nosparse_allsm_dmodel64_cern}"
ENCODER_SEEDS_DIR="${ENCODER_SEEDS_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/${PHASE1_EXPERIMENT}}"
OUTPUT_DIR="${OUTPUT_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/strategies_v2/${PHASE1_EXPERIMENT}_allsm_seeds}"
N_SEEDS="${N_SEEDS:-10}"

echo "[wrapper] Phase1 experiment  : ${PHASE1_EXPERIMENT}"
echo "[wrapper] Phase2 experiment  : ${PHASE2_EXPERIMENT}"
echo "[wrapper] Encoder seeds dir  : ${ENCODER_SEEDS_DIR}"
echo "[wrapper] Output dir         : ${OUTPUT_DIR}"
echo "[wrapper] N seeds            : ${N_SEEDS}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

for SEED in $(seq 0 $((N_SEEDS - 1))); do
    SEED_CKPT_DIR="${ENCODER_SEEDS_DIR}/seed_${SEED}"
    SEED_OUT="${OUTPUT_DIR}/seed_${SEED}"

    echo ""
    echo "[`date`] ===== Seed ${SEED} / $((N_SEEDS - 1)) ====="
    echo "[wrapper] Encoder out : ${SEED_CKPT_DIR}"
    echo "[wrapper] AE out      : ${SEED_OUT}"

    mkdir -p "${SEED_CKPT_DIR}" "${SEED_OUT}"

    # Phase 1: train encoder
    TRAIN_CMD="python src/train.py \
        experiment=${PHASE1_EXPERIMENT} \
        seed=${SEED} \
        paths.output_dir=${SEED_CKPT_DIR} \
        callbacks.model_checkpoint.dirpath=${SEED_CKPT_DIR}/checkpoints \
        logger.mlflow.run_name=${PHASE1_EXPERIMENT}_seed${SEED}"

    echo "[wrapper] Training encoder: ${TRAIN_CMD}"
    apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        ${TRAIN_CMD}
    "

    # Pick best checkpoint
    BEST_CKPT=$(ls "${SEED_CKPT_DIR}/checkpoints"/epoch_*.ckpt 2>/dev/null | sort | tail -n 1 || true)
    if [ -z "${BEST_CKPT}" ]; then
        BEST_CKPT="${SEED_CKPT_DIR}/checkpoints/last.ckpt"
    fi
    echo "[wrapper] Best checkpoint: ${BEST_CKPT}"

    # Phase 2: AE anomaly detection with allsm config
    AE_CMD="python src/train_full_anomaly_pipeline.py \
        --phase1-experiment ${PHASE1_EXPERIMENT} \
        --phase2-experiment ${PHASE2_EXPERIMENT} \
        --output-dir ${SEED_OUT} \
        --seed ${SEED} \
        --phase1-ckpt ${BEST_CKPT}"

    echo "[wrapper] AE anomaly detection: ${AE_CMD}"
    apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        ${AE_CMD}
    "

    echo "[`date`] Seed ${SEED} done"
done

echo ""
echo "[`date`] All ${N_SEEDS} seeds finished."
echo "[wrapper] Encoder checkpoints: ${ENCODER_SEEDS_DIR}"
echo "[wrapper] AE results         : ${OUTPUT_DIR}"
exit 0
