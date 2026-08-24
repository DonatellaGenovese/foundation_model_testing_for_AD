#!/bin/bash
# =====================================================
#  HTCondor wrapper: AugSupCon nosparse d_model=128 — single run
#
#  Phase 1: AugSupCon encoder (12 classes, no sparse, d_model=128)
#  Phase 2: Autoencoder anomaly detection on embeddings (Trial 36 params)
#
#  Output: /eos/user/d/dgenoves/anomaly_pipeline/strategies_v2/augsupcon_nosparse_dmodel128/
# =====================================================

set -euo pipefail

echo "[`date`] Starting strategies_v2 nosparse dmodel128 job on $(hostname)"
echo "[`date`] Running as $(whoami)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

PHASE1_EXPERIMENT="aug_supcon_12class_nosparse_dmodel128_cern"
PHASE2_EXPERIMENT="anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern"
OUTPUT_DIR="/eos/user/d/dgenoves/anomaly_pipeline/strategies_v2/augsupcon_nosparse_dmodel128"
SEED="${SEED:-0}"

echo "[wrapper] Phase 1 experiment : ${PHASE1_EXPERIMENT}"
echo "[wrapper] Phase 2 experiment : ${PHASE2_EXPERIMENT}"
echo "[wrapper] Output dir         : ${OUTPUT_DIR}/seed_${SEED}"
echo "[wrapper] Seed               : ${SEED}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

PYTHON_CMD="python src/train_full_anomaly_pipeline.py \
    --phase1-experiment ${PHASE1_EXPERIMENT} \
    --phase2-experiment ${PHASE2_EXPERIMENT} \
    --output-dir ${OUTPUT_DIR}/seed_${SEED} \
    --seed ${SEED}"

echo "[wrapper] Running: ${PYTHON_CMD}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    ${PYTHON_CMD}
"

echo "[`date`] Done. Results in: ${OUTPUT_DIR}/seed_${SEED}"
exit 0
