#!/bin/bash
# =====================================================
#  HTCondor wrapper: multi-seed AE strategy comparison
#  for AugSupCon 12-class nosparse d_model=128
#
#  For each seed in 0..N_SEEDS-1:
#    - reads embeddings from BASE_DIR/seed_N/embeddings/
#    - runs 5 AE strategies (mse_qcd, auroc_all, auroc_cls12/13/14)
#    - saves results to BASE_DIR/seed_N/strategies/
#
#  Must be run AFTER full_pipeline_seeds_nosparse_dmodel128.sub completes.
#
#  Environment variables (set in .sub file):
#    BASE_DIR   - root dir where seed_N subdirs live
#    N_SEEDS    - number of seeds (default: 10)
# =====================================================

set -euo pipefail

echo "[`date`] Starting multi-seed strategies job on $(hostname)"
echo "[`date`] Running as $(whoami)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

BASE_DIR="${BASE_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/strategies_v2/nosparse_dmodel128_seeds}"
N_SEEDS="${N_SEEDS:-10}"

PHASE2_EXPERIMENT="anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern"
MODEL_TAG="aug_supcon_nosparse_dmodel128"
SIGNAL_CLASSES="12 13 14"

echo "[wrapper] Base dir           : ${BASE_DIR}"
echo "[wrapper] N seeds            : ${N_SEEDS}"
echo "[wrapper] Phase 2 experiment : ${PHASE2_EXPERIMENT}"
echo "[wrapper] Signal classes     : ${SIGNAL_CLASSES}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

for SEED in $(seq 0 $((N_SEEDS - 1))); do
    EMBEDDINGS_DIR="${BASE_DIR}/seed_${SEED}/embeddings"
    STRATEGIES_DIR="${BASE_DIR}/seed_${SEED}/strategies"

    echo ""
    echo "[`date`] ===== Seed ${SEED} / $((N_SEEDS - 1)) ====="
    echo "[wrapper] Embeddings : ${EMBEDDINGS_DIR}"
    echo "[wrapper] Output     : ${STRATEGIES_DIR}"

    if [ ! -d "${EMBEDDINGS_DIR}" ]; then
        echo "[wrapper] WARNING: embeddings dir not found, skipping seed ${SEED}"
        continue
    fi

    PYTHON_CMD="python scripts/run_anomaly_strategies.py \
        --phase2-experiment ${PHASE2_EXPERIMENT} \
        --embeddings-dir ${EMBEDDINGS_DIR} \
        --output-dir ${STRATEGIES_DIR} \
        --model-tag ${MODEL_TAG} \
        --signal-classes ${SIGNAL_CLASSES} \
        --seed ${SEED}"

    echo "[wrapper] Running: ${PYTHON_CMD}"

    apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        ${PYTHON_CMD}
    "

    echo "[`date`] Seed ${SEED} done"
done

echo ""
echo "[`date`] All ${N_SEEDS} seeds finished. Results in: ${BASE_DIR}/seed_N/strategies/"
exit 0
