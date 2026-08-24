#!/bin/bash
# =====================================================
#  HTCondor wrapper: raw AE baseline × 10 seeds — 12-class nosparse + Higgs
#
#  Runs all 5 strategies on raw input features (no encoder) for each seed.
#  Results saved to strategies_v2/nosparse_dmodel128_seeds/raw/seed_N/
# =====================================================

set -euo pipefail

echo "[`date`] Starting raw baseline nosparse job on $(hostname)"
echo "[`date`] Running as $(whoami)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

BASE_DIR=/eos/user/d/dgenoves/anomaly_pipeline/strategies_v2/nosparse_dmodel128_seeds/raw
RAW_EXPERIMENT=anomaly_qcd_vs_higgs_raw_nosparse_cern
N_SEEDS=10

echo "[wrapper] Raw experiment : ${RAW_EXPERIMENT}"
echo "[wrapper] Base dir       : ${BASE_DIR}"
echo "[wrapper] N seeds        : ${N_SEEDS}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

for SEED in $(seq 0 $((N_SEEDS - 1))); do
    SEED_OUT="${BASE_DIR}/seed_${SEED}"
    mkdir -p "${SEED_OUT}"

    echo ""
    echo "[`date`] ===== Seed ${SEED} / $((N_SEEDS - 1)) ====="
    echo "[wrapper] Output: ${SEED_OUT}"

    PYTHON_CMD="python scripts/run_anomaly_strategies.py \
        --phase2-experiment ${RAW_EXPERIMENT} \
        --skip-embedding \
        --raw-experiment ${RAW_EXPERIMENT} \
        --signal-classes 12 13 14 \
        --seed ${SEED} \
        --output-dir ${SEED_OUT} \
        --model-tag raw_nosparse"

    echo "[wrapper] Running: ${PYTHON_CMD}"

    apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        ${PYTHON_CMD}
    "

    echo "[`date`] Seed ${SEED} done"
done

echo ""
echo "[`date`] All ${N_SEEDS} seeds finished. Results in: ${BASE_DIR}"
exit 0
