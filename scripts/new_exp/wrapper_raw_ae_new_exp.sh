#!/bin/bash
# =====================================================
#  HTCondor wrapper: raw AE baseline — new_exp
#
#  Trains AE on raw features (no encoder) via
#  scripts/run_anomaly_strategies.py --skip-embedding.
#
#  Environment variables (set in .sub file):
#    RAW_EXPERIMENT  - Hydra experiment name
#    OUTPUT_DIR      - root output (seed_N/ written underneath)
#    SEEDS           - space-separated seed list
#    MODEL_TAG       - tag written into strategies_summary.json
#    SCORE_CLASSES   - optional, e.g. "0" for QCD-only threshold
#    STRATEGIES      - optional space-separated subset (default: all)
# =====================================================

set -euo pipefail

echo "[`date`] Starting raw AE (new_exp) on $(hostname)"
echo "[`date`] Running as $(whoami)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

RAW_EXPERIMENT="${RAW_EXPERIMENT:?RAW_EXPERIMENT must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
SEEDS="${SEEDS:-7 42 137 1337 31337}"
MODEL_TAG="${MODEL_TAG:-raw_ae}"
SCORE_CLASSES="${SCORE_CLASSES:-}"
STRATEGIES="${STRATEGIES:-}"

echo "[wrapper] Raw experiment : ${RAW_EXPERIMENT}"
echo "[wrapper] Output dir     : ${OUTPUT_DIR}"
echo "[wrapper] Seeds          : ${SEEDS}"
echo "[wrapper] Model tag      : ${MODEL_TAG}"
echo "[wrapper] Score classes  : ${SCORE_CLASSES:-<default>}"
echo "[wrapper] Strategies     : ${STRATEGIES:-<all>}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

for SEED in ${SEEDS}; do
    SEED_OUT="${OUTPUT_DIR}/seed_${SEED}"

    if [ -f "${SEED_OUT}/strategies_summary.json" ]; then
        # Resume-friendly: skip only if raw_results already present
        if python3 -c "import json; d=json.load(open('${SEED_OUT}/strategies_summary.json')); raise SystemExit(0 if d.get('raw_results') else 1)" 2>/dev/null; then
            echo "[`date`] Seed ${SEED}: already done, skipping"
            continue
        fi
    fi

    echo ""
    echo "[`date`] ===== Seed ${SEED} ====="
    mkdir -p "${SEED_OUT}"

    EXTRA=""
    if [ -n "${SCORE_CLASSES}" ]; then
        EXTRA="${EXTRA} --score-classes ${SCORE_CLASSES}"
    fi
    if [ -n "${STRATEGIES}" ]; then
        EXTRA="${EXTRA} --strategies ${STRATEGIES}"
    fi

    PYTHON_CMD="python scripts/run_anomaly_strategies.py \
        --phase2-experiment ${RAW_EXPERIMENT} \
        --skip-embedding \
        --raw-experiment ${RAW_EXPERIMENT} \
        --signal-classes 12 13 14 \
        --seed ${SEED} \
        --output-dir ${SEED_OUT} \
        --model-tag ${MODEL_TAG}${EXTRA}"

    echo "[wrapper] Running: ${PYTHON_CMD}"

    apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        ${PYTHON_CMD}
    "

    echo "[`date`] Seed ${SEED} done"
done

echo ""
echo "[`date`] All seeds finished. Results in: ${OUTPUT_DIR}"
exit 0
