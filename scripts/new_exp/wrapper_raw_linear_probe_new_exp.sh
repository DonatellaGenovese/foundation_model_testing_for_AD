#!/bin/bash
# =====================================================
#  HTCondor wrapper: raw linear probe (lower-bound baseline) — new_exp
#
#  Trains nn.Linear(340, 12) directly on raw features (no encoder),
#  on the FILTERED dataset (v2_12class_nosparse_highlevel/preprocessed),
#  for comparison with the new_exp encoders.
#
#  Environment variables (set in .sub file):
#    SEEDS      - space-separated seed list (default: 7 42 137 1337 31337)
#    OUTPUT_DIR - root output directory
# =====================================================

set -euo pipefail

echo "[`date`] Starting raw linear probe (new_exp) on $(hostname)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
DATA_DIR=/eos/user/d/dgenoves/foundation_model_testing_data/v2_12class_nosparse_highlevel/preprocessed

SEEDS="${SEEDS:-7 42 137 1337 31337}"
OUTPUT_DIR="${OUTPUT_DIR:-/eos/user/d/dgenoves/anomaly_pipeline/new_exp/raw_linear_probe}"

echo "[wrapper] Seeds      : ${SEEDS}"
echo "[wrapper] Output dir : ${OUTPUT_DIR}"
echo "[wrapper] Data dir   : ${DATA_DIR}"

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

cd ${PROJECT_DIR}

for SEED in ${SEEDS}; do
    SEED_OUT="${OUTPUT_DIR}/seed_${SEED}"

    if [ -f "${SEED_OUT}/probe_results.json" ]; then
        echo "[`date`] Seed ${SEED}: already done, skipping"
        continue
    fi

    echo ""
    echo "[`date`] ===== Seed ${SEED} ====="
    mkdir -p "${SEED_OUT}"

    apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        python scripts/eval_raw_linear_probe.py \
            --data-dir ${DATA_DIR} \
            --output-dir ${SEED_OUT} \
            --seed ${SEED} \
            --max-epochs 50 \
            --lr 1e-3 \
            --batch-size 1024
    "
    echo "[`date`] Seed ${SEED} done"
done

echo ""
echo "[`date`] All seeds done — aggregating results..."

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    python scripts/aggregate_probe_results.py \
        --results-dir ${OUTPUT_DIR} \
        --output ${OUTPUT_DIR}/aggregated_summary.json
"

echo "[`date`] Done. Results in: ${OUTPUT_DIR}"
exit 0
