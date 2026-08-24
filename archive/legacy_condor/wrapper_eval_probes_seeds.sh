#!/bin/bash
# =====================================================
#  Generic wrapper: run eval_probes.py on N seeds
#  for a given encoder experiment.
#
#  Environment variables:
#    EXPERIMENT      - Hydra experiment name
#    ENCODER_DIR     - Root dir with seed_N/checkpoints/
#    OUTPUT_DIR      - Root output dir (seed_N/ created here)
#    N_SEEDS         - Number of seeds (default: 10)
#    SEED_START      - First seed index (default: 0)
# =====================================================

set -uo pipefail

echo "[`date`] Starting eval_probes seeds job on $(hostname)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

EXPERIMENT="${EXPERIMENT}"
ENCODER_DIR="${ENCODER_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}"
N_SEEDS="${N_SEEDS:-10}"
SEED_START="${SEED_START:-0}"

echo "[wrapper] Experiment  : ${EXPERIMENT}"
echo "[wrapper] Encoder dir : ${ENCODER_DIR}"
echo "[wrapper] Output dir  : ${OUTPUT_DIR}"
echo "[wrapper] Seeds       : ${SEED_START}...$((SEED_START + N_SEEDS - 1))"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

for SEED in $(seq ${SEED_START} $((SEED_START + N_SEEDS - 1))); do
    SEED_OUT="${OUTPUT_DIR}/seed_${SEED}"
    CKPT=$(find "${ENCODER_DIR}/seed_${SEED}/checkpoints" -name "epoch_*.ckpt" 2>/dev/null | sort | tail -1)

    if [ -z "${CKPT}" ]; then
        echo "[wrapper] WARNING: no checkpoint found for seed ${SEED}, skipping."
        continue
    fi

    # Skip if already done (check both path layouts)
    if [ -f "${SEED_OUT}/probe_evaluation/probe_results.json" ] || [ -f "${SEED_OUT}/probe_results.json" ]; then
        echo "[wrapper] Seed ${SEED} already done, skipping."
        continue
    fi

    mkdir -p "${SEED_OUT}"
    echo ""
    echo "[`date`] ===== Seed ${SEED} ====="
    echo "[wrapper] Checkpoint: ${CKPT}"
    echo "[wrapper] Output    : ${SEED_OUT}"

    if apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        python src/eval_probes.py \
            experiment=${EXPERIMENT} \
            ckpt_path=${CKPT} \
            seed=${SEED} \
            paths.output_dir=${SEED_OUT}
    "; then
        echo "[`date`] Seed ${SEED} done"
    else
        echo "[wrapper] WARNING: Seed ${SEED} FAILED — skipping and continuing to next seed"
    fi
done

# Aggregate results across seeds
echo ""
echo "[`date`] Aggregating results..."
apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    python scripts/aggregate_probe_results.py \
        --results-dir ${OUTPUT_DIR} \
        --output ${OUTPUT_DIR}/aggregated_summary.json
"

echo "[`date`] All done. Results in: ${OUTPUT_DIR}"
exit 0
