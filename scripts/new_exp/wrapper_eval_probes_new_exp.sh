#!/bin/bash
# =====================================================
#  Wrapper: run eval_probes.py on an explicit seed list
#  for a given new_exp encoder experiment.
#
#  Environment variables:
#    EXPERIMENT  - Hydra experiment name (e.g. new_exp/simclr_dmodel128)
#    ENCODER_DIR - Root dir with seed_<N>/checkpoints/
#    OUTPUT_DIR  - Root output dir (seed_<N>/ created here)
#    SEEDS       - space- or comma-separated seed list
#                  (default: 7 42 137 1337 31337)
# =====================================================

set -uo pipefail

echo "[`date`] Starting eval_probes (new_exp) on $(hostname)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

EXPERIMENT="${EXPERIMENT}"
ENCODER_DIR="${ENCODER_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}"
SEEDS="${SEEDS:-7 42 137 1337 31337}"
SEEDS="${SEEDS//,/ }"   # normalise comma-separated to space-separated

echo "[wrapper] Experiment  : ${EXPERIMENT}"
echo "[wrapper] Encoder dir : ${ENCODER_DIR}"
echo "[wrapper] Output dir  : ${OUTPUT_DIR}"
echo "[wrapper] Seeds       : ${SEEDS}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

for SEED in ${SEEDS}; do
    SEED_OUT="${OUTPUT_DIR}/seed_${SEED}"
    CKPT=$(find "${ENCODER_DIR}/seed_${SEED}/checkpoints" -name "epoch_*.ckpt" 2>/dev/null | sort | tail -1)

    if [ -z "${CKPT}" ]; then
        echo "[wrapper] WARNING: no checkpoint found for seed ${SEED}, skipping."
        continue
    fi

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
            eval.linear_probe.max_epochs=50 \
            paths.output_dir=${SEED_OUT}
    "; then
        echo "[`date`] Seed ${SEED} done"
    else
        echo "[wrapper] WARNING: Seed ${SEED} FAILED — continuing to next seed"
    fi
done

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
