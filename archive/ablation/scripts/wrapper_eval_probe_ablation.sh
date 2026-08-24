#!/bin/bash
# =====================================================
#  HTCondor wrapper for ablation probe evaluation
#  Environment vars (set in .sub file):
#    EXPERIMENT   — Hydra experiment (e.g. ablation/aug_supcon_random_feature)
#    CKPT_PATH    — full EOS path to epoch_*.ckpt
#    SEED         — integer seed
#    OUTPUT_DIR   — directory where probe_evaluation/ will be written
# =====================================================

set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "[$(date)] Starting ablation probe eval"
echo "[$(date)] Host: $(hostname) | User: $(whoami)"
echo "[$(date)] Experiment: ${EXPERIMENT} | Seed: ${SEED}"
echo "[$(date)] Checkpoint: ${CKPT_PATH}"
echo "[$(date)] Output: ${OUTPUT_DIR}"

cd ${PROJECT_DIR}

APPTAINER_FLAGS="--nv \
    --bind /afs:/afs \
    --bind /eos:/eos \
    --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    python src/eval_probes.py \
        experiment=${EXPERIMENT} \
        ckpt_path=${CKPT_PATH} \
        seed=${SEED} \
        paths.output_dir=${OUTPUT_DIR} \
        eval.save_visualizations=false
"

EXIT_CODE=$?
echo "[$(date)] Probe eval finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
