#!/bin/bash
# =====================================================
#  Determinism test: run eval_probes.py twice on the
#  same checkpoint and verify results are identical.
#
#  RUN_ID must be set (1 or 2) via environment variable.
# =====================================================

set -euo pipefail

echo "[`date`] Starting probe determinism test (run ${RUN_ID}) on $(hostname)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

EXPERIMENT="vcreg_12class_nosparse_dmodel128_cern"
CKPT="/eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/vcreg_12class_nosparse_dmodel128_cern/seed_0/checkpoints/epoch_045.ckpt"
OUTPUT_DIR="/eos/user/d/dgenoves/anomaly_pipeline/probe_determinism_test/run${RUN_ID}"
SEED=0

cd ${PROJECT_DIR}

apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR} \
    "${IMAGE}" bash -lc "
        cd ${PROJECT_DIR}
        export PROJECT_ROOT=${PROJECT_DIR}
        python src/eval_probes.py \
            experiment=${EXPERIMENT} \
            ckpt_path=${CKPT} \
            seed=${SEED} \
            paths.output_dir=${OUTPUT_DIR}
    "

echo "[`date`] Done. Results in: ${OUTPUT_DIR}"
exit 0
