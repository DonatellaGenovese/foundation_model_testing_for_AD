#!/bin/bash
# =====================================================
#  HTCondor wrapper for vectorization jobs
# =====================================================

set -euo pipefail

MANIFEST_PATH="$1"
PROJECT_DIR="$2"
EXPERIMENT="${3:-fm_testing_18class_highlevel}"

IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "[`date`] Starting vectorization job on $(hostname)"
echo "[`date`] Running as $(whoami)"
echo "Manifest path: ${MANIFEST_PATH}"

cd ${PROJECT_DIR}

apptainer exec --cleanenv \
    --bind /eos:/eos \
    --writable-tmpfs \
    ${IMAGE} bash -lc "python src/data/vectorize_job.py --manifest-path ${MANIFEST_PATH} experiment=${EXPERIMENT}"

EXIT_CODE=$?
echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
