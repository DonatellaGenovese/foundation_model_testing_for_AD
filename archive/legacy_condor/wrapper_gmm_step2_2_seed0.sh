#!/bin/bash
# HTCondor wrapper: GMM Step 2.2 — Physical Profiles, seed_0 encoder

set -euo pipefail
echo "[$(date)] Starting GMM Step 2.2 (seed_0) on $(hostname)"

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

GMM_DIR=/eos/user/d/dgenoves/anomaly_pipeline/gmm_encoder_seeds/vcreg_nosparse_dmodel128/encoder_seed_0
CKPT=/eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/vcreg_12class_nosparse_dmodel128_cern/seed_0/checkpoints/epoch_045.ckpt
VEC_DIR=/eos/user/d/dgenoves/foundation_model_testing_data/v2_12class_nosparse_highlevel/vectorized/test
OUTPUT_DIR=/eos/user/d/dgenoves/anomaly_pipeline/xai_gmm_mocktest/vcreg_d128_seed0_ae_filtered/step2_2_physics

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs"

PYTHON_CMD="python scripts/xai_mocktest/gmm_step2_2_physical_profiles.py \
    --gmm-dir        ${GMM_DIR} \
    --gmm-format     npz \
    --k-values       6 10 12 15 \
    --best-k         12 \
    --ckpt-path      ${CKPT} \
    --vectorized-dir ${VEC_DIR} \
    --output-dir     ${OUTPUT_DIR} \
    --pca-n          0 \
    --max-per-class  20000"

echo "GMM_DIR    : ${GMM_DIR}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    ${PYTHON_CMD}
"
EXIT_CODE=$?
echo "[$(date)] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
