#!/bin/bash
# HTCondor wrapper — XAI paper steps 02 → 04 (edit paths / K as needed)
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

EMBEDDINGS_DIR=/eos/user/d/dgenoves/anomaly_pipeline/gmm_encoder_seeds/vcreg_nosparse_dmodel128/encoder_seed_0/embeddings
AE_CKPT=/eos/user/d/dgenoves/anomaly_pipeline/ad_encoder_seeds/vcreg_nosparse_dmodel128/encoder_seed_0/mse_normal/checkpoints/ae-epochepoch=49.ckpt
ENCODER_CKPT=/eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/vcreg_12class_nosparse_dmodel128_cern/seed_0/checkpoints/epoch_045.ckpt
VEC_DIR=/eos/user/d/dgenoves/foundation_model_testing_data/v2_nosparse_higgs_allsm_highlevel/vectorized/test
OUT_ROOT=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d128_seed0
K=12

apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}

  python scripts/xai/02_fit_gmm.py \
    --embeddings-dir ${EMBEDDINGS_DIR} \
    --output-dir ${OUT_ROOT}/02_gmm \
    --k ${K}

  python scripts/xai/03_assign_flagged.py \
    --embeddings-dir ${EMBEDDINGS_DIR} \
    --gmm-path ${OUT_ROOT}/02_gmm/gmm_K${K}.pkl \
    --ae-checkpoint ${AE_CKPT} \
    --output-dir ${OUT_ROOT}/03_assign \
    --fpr 0.10

  python scripts/xai/04_profile_and_rank.py \
    --gmm-path ${OUT_ROOT}/02_gmm/gmm_K${K}.pkl \
    --ae-checkpoint ${AE_CKPT} \
    --ckpt-path ${ENCODER_CKPT} \
    --vectorized-dir ${VEC_DIR} \
    --output-dir ${OUT_ROOT}/04_profile \
    --save-matched ${OUT_ROOT}/04_profile/matched_sm_hh4b.npz \
    --min-frac 0.05 \
    --fpr 0.10
"
