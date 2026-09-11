#!/bin/bash
# HTCondor wrapper — does the physics conclusion survive at K=7, and does it survive
# the projection? Runs step 04 (the Wasserstein ranking) at K=7 twice: on the raw
# 256-dimensional embedding and on its 64-dimensional PCA. The AE scores the full
# embedding in both, so the flag set is identical and the two runs differ only in how
# the flagged events are partitioned — which isolates the effect of the projection.
#
# The test that matters: n_bjets must come out rank 1. Every stability criterion we
# have (occupancy, separation, ARI) is a property of the clustering; this is the only
# one that is a property of the paper's claim.
#
#   PCADIM  0 = raw embedding, else PCA dimensions (default 0)
#   KVAL    number of components (default 7)
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

DMODEL="${DMODEL:-256}"
SEED="${SEED:-3}"
KVAL="${KVAL:-7}"
PCADIM="${PCADIM:-0}"

NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
RUN=vcreg_12class_nosparse_dmodel${DMODEL}_cern
EMB=${NE}/xai_embeddings_smnorm/${RUN}/encoder_seed_${SEED}/embeddings
PAPER=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d${DMODEL}_seed${SEED}_smnorm
KSEL=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/k_selection_v3
AD=${NE}/ad_results/${RUN}/encoder_seed_${SEED}

if [ "${PCADIM}" = "0" ]; then
    TAG="raw"
    GMM=${KSEL}/vcreg_d${DMODEL}_seed${SEED}_diag/gmm_K${KVAL}.pkl
    PCA_ARGS=""
else
    TAG="pca${PCADIM}"
    GMM=${KSEL}/vcreg_d${DMODEL}_seed${SEED}_diag_pca${PCADIM}/gmm_K${KVAL}.pkl
    PCA_ARGS="--pca-dim ${PCADIM} --pca-embeddings-dir ${EMB} --pca-seed ${SEED}"
fi
OUT=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/rank_k${KVAL}/vcreg_d${DMODEL}_seed${SEED}_${TAG}

# Globbed, not hardcoded: the AE checkpoint epoch differs per run (d32 stopped at
# 44/45/48, d256 at 49), and an earlier version of this pipeline pinned epoch 49 and
# silently reported checkpoints as missing elsewhere.
AE_CKPT=$(ls -t ${AD}/mse_normal/checkpoints/ae-epoch*.ckpt 2>/dev/null | head -1)

echo "[$(date)] Wasserstein ranking at K=${KVAL}, space=${TAG}, host $(hostname)"
echo "  gmm     : ${GMM}"
echo "  ae ckpt : ${AE_CKPT}"
echo "  output  : ${OUT}"

for f in "${GMM}" "${AE_CKPT}" "${PAPER}/04_profile/matched_sm_hh4b.npz"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}"; exit 1; }
done

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python -u scripts/xai/04_profile_and_rank.py \
    --matched-npz ${PAPER}/04_profile/matched_sm_hh4b.npz \
    --gmm-path ${GMM} \
    --ae-checkpoint '${AE_CKPT}' \
    --output-dir ${OUT} \
    ${PCA_ARGS}
"

echo "[$(date)] done"
