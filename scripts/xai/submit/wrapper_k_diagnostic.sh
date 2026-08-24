#!/bin/bash
# =====================================================
#  HTCondor wrapper — K-selection diagnostic on a WIDE K grid.
#
#  Why this exists. On the production grid [4,6,8,10,12] neither selection
#  criterion fired: BIC decreased monotonically and mean pairwise ARI never
#  reached 0.8, so step 01 fell back to min-BIC — i.e. "the largest K we tried".
#  Two measurements show the grid, not the method, is the problem:
#    - the BIC penalty is ~5-8x too weak to produce a minimum here
#      (71k at K=12/d256 against BIC steps of ~520k);
#    - on d256 the BIC steps are still ACCELERATING at the grid edge
#      (-515k,-449k,-496k,-620k), so the optimum is far to the right.
#      On d128 they decelerate (-274k,-224k,-206k,-201k) — different regimes.
#
#  This runs step 01 alone over a wide grid to establish whether a real elbow
#  exists at all, and where ARI peaks. It writes to a separate output dir so the
#  production 01_select_k/ results are never touched.
#
#  Environment:
#    DMODEL   embedding dimension    (default 256)
#    SEED     encoder seed           (default 3)
#    KVALS    K grid                 (default "4 8 12 16 20 24 32 40")
#    NINITS   restarts for ARI       (default 5)
# =====================================================

set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

DMODEL="${DMODEL:-256}"
SEED="${SEED:-3}"
KVALS="${KVALS:-4 8 12 16 20 24 32 40}"
NINITS="${NINITS:-5}"

NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
RUN=vcreg_12class_nosparse_dmodel${DMODEL}_cern
EMBEDDINGS_DIR=${NE}/xai_embeddings_smnorm/${RUN}/encoder_seed_${SEED}/embeddings
OUT=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/k_diagnostic/vcreg_d${DMODEL}_seed${SEED}_smnorm

echo "[$(date)] K diagnostic — d=${DMODEL}, seed ${SEED}, host $(hostname)"
echo "  embeddings : ${EMBEDDINGS_DIR}"
echo "  K grid     : ${KVALS}"
echo "  output     : ${OUT}"

[ -e "${EMBEDDINGS_DIR}/train_embeddings.npz" ] || {
    echo "MISSING: ${EMBEDDINGS_DIR}/train_embeddings.npz"; exit 1; }

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python scripts/xai/01_select_k.py \
    --embeddings-dir ${EMBEDDINGS_DIR} \
    --output-dir ${OUT} \
    --k-values ${KVALS} \
    --n-ari-inits ${NINITS} \
    --ari-threshold 0.8
"

echo "[$(date)] done"
