#!/bin/bash
# HTCondor wrapper — K selection from the physics profiles alone (no BIC, no ARI).
#   DMODEL   embedding dimension            (default 256)
#   SEED     encoder seed                   (default 3)
#   KVALS    K grid                         (default "3 4 5 6 7 8 9 10 11 12")
#   PCADIM   0 = no projection, else dims   (default 0)
#   NPERM    permutations per pair          (default 200)
#
# Reuses the GMM pickles that select_k_interpretable.py already wrote for K=5..12
# in the matching space, so only K=3 and K=4 are fitted here. GMM_DIR must therefore
# point at the run for the SAME PCADIM, or the reused mixtures would live in a
# different basis than the data they are applied to.
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

DMODEL="${DMODEL:-256}"
SEED="${SEED:-3}"
KVALS="${KVALS:-3 4 5 6 7 8 9 10 11 12}"
PCADIM="${PCADIM:-0}"
NPERM="${NPERM:-200}"

NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
RUN=vcreg_12class_nosparse_dmodel${DMODEL}_cern
EMB=${NE}/xai_embeddings_smnorm/${RUN}/encoder_seed_${SEED}/embeddings
PAPER=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d${DMODEL}_seed${SEED}_smnorm
KSEL=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/k_selection_v3

if [ "${PCADIM}" = "0" ]; then
    TAG="raw"
    GMM_DIR=${KSEL}/vcreg_d${DMODEL}_seed${SEED}_diag
else
    TAG="pca${PCADIM}"
    GMM_DIR=${KSEL}/vcreg_d${DMODEL}_seed${SEED}_diag_pca${PCADIM}
fi
OUT=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/k_profiles/vcreg_d${DMODEL}_seed${SEED}_${TAG}

echo "[$(date)] K selection from profiles — d=${DMODEL}, seed ${SEED}, host $(hostname)"
echo "  K grid  : ${KVALS}"
echo "  pca dim : ${PCADIM}   permutations: ${NPERM}"
echo "  gmm dir : ${GMM_DIR}"
echo "  output  : ${OUT}"

for f in "${EMB}/train_embeddings.npz" "${PAPER}/04_profile/matched_sm_hh4b.npz"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}"; exit 1; }
done
[ -d "${GMM_DIR}" ] || { echo "MISSING gmm dir: ${GMM_DIR}"; exit 1; }

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python -u scripts/xai/select_k_profiles.py \
    --embeddings-dir ${EMB} \
    --matched-npz ${PAPER}/04_profile/matched_sm_hh4b.npz \
    --gmm-dir ${GMM_DIR} \
    --output-dir ${OUT} \
    --k-values ${KVALS} \
    --pca-dim ${PCADIM} \
    --n-perm ${NPERM} \
    --seed ${SEED}
"

echo "[$(date)] done"
