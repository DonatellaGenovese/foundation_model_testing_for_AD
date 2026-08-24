#!/bin/bash
# HTCondor wrapper — interpretability-oriented K selection (stability + physical
# distinctiveness at production settings). CPU only.
#   DMODEL  embedding dimension (default 256)
#   SEED    encoder seed        (default 3)
#   KVALS   K grid              (default "4 6 8 10 12 14 16")
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

DMODEL="${DMODEL:-256}"
SEED="${SEED:-3}"
KVALS="${KVALS:-5 6 7 8 9 10 11 12}"
NRESTARTS="${NRESTARTS:-12}"
COVTYPE="${COVTYPE:-diag}"
PCAVAR="${PCAVAR:-0}"

NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
RUN=vcreg_12class_nosparse_dmodel${DMODEL}_cern
EMB=${NE}/xai_embeddings_smnorm/${RUN}/encoder_seed_${SEED}/embeddings
PAPER=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d${DMODEL}_seed${SEED}_smnorm
TAG="${COVTYPE}"
if [ "${PCAVAR}" != "0" ]; then TAG="${TAG}_pca${PCAVAR}"; fi
OUT=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/k_selection_v3/vcreg_d${DMODEL}_seed${SEED}_${TAG}

echo "[$(date)] K selection v2 — d=${DMODEL}, seed ${SEED}, host $(hostname)"
echo "  K grid : ${KVALS}"
echo "  restarts: ${NRESTARTS}   covariance: ${COVTYPE}   pca_var: ${PCAVAR}"
echo "  output : ${OUT}"

for f in "${EMB}/train_embeddings.npz" "${PAPER}/04_profile/matched_sm_hh4b.npz"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}"; exit 1; }
done

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python scripts/xai/select_k_interpretable.py \
    --embeddings-dir ${EMB} \
    --matched-npz ${PAPER}/04_profile/matched_sm_hh4b.npz \
    --output-dir ${OUT} \
    --k-values ${KVALS} \
    --n-init 5 --n-restarts ${NRESTARTS} \
    --cov-type ${COVTYPE} \
    --pca-var ${PCAVAR}
"

echo "[$(date)] done"
