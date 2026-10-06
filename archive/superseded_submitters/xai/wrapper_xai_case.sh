#!/bin/bash
# HTCondor wrapper — interpretability on a CASE signal, against the SAME SM background
# the HH->4b analysis uses.
#
# Two stages: build the mixed matched array (SM rows reused verbatim from the HH->4b
# file, signal block built from the CASE trees), then run the Wasserstein ranking.
#
# Only step 04 runs. Step 03 and step 04's own matched-array builder read the
# fifteen-class tree, which does not contain the CASE processes, so they cannot be
# used here; step 04 accepts a prebuilt --matched-npz and that is the path taken.
#
# OUT_ROOT is deliberately distinct from the HH->4b root: that run writes
# matched_sm_hh4b.npz and its profile outputs under its own tree, and pointing this
# job at the same root would overwrite them.
#
#   CASE_LABEL  CASE folder name        (default HVdilep_Zp1000_piD2_mumu)
#   SIGLABEL    label for the signal    (default 20; must stay clear of 0-14)
#   PCADIM      GMM projection          (default 64)
#   KVAL        mixture components      (default 7)
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

DMODEL="${DMODEL:-256}"
SEED="${SEED:-3}"
CASE_LABEL="${CASE_LABEL:-HVdilep_Zp1000_piD2_mumu}"
SIGLABEL="${SIGLABEL:-20}"
PCADIM="${PCADIM:-64}"
KVAL="${KVAL:-7}"

NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
RUN=vcreg_12class_nosparse_dmodel${DMODEL}_cern
EMB=${NE}/xai_embeddings_smnorm/${RUN}/encoder_seed_${SEED}/embeddings
PAPER=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d${DMODEL}_seed${SEED}_smnorm
KSEL=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/k_selection_v3
AD=${NE}/ad_results/${RUN}/encoder_seed_${SEED}

OUT_ROOT=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/case_${CASE_LABEL}_d${DMODEL}_seed${SEED}
MATCHED=${OUT_ROOT}/matched_sm_${CASE_LABEL}.npz
GMM=${KSEL}/vcreg_d${DMODEL}_seed${SEED}_diag_pca${PCADIM}/gmm_K${KVAL}.pkl

ENC_CKPT=$(ls -t ${NE}/${RUN}/seed_${SEED}/checkpoints/epoch_*.ckpt 2>/dev/null | head -1)
AE_CKPT=$(ls -t ${AD}/mse_normal/checkpoints/ae-epoch*.ckpt 2>/dev/null | head -1)

echo "[$(date)] XAI on CASE signal — ${CASE_LABEL}, host $(hostname)"
echo "  encoder : ${ENC_CKPT}"
echo "  AE      : ${AE_CKPT}"
echo "  GMM     : ${GMM}   (PCA ${PCADIM}, K=${KVAL})"
echo "  output  : ${OUT_ROOT}"

for f in "${ENC_CKPT}" "${AE_CKPT}" "${GMM}" "${PAPER}/04_profile/matched_sm_hh4b.npz"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}"; exit 1; }
done

mkdir -p ${OUT_ROOT}

apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}

  if [ -f '${MATCHED}' ]; then
    echo '[stage 1] matched array already present — reusing'
  else
    echo '[stage 1] building the mixed matched array'
    python -u scripts/xai/build_matched_case.py \
      --case-label ${CASE_LABEL} \
      --signal-label ${SIGLABEL} \
      --sm-matched ${PAPER}/04_profile/matched_sm_hh4b.npz \
      --ckpt '${ENC_CKPT}' \
      --output ${MATCHED}
  fi

  echo '[stage 2] Wasserstein ranking'
  python -u scripts/xai/04_profile_and_rank.py \
    --matched-npz ${MATCHED} \
    --gmm-path ${GMM} \
    --ae-checkpoint '${AE_CKPT}' \
    --signal-label ${SIGLABEL} \
    --pca-dim ${PCADIM} --pca-embeddings-dir ${EMB} --pca-seed ${SEED} \
    --output-dir ${OUT_ROOT}/04_profile
"

echo "[$(date)] done"
