#!/bin/bash
# =====================================================
#  HTCondor wrapper — XAI steps 03 -> 06 at the SELECTED K, in the projected space.
#
#  Differs from wrapper_xai_full.sh in two ways, both deliberate:
#
#  1. K is NOT chosen here. wrapper_xai_full.sh runs step 01 (BIC elbow + ARI) and
#     takes whatever it returns, which was 12. The paper's K comes instead from the
#     occupancy/duplicate criterion in select_k_profiles.py, which gives 7:
#     feasible = [3,4,5,6,7], and K=8 drops a component to 0.128 of the uniform
#     share against a 0.2 floor. So K is pinned from the outside.
#
#  2. The mixture is fitted in a 64-dimensional PCA of the embedding, not in all
#     256. The AE still scores the full embedding at every step, so no anomaly
#     score and no threshold changes — only the partition is projected.
#
#  The K=7 PCA mixture already exists: it was fitted by the K scan. Refitting it
#  here would risk a different local optimum for no gain, so it is reused.
#
#  Environment (set in the .sub):
#    SEED, DMODEL, KVAL, PCADIM (0 = unprojected), SIGNAL, FPR, MIN_FRAC, OUT_ROOT
# =====================================================
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

SEED="${SEED:-3}"; DMODEL="${DMODEL:-256}"; KVAL="${KVAL:-7}"; PCADIM="${PCADIM:-64}"
FPR="${FPR:-0.10}"; MIN_FRAC="${MIN_FRAC:-0.05}"
SIGNAL="${SIGNAL:-hh4b}"

NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
XP=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper
RUN=vcreg_12class_nosparse_dmodel${DMODEL}_cern
AD=${NE}/ad_results/${RUN}/encoder_seed_${SEED}
EMB=${NE}/xai_embeddings_smnorm/${RUN}/encoder_seed_${SEED}/embeddings

AE_CKPT=$(ls -t ${AD}/mse_normal/checkpoints/ae-epoch*.ckpt 2>/dev/null | head -1)
ENCODER_CKPT=$(ls ${NE}/${RUN}/seed_${SEED}/checkpoints/epoch_*.ckpt 2>/dev/null | tail -1)

if [ "${PCADIM}" = "0" ]; then
    TAG=raw
    GMM=${XP}/k_selection_v3/vcreg_d${DMODEL}_seed${SEED}_diag/gmm_K${KVAL}.pkl
    PCA_ARGS=""
else
    TAG=pca${PCADIM}
    GMM=${XP}/k_selection_v3/vcreg_d${DMODEL}_seed${SEED}_diag_pca${PCADIM}/gmm_K${KVAL}.pkl
    PCA_ARGS="--pca-dim ${PCADIM} --pca-embeddings-dir ${EMB} --pca-seed ${SEED}"
fi

# Directory holding ${GMM}: the K-selection output, which already contains gmm_K5/K7/K9
# fitted in this same space by the same protocol. Step 05 reads its K grid from here.
GMM_DIR=$(dirname "${GMM}")

# Step 04 already ran at K=7 in both spaces (rank_k7/), and it is what wrote the
# matched npz and profile_meta.json that steps 05 and 06 consume. Reuse them rather
# than recomputing: a second run would rebuild the matched array and there would be
# no guarantee it paired the same events.
# Per-signal inputs. Step 04 already ran at K=7 for both, and it is what wrote the
# matched npz and profile_meta.json that steps 05 and 06 consume, so those are reused
# rather than recomputed: a second step-04 run would rebuild the matched array with no
# guarantee it paired the same events.
#
# THE TWO SIGNALS DO NOT SHARE AN EMBEDDING TREE, and only one of them can be read
# from a split tree at all. The CASE tree holds QCD plus the seven CASE processes and
# no Standard-Model classes, so step 03 reads the matched array there instead — that
# array carries the 12 SM classes and the signal together, which is the population the
# flag rates describe. Step 05 still points at the SM tree, because all it takes from
# it is the SM *train* split used to refit the K+-2 mixtures, and the SM background is
# common to both signals by construction (build_matched_case.py reuses it verbatim).
case "${SIGNAL}" in
  hh4b)
    SIGLABEL=13
    RANK=${XP}/rank_k7/vcreg_d${DMODEL}_seed${SEED}_${TAG}
    MATCHED=${XP}/vcreg_d${DMODEL}_seed${SEED}_smnorm/04_profile/matched_sm_hh4b.npz
    STEP03_SRC="--embeddings-dir ${EMB}"
    ;;
  hvdilep)
    SIGLABEL=20
    CASEDIR=${XP}/case_HVdilep_Zp1000_piD2_mumu_d${DMODEL}_seed${SEED}
    RANK=${CASEDIR}/04_profile
    MATCHED=${CASEDIR}/matched_sm_HVdilep_Zp1000_piD2_mumu.npz
    STEP03_SRC="--matched-npz ${MATCHED}"
    ;;
  *) echo "unknown SIGNAL=${SIGNAL}"; exit 1 ;;
esac
OUT_ROOT="${OUT_ROOT:-${XP}/k7_${SIGNAL}_${TAG}_d${DMODEL}_seed${SEED}}"

echo "[$(date)] XAI at K=${KVAL}, space=${TAG}, signal=${SIGNAL} (label ${SIGLABEL}) — VCReg d=${DMODEL} seed ${SEED}"
echo "  GMM      : ${GMM}"
echo "  AE ckpt  : ${AE_CKPT}"
echo "  matched  : ${MATCHED}"
echo "  step04   : ${RANK}"
echo "  output   : ${OUT_ROOT}"

for p in "${GMM}" "${AE_CKPT}" "${ENCODER_CKPT}" "${MATCHED}" \
         "${RANK}/profile_meta.json" "${EMB}/test_embeddings.npz"; do
    [ -e "${p}" ] || { echo "MISSING INPUT: ${p}"; exit 1; }
done

apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}

  echo '[step 03] AE-flag + assign'
  python -u scripts/xai/03_assign_flagged.py \
    ${STEP03_SRC} \
    --signal-label ${SIGLABEL} \
    --gmm-path ${GMM} \
    --ae-checkpoint ${AE_CKPT} \
    --output-dir ${OUT_ROOT}/03_assign \
    --fpr ${FPR} ${PCA_ARGS}

  echo '[step 05] robustness at K+-2'
  # --gmm-dir points at the K-selection directory, the one ${GMM} itself comes from, so the
  # K grid reuses the very mixtures the rest of the analysis runs on. It must NOT point at
  # a fresh directory: step 05 would then refit every K, and a refit landing on a poorer
  # optimum shifts the signal between components, which reads as an instability of the
  # partition rather than as a weaker fit. (Not 02_gmm/ either -- that holds 256-dimensional
  # K=12 mixtures from a different space, which check_gmm_dims rejects.)
  python -u scripts/xai/05_robustness_kpm2.py \
    --embeddings-dir ${EMB} \
    --matched-npz ${MATCHED} \
    --ae-checkpoint ${AE_CKPT} \
    --k ${KVAL} \
    --gmm-dir ${GMM_DIR} \
    --output-dir ${OUT_ROOT}/05_robustness \
    --signal-label ${SIGLABEL} \
    --fpr ${FPR} --min-frac ${MIN_FRAC} ${PCA_ARGS}

  echo '[step 06] AE mechanism vs GMM geometry'
  python -u scripts/xai/06_ae_mechanism.py \
    --matched-npz ${MATCHED} \
    --gmm-path ${GMM} \
    --ae-checkpoint ${AE_CKPT} \
    --profile-meta ${RANK}/profile_meta.json \
    --output-dir ${OUT_ROOT}/06_ae_mechanism ${PCA_ARGS}
"
echo "[$(date)] done"
