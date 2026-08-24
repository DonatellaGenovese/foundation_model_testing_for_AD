#!/bin/bash
# =====================================================
#  HTCondor wrapper — full XAI pipeline, steps 01 -> 06
#
#  Environment (set in the .sub file):
#    SEED        encoder seed to explain            (default 3)
#    DMODEL      embedding dimension                (default 256)
#    FPR         AE operating point                 (default 0.10)
#    MIN_FRAC    min component occupancy for F2     (default 0.05)
#    OUT_ROOT    output root                        (default derived from SEED)
#
#  Embeddings come from new_exp/xai_embeddings_smnorm/ (all 12 SM classes + 3
#  signals). The embeddings shipped with ad_results/ are NOT usable here: they
#  are filtered to normal|anomaly classes, i.e. QCD + Higgs only, so fitting the
#  GMM on them decomposes QCD rather than the SM.
#
#  Normalisation: everything below is on the smnorm dataset, whose statistics
#  are fitted on the 12 SM classes only. That matches the AD run under
#  ad_results/ (rerun 2026-08-03/04 with anomaly_embedding_vcreg_smnorm), so the
#  AE checkpoint is reused as-is: it is a function on embedding vectors, the
#  encoder is unchanged, and its val-calibrated threshold still applies.
#  Do not mix these paths with the older allsm ones — there the normalisation
#  statistics were fitted on 12 SM + 3 signals.
#
#  K is not hardcoded: step 01 selects it (BIC elbow + ARI) and writes it to
#  k_selection.json; the wrapper reads it back and pins it for steps 02-06 so
#  every step refers to the same mixture.
# =====================================================

set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

SEED="${SEED:-3}"
DMODEL="${DMODEL:-256}"
FPR="${FPR:-0.10}"
MIN_FRAC="${MIN_FRAC:-0.05}"

NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
RUN=vcreg_12class_nosparse_dmodel${DMODEL}_cern
AD=${NE}/ad_results/${RUN}/encoder_seed_${SEED}

EMBEDDINGS_DIR=${NE}/xai_embeddings_smnorm/${RUN}/encoder_seed_${SEED}/embeddings
# The AE checkpoint carries the epoch it stopped at, which is not always 49 —
# the d32 runs end at 44, 45 and 48. Resolve it rather than assuming.
AE_CKPT=$(ls -t ${AD}/mse_normal/checkpoints/ae-epoch*.ckpt 2>/dev/null | head -1)
ENCODER_CKPT=$(ls ${NE}/${RUN}/seed_${SEED}/checkpoints/epoch_*.ckpt 2>/dev/null | tail -1)
# Physics variables (step 04) come from the *vectorised* tree, which holds RAW
# kinematics — normalisation lives only in preprocessed/, so reading the allsm
# tree here carries no leakage. It must be the allsm one: `preprocess_smnorm.py`
# read its vectorised input in place from there, so smnorm/preprocessed is
# event-for-event aligned with allsm/vectorized (verified: QCD 23284, HH4b 29974,
# tt_had 29436 in both). `build_matched_arrays` aligns physics to embeddings
# POSITIONALLY, so a tree with different events would silently pair the physics of
# one event with the embedding of another.
# smnorm/vectorized/ is NOT usable: it is a partial artefact of the AD run with
# different event filtering (QCD 23427) and no SM classes at all.
VEC_DIR=/eos/user/d/dgenoves/foundation_model_testing_data/v2_nosparse_higgs_allsm_highlevel/vectorized/test
# Must be passed explicitly: step 04 would otherwise derive it from VEC_DIR and
# land on allsm/preprocessed, i.e. the normalisation fitted on 12 SM + 3 signals.
PREPROC_DIR=/eos/user/d/dgenoves/foundation_model_testing_data/v2_nosparse_higgs_smnorm_highlevel/preprocessed/test
OUT_ROOT="${OUT_ROOT:-/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d${DMODEL}_seed${SEED}_smnorm}"

echo "[$(date)] XAI pipeline — VCReg d=${DMODEL}, seed ${SEED}"
echo "  host       : $(hostname)"
echo "  embeddings : ${EMBEDDINGS_DIR}"
echo "  AE ckpt    : ${AE_CKPT}"
echo "  encoder    : ${ENCODER_CKPT}"
echo "  output     : ${OUT_ROOT}"

for p in "${EMBEDDINGS_DIR}/train_embeddings.npz" "${EMBEDDINGS_DIR}/test_embeddings.npz" \
         "${AE_CKPT}" "${ENCODER_CKPT}" "${VEC_DIR}"; do
    [ -e "${p}" ] || { echo "MISSING INPUT: ${p}"; exit 1; }
done

[ -e "${PREPROC_DIR}" ] || { echo "MISSING INPUT: ${PREPROC_DIR}"; exit 1; }

# build_matched_arrays skips a class silently when its vectorised dir is absent
# (`if X_raw is None: continue`), so an incomplete tree yields an SM profile built
# on whichever classes happened to be there, with no error. Fail loudly instead.
for D in "${VEC_DIR}" "${PREPROC_DIR}"; do
    N=$(find "${D}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [ "${N}" -lt 15 ]; then
        echo "INCOMPLETE TREE: ${D} has ${N}/15 class dirs."
        find "${D}" -mindepth 1 -maxdepth 1 -type d -printf '    %f\n'
        exit 1
    fi
done

apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}

  if [ -f '${OUT_ROOT}/01_select_k/k_selection.json' ]; then
    echo '[step 01] k_selection.json already present, reusing it'
  else
    echo '[step 01] K selection (BIC elbow + ARI)'
    python scripts/xai/01_select_k.py \
      --embeddings-dir ${EMBEDDINGS_DIR} \
      --output-dir ${OUT_ROOT}/01_select_k \
      --k-values 4 6 8 10 12 \
      --ari-threshold 0.8
  fi

  K=\$(python -c \"import json;print(json.load(open('${OUT_ROOT}/01_select_k/k_selection.json'))['selection']['selected_K'])\")
  echo \"[step 01] selected K = \${K}\"

  echo '[step 02] fit GMM'
  python scripts/xai/02_fit_gmm.py \
    --embeddings-dir ${EMBEDDINGS_DIR} \
    --output-dir ${OUT_ROOT}/02_gmm \
    --k \${K}

  echo '[step 03] AE-flag + assign (F1)'
  python scripts/xai/03_assign_flagged.py \
    --embeddings-dir ${EMBEDDINGS_DIR} \
    --gmm-path ${OUT_ROOT}/02_gmm/gmm_K\${K}.pkl \
    --ae-checkpoint ${AE_CKPT} \
    --output-dir ${OUT_ROOT}/03_assign \
    --fpr ${FPR}

  echo '[step 04] profile + Wasserstein rank (F2, T1)'
  python scripts/xai/04_profile_and_rank.py \
    --gmm-path ${OUT_ROOT}/02_gmm/gmm_K\${K}.pkl \
    --ae-checkpoint ${AE_CKPT} \
    --ckpt-path ${ENCODER_CKPT} \
    --vectorized-dir ${VEC_DIR} \
    --preproc-split-dir ${PREPROC_DIR} \
    --output-dir ${OUT_ROOT}/04_profile \
    --save-matched ${OUT_ROOT}/04_profile/matched_sm_hh4b.npz \
    --min-frac ${MIN_FRAC} \
    --fpr ${FPR}

  echo '[step 05] robustness at K+-2'
  python scripts/xai/05_robustness_kpm2.py \
    --embeddings-dir ${EMBEDDINGS_DIR} \
    --matched-npz ${OUT_ROOT}/04_profile/matched_sm_hh4b.npz \
    --ae-checkpoint ${AE_CKPT} \
    --k \${K} \
    --gmm-dir ${OUT_ROOT}/02_gmm \
    --output-dir ${OUT_ROOT}/05_robustness \
    --fpr ${FPR} \
    --min-frac ${MIN_FRAC}

  echo '[step 06] AE mechanism vs GMM geometry (convergence)'
  python scripts/xai/06_ae_mechanism.py \
    --matched-npz ${OUT_ROOT}/04_profile/matched_sm_hh4b.npz \
    --gmm-path ${OUT_ROOT}/02_gmm/gmm_K\${K}.pkl \
    --ae-checkpoint ${AE_CKPT} \
    --profile-meta ${OUT_ROOT}/04_profile/profile_meta.json \
    --output-dir ${OUT_ROOT}/06_ae_mechanism
"

echo "[$(date)] XAI pipeline finished"
