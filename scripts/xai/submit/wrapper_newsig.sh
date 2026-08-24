#!/bin/bash
# =====================================================
#  HTCondor wrapper — additional signal proxies, inference only.
#
#  One job per (model, d_model); the five encoder seeds are looped inside, so the
#  GPU allocation is amortised over them and the dataset is opened once.
#
#  Stage 1 (vectorise + apply the 12-class SM statistics) runs only if the dataset
#  is not already there. It is shared across every job, so submit one job first
#  and let it build the dataset before fanning out — otherwise concurrent jobs
#  race on the same tree.
#
#  Environment:
#    MODEL    vcreg | supcon | simclr   (default vcreg)
#    DMODEL   embedding dimension       (default 256)
#    SEEDS    override the seed list    (default: per-model, see below)
# =====================================================

set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

MODEL="${MODEL:-vcreg}"
DMODEL="${DMODEL:-256}"

# VCReg runs are indexed 0-4; the contrastive runs carry the seed value itself.
if [ -z "${SEEDS:-}" ]; then
  case "${MODEL}" in
    vcreg)          SEEDS="0 1 2 3 4" ;;
    supcon|simclr)  SEEDS="7 42 137 1337 31337" ;;
    # vicreg uses the VCReg-aligned set, where 12345 replaces 137.
    vicreg)         SEEDS="7 42 12345 1337 31337" ;;
    *) echo "unknown MODEL=${MODEL}"; exit 1 ;;
  esac
fi

DATA=/eos/user/d/dgenoves/foundation_model_testing_data/v2_nosparse_newsig_smnorm_highlevel

echo "[$(date)] additional signals — ${MODEL}, d=${DMODEL}, seeds: ${SEEDS}"
echo "  host: $(hostname)"

apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}

  n_prep=\$(find ${DATA}/preprocessed/test -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [ \"\${n_prep}\" -eq 6 ]; then
    echo '[stage 1] prepared dataset already present (6/6 classes) — skipping'
  else
    echo \"[stage 1] preparing dataset (found \${n_prep}/6 classes)\"
    python3 scripts/prepare_newsig_smnorm.py
  fi

  rc=0
  for S in ${SEEDS}; do
    echo \"===== ${MODEL} d${DMODEL} seed \${S} =====\"
    python3 scripts/infer_new_signals.py --model ${MODEL} --dmodel ${DMODEL} --seed \${S} || {
      echo \"seed \${S} FAILED\"; rc=1; }
  done
  exit \${rc}
"

echo "[$(date)] done"
