#!/bin/bash
# HTCondor wrapper — anomaly detection on the CASE held-out signals.
#
# Inference only: encoder and AE already exist and the operating threshold is the
# val-calibrated value stored in the AE checkpoint, so scoring a new process is a
# forward pass against a threshold fixed before the process was ever seen.
#
# Unlike wrapper_newsig.sh there is no dataset-preparation stage: the CASE tree is
# built once by scripts/prepare_case_smnorm.py, which needs a hand-built split
# manifest and so does not belong in a per-model loop.
#
#   MODEL   vcreg | supcon | simclr | vicreg   (default vcreg)
#   DMODEL  embedding dimension                (default 256)
#   SEEDS   override the seed list             (default: per-model, see below)
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
DATA=/eos/user/d/dgenoves/foundation_model_testing_data/v2_nosparse_case_smnorm_highlevel

MODEL="${MODEL:-vcreg}"
DMODEL="${DMODEL:-256}"

# VCReg runs are indexed 0-4; the contrastive runs carry the seed value itself, and
# vicreg uses the VCReg-aligned set where 12345 replaces 137.
if [ -z "${SEEDS:-}" ]; then
  case "${MODEL}" in
    vcreg)          SEEDS="0 1 2 3 4" ;;
    supcon|simclr)  SEEDS="7 42 137 1337 31337" ;;
    vicreg)         SEEDS="7 42 12345 1337 31337" ;;
    *) echo "unknown MODEL=${MODEL}"; exit 1 ;;
  esac
fi

echo "[$(date)] CASE signals — ${MODEL}, d=${DMODEL}, seeds: ${SEEDS}"
echo "  host: $(hostname)"

n_prep=$(ls -d ${DATA}/preprocessed/test/*/ 2>/dev/null | wc -l)
if [ "${n_prep}" -lt 8 ]; then
  echo "MISSING: CASE dataset has ${n_prep}/8 classes under ${DATA}/preprocessed/test"
  echo "Run scripts/prepare_case_smnorm.py first."
  exit 1
fi
echo "  dataset: ${n_prep}/8 classes present"

cd ${PROJECT_DIR}

apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -uo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  rc=0
  for S in ${SEEDS}; do
    echo \"===== ${MODEL} d${DMODEL} seed \${S} =====\"
    python3 -u scripts/infer_new_signals.py --dataset case --model ${MODEL} \
        --dmodel ${DMODEL} --seed \${S} || { echo \"seed \${S} FAILED\"; rc=1; }
  done
  exit \${rc}
"

echo "[$(date)] done"
