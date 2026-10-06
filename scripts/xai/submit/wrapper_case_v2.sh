#!/bin/bash
# HTCondor wrapper — anomaly detection on the CASE signals of the collide_v2 production.
#
# Inference only, on the tree with 50,000 events per signal: nothing is trained and no
# threshold is recalibrated. `--dataset case_v2` scores the 20,000 events per signal that
# scripts/select_case_events.py drew (the statistics of the proxy signals), against the
# QCD reference HH->4b was scored against: the same 20,000 test events, embedded by the
# same encoder. Embeddings are extracted once per model and seed, over the whole test
# split; the selection is applied to the embeddings.
#
#   MODEL   vcreg | supcon | simclr | vicreg   (default vcreg)
#   DMODEL  embedding dimension                (default 256)
#   SEEDS   override the seed list             (default: the five seeds of each model)
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
DATA=/eos/user/d/dgenoves/foundation_model_testing_data/v3_nosparse_case_smnorm_highlevel
SEL=${DATA}/selections
MODEL="${MODEL:-vcreg}"
DMODEL="${DMODEL:-256}"
if [ -z "${SEEDS:-}" ]; then
  case "${MODEL}" in
    vcreg)          SEEDS="0 1 2 3 4" ;;
    supcon|simclr)  SEEDS="7 42 137 1337 31337" ;;
    vicreg)         SEEDS="7 42 12345 1337 31337" ;;
    *) echo "unknown MODEL=${MODEL}"; exit 1 ;;
  esac
fi
RUN=$( [ "${MODEL}" = vcreg ] && echo "vcreg_12class_nosparse_dmodel${DMODEL}_cern" \
       || echo "${MODEL}_12class_nosparse_dmodel${DMODEL}_cern" )

echo "[$(date)] CASE collide_v2 — ${MODEL}, d=${DMODEL}, seeds: ${SEEDS}"
echo "  host: $(hostname)"
n_prep=$(ls -d ${DATA}/preprocessed/test/*/ 2>/dev/null | wc -l)
[ "${n_prep}" -ge 4 ] || { echo "MISSING: ${n_prep}/4 classes under ${DATA}/preprocessed/test"; exit 1; }
for s in hToAA_4b_ma60 hToAA_4tau_ma15 HVdilep_Zp1000_piD2_mumu; do
  [ -f "${SEL}/${s}_random.npz" ] || { echo "MISSING selection ${SEL}/${s}_random.npz"; exit 1; }
done
echo "  dataset and selections present"

cd ${PROJECT_DIR}
apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -uo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  rc=0
  for S in ${SEEDS}; do
    echo \"===== ${MODEL} d${DMODEL} seed \${S} =====\"
    python3 -u scripts/infer_new_signals.py --dataset case_v2 --model ${MODEL} \
        --dmodel ${DMODEL} --seed \${S} || { echo \"seed \${S} FAILED\"; rc=1; }
  done
  exit \${rc}
"
echo "[$(date)] done"
