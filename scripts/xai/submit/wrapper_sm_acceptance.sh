#!/bin/bash
# HTCondor wrapper — SM acceptance at the QCD 10% AE threshold, VCReg d256, seeds 0-4.
# Needs the full-SM test embeddings of all five seeds (extract_xai_emb.sub for seed 3,
# extract_xai_emb_seeds.sub for the others).
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "[$(date)] SM acceptance — host $(hostname)"
apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python3 scripts/xai/sm_acceptance.py --dmodel 256 --seeds 0 1 2 3 4
"
echo "[$(date)] done"
