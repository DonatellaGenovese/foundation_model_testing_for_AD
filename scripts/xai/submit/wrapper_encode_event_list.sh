#!/bin/bash
# HTCondor wrapper — step 1 of the interpretability chain: encode the published event lists.
#   EMB  output directory (default: $EMB of scripts/xai/paths.sh)
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "[$(date)] encode event lists — host $(hostname)"
apptainer exec --nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR} ${EMB:+EMB=${EMB}}
  source scripts/xai/paths.sh
  python3 scripts/xai/encode_event_list.py --lists \$LISTS \
      --data \$FMD/v2_nosparse_higgs_smnorm_highlevel/preprocessed --ckpt \$ENC --output-dir \$EMB
"
echo "[$(date)] done"
