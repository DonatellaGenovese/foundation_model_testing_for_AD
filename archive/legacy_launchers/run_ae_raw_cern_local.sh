#!/bin/bash
# Run AE anomaly detection (QCD vs Higgs, raw features) inside the Apptainer container.
# Usage: bash run_ae_raw_cern_local.sh [extra hydra overrides...]

set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "========================================"
echo "AE Anomaly Detection — raw features"
echo "Project : $PROJECT_DIR"
echo "Image   : $IMAGE"
echo "Started : $(date)"
echo "========================================"

cd "$PROJECT_DIR"

export PROJECT_ROOT="$PROJECT_DIR"

APPTAINER_FLAGS="--bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
  --env PROJECT_ROOT=$PROJECT_ROOT \
  --env PYTHONPATH=$PROJECT_DIR:$PYTHONPATH"

apptainer exec $APPTAINER_FLAGS "$IMAGE" bash -lc "
  cd $PROJECT_DIR
  export PROJECT_ROOT=$PROJECT_DIR
  export PYTHONPATH=$PROJECT_DIR:\$PYTHONPATH
  python src/train_anomaly_detection.py \
    experiment=anomaly_qcd_vs_higgs_raw_cern \
    seed=42 \
    $*
"

EXIT_CODE=$?
echo "========================================"
echo "Finished at $(date) — exit code $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
