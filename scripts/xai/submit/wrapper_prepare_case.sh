#!/bin/bash
# HTCondor wrapper — build the CASE held-out-signal dataset (vectorise + apply the
# SM-only normalisation), then draw the 20,000 events per signal the paper scores.
# See scripts/prepare_case_smnorm.py for why the split manifest is built by hand
# rather than by make_split_manifest.
#
# Runs on a batch node rather than interactively because it reads roughly 120 GB of
# parquet: 21 QCD files, and the 15 signal files once for each split.
#
#   EXPERIMENT  experiment config   (default: the script's, new_exp/anomaly_case_v2_smnorm)
#   QCD_FILES   QCD files per split (default: the script's, 7)
#   SELECT      1 draws the selections of scripts/select_case_events.py (default 1;
#               0 for the first production, which is scored in full)
set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "[$(date)] preparing CASE signals — host $(hostname)"

cd ${PROJECT_DIR}

apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR}
  python3 -u scripts/prepare_case_smnorm.py ${EXPERIMENT:+--experiment ${EXPERIMENT}} ${QCD_FILES:+--qcd-files-per-split ${QCD_FILES}}
  if [ \"${SELECT:-1}\" = 1 ]; then
    for s in hToAA_4b_ma60 hToAA_4tau_ma15 HVdilep_Zp1000_piD2_mumu; do
      python3 -u scripts/select_case_events.py --case-label \$s
    done
  fi
"

echo "[$(date)] done"
