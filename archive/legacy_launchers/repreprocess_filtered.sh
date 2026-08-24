#!/bin/bash
# Rerun preprocessing on the filtered vectorized data.
#
# Must be run AFTER filter_empty_vectorized.py has cleaned the vectorized .npy files.
#
# Steps:
#   1. Rename old preprocessed dirs to *_old (fast EOS rename = backup)
#   2. Submit condor preprocessing jobs via submit_preprocessing_jobs.py
#
# Usage:
#   bash repreprocess_filtered.sh [--dry-run]

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

EOS_DATA=/eos/user/d/dgenoves/foundation_model_testing_data
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD

# ─── Datasets and their experiment configs ───────────────────────────────────
declare -A DATASET_EXPERIMENTS
DATASET_EXPERIMENTS["v2_12class_nosparse_highlevel"]="vcreg_12class_nosparse_dmodel128_cern"
DATASET_EXPERIMENTS["v2_nosparse_higgs_allsm_highlevel"]="anomaly_qcd_vs_higgs_raw_allsm_nosparse_cern"
DATASET_EXPERIMENTS["v2_nosparse_higgs_smstats_highlevel"]="anomaly_qcd_vs_higgs_embedding_selfsupcon_nosparse_dmodel128_cern"
DATASET_EXPERIMENTS["v2_18class_highlevel_full"]=""   # no active config — skip

# ─── Step 1: rename preprocessed dirs ────────────────────────────────────────
echo "=== Step 1: rename preprocessed → preprocessed_old ==="
for dataset in "${!DATASET_EXPERIMENTS[@]}"; do
    experiment="${DATASET_EXPERIMENTS[$dataset]}"
    preproc_dir="${EOS_DATA}/${dataset}/preprocessed"
    backup_dir="${EOS_DATA}/${dataset}/preprocessed_old"

    if [ -z "$experiment" ]; then
        echo "  [SKIP] ${dataset}: no experiment config — leaving preprocessed dir untouched"
        continue
    fi
    if [ ! -d "${preproc_dir}" ]; then
        echo "  [SKIP] ${dataset}: no preprocessed dir found"
        continue
    fi
    if [ -d "${backup_dir}" ]; then
        echo "  [WARN] ${dataset}: backup already exists at preprocessed_old — removing it first"
        if [ $DRY_RUN -eq 0 ]; then
            rm -rf "${backup_dir}"
        else
            echo "    [DRY] rm -rf ${backup_dir}"
        fi
    fi

    echo "  mv ${preproc_dir} → ${backup_dir}"
    if [ $DRY_RUN -eq 0 ]; then
        mv "${preproc_dir}" "${backup_dir}"
    fi
done

# ─── Step 2: submit preprocessing jobs ───────────────────────────────────────
echo ""
echo "=== Step 2: submit preprocessing jobs ==="
cd "${PROJECT_DIR}"

for dataset in "${!DATASET_EXPERIMENTS[@]}"; do
    experiment="${DATASET_EXPERIMENTS[$dataset]}"
    if [ -z "$experiment" ]; then
        echo "  [SKIP] ${dataset}: no experiment config"
        continue
    fi

    echo "  Submitting preprocessing for ${dataset} (experiment=${experiment})..."
    if [ $DRY_RUN -eq 0 ]; then
        python scripts/submit_preprocessing_jobs.py \
            data_sources=cern \
            paths=fm_testing \
            experiment="${experiment}"
    else
        echo "    [DRY] python scripts/submit_preprocessing_jobs.py data_sources=cern paths=fm_testing experiment=${experiment}"
    fi
done

echo ""
echo "Done. Monitor condor with: condor_q"
echo "Once preprocessing completes, verify with:"
for dataset in "${!DATASET_EXPERIMENTS[@]}"; do
    echo "  ls ${EOS_DATA}/${dataset}/preprocessed/norm_stats.json"
done
echo ""
echo "Then delete backups with:"
for dataset in "${!DATASET_EXPERIMENTS[@]}"; do
    echo "  rm -rf ${EOS_DATA}/${dataset}/preprocessed_old"
done
