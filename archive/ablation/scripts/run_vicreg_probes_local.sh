#!/bin/bash
# Run all VICReg d_model probe evals sequentially on the login node.
# Usage: bash scripts/ablation/run_vicreg_probes_local.sh
# Logs written to logs/local_runs/vicreg_probes/

set -euo pipefail

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
WRAPPER=${PROJECT_DIR}/scripts/ablation/wrapper_eval_probe_ablation.sh
LOG_DIR=${PROJECT_DIR}/logs/local_runs/vicreg_probes
mkdir -p ${LOG_DIR}

export CUDA_VISIBLE_DEVICES=0

JOBS=(
  "ablation/training/vicreg_dmodel32|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-27-30/checkpoints/epoch_032.ckpt|7|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-27-30"
  "ablation/training/vicreg_dmodel32|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-25-36/checkpoints/epoch_046.ckpt|42|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-25-36"
  "ablation/training/vicreg_dmodel32|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-25-57/checkpoints/epoch_009.ckpt|137|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-25-57"
  "ablation/training/vicreg_dmodel32|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-25-59/checkpoints/epoch_005.ckpt|1337|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-25-59"
  "ablation/training/vicreg_dmodel32|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-28-32/checkpoints/epoch_046.ckpt|31337|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-28-32"
  "ablation/training/vicreg_dmodel64|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-29-28/checkpoints/epoch_001.ckpt|7|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-29-28"
  "ablation/training/vicreg_dmodel64|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-30-46/checkpoints/epoch_036.ckpt|42|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-30-46"
  "ablation/training/vicreg_dmodel64|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-29-17/checkpoints/epoch_003.ckpt|137|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-29-17"
  "ablation/training/vicreg_dmodel64|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-33-00/checkpoints/epoch_000.ckpt|1337|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-33-00"
  "ablation/training/vicreg_dmodel64|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-33-26/checkpoints/epoch_000.ckpt|31337|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-33-26"
  "ablation/training/vicreg_dmodel256|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-42-07/checkpoints/epoch_037.ckpt|7|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-42-07"
  "ablation/training/vicreg_dmodel256|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-42-33/checkpoints/epoch_014.ckpt|42|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-42-33"
  "ablation/training/vicreg_dmodel256|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-42-45/checkpoints/epoch_019.ckpt|137|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-42-45"
  "ablation/training/vicreg_dmodel256|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-44-44/checkpoints/epoch_049.ckpt|1337|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-44-44"
  "ablation/training/vicreg_dmodel256|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-44-10/checkpoints/epoch_001.ckpt|31337|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-44-10"
  "ablation/training/vicreg_dmodel512|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-44-43/checkpoints/epoch_000.ckpt|7|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-44-43"
  "ablation/training/vicreg_dmodel512|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-43-39/checkpoints/epoch_049.ckpt|42|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-43-39"
  "ablation/training/vicreg_dmodel512|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-45-15/checkpoints/epoch_049.ckpt|137|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-45-15"
  "ablation/training/vicreg_dmodel512|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-46-02/checkpoints/epoch_049.ckpt|1337|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-46-02"
  "ablation/training/vicreg_dmodel512|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-46-52/checkpoints/epoch_009.ckpt|31337|/eos/user/d/dgenoves/anomaly_pipeline/ablation/d_model/logs/train/runs/2026-06-26_15-46-52"
)

TOTAL=${#JOBS[@]}
DONE=0
FAILED=0

echo "[$(date)] Starting VICReg local probe eval — ${TOTAL} jobs"
echo "Log dir: ${LOG_DIR}"
echo ""

for job in "${JOBS[@]}"; do
  IFS='|' read -r EXPERIMENT CKPT_PATH SEED OUTPUT_DIR <<< "$job"
  export EXPERIMENT CKPT_PATH SEED OUTPUT_DIR

  JOB_NAME=$(echo $EXPERIMENT | sed 's|ablation/training/||')_seed${SEED}
  LOG_OUT=${LOG_DIR}/${JOB_NAME}.out
  LOG_ERR=${LOG_DIR}/${JOB_NAME}.err

  # Skip if already done
  if eos root://eosuser.cern.ch ls "${OUTPUT_DIR}/probe_evaluation/probe_results.json" &>/dev/null; then
    echo "[SKIP] ${JOB_NAME} — already done"
    ((DONE++)) || true
    continue
  fi

  echo "[$(date)] Running: ${JOB_NAME}"
  if bash ${WRAPPER} "local" > "${LOG_OUT}" 2> "${LOG_ERR}"; then
    echo "[$(date)] ✓ ${JOB_NAME}"
    ((DONE++)) || true
  else
    echo "[$(date)] ✗ ${JOB_NAME} — see ${LOG_ERR}"
    ((FAILED++)) || true
  fi
done

echo ""
echo "[$(date)] Done: ${DONE} succeeded, ${FAILED} failed out of ${TOTAL} total"
