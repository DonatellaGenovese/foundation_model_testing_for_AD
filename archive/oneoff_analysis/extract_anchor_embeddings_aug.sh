#!/bin/bash
# =====================================================
#  Extract anchor embeddings for Aug SupCon AE ablation reuse.
#
#  Run this ONCE after Section 1 (encoder ablation) completes,
#  before submitting Section 2 (AE ablation).
#
#  Runs the full pipeline with the anchor encoder checkpoint:
#    - Extracts embeddings → saved to anchor_embeddings_aug/embeddings/
#    - Trains AE with anchor config (free bonus: anchor AE result)
#
#  The embeddings dir is then shared across all 22 AE ablation jobs.
#
#  Usage:
#    bash scripts/extract_anchor_embeddings_aug.sh \
#        /eos/user/d/dgenoves/anomaly_pipeline/ablation/aug/d_model/512/phase1/checkpoints/epoch_XXX.ckpt
#
#  Output:
#    /eos/user/d/dgenoves/anomaly_pipeline/ablation/aug/anchor_embeddings/embeddings/
#      train_embeddings.npz
#      val_embeddings.npz
#      test_embeddings.npz
#      metadata.json
# =====================================================

set -euo pipefail

ANCHOR_CKPT="${1:-}"
if [ -z "${ANCHOR_CKPT}" ]; then
    echo "Usage: bash scripts/extract_anchor_embeddings_aug.sh <path/to/anchor_checkpoint.ckpt>"
    exit 1
fi
if [ ! -f "${ANCHOR_CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${ANCHOR_CKPT}"
    exit 1
fi

PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif
ANCHOR_OUTPUT_DIR=/eos/user/d/dgenoves/anomaly_pipeline/ablation/aug/anchor_embeddings

echo "[`date`] Running anchor pipeline (embedding extraction + anchor AE)"
echo "[`date`] Checkpoint : ${ANCHOR_CKPT}"
echo "[`date`] Output dir : ${ANCHOR_OUTPUT_DIR}"

APPTAINER_FLAGS="--nv --bind /afs:/afs --bind /eos:/eos --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env PROJECT_ROOT=${PROJECT_DIR}"

apptainer exec ${APPTAINER_FLAGS} "${IMAGE}" bash -lc "
    cd ${PROJECT_DIR}
    export PROJECT_ROOT=${PROJECT_DIR}
    python src/train_full_anomaly_pipeline.py \
        --phase1-experiment aug_supcon_15class_ablation_cern \
        --phase2-experiment anomaly_qcd_vs_higgs_embedding_augsupcon_ablation_cern \
        --output-dir ${ANCHOR_OUTPUT_DIR} \
        --phase1-ckpt ${ANCHOR_CKPT} \
        --phase2-override \
            logger.mlflow.run_name=aug_ablation_ae__anchor \
            logger.mlflow.experiment_name=aug_supcon_ablation \
            '+tags=[ablation,aug_supcon,ae_ablation,anchor]'
"

EMBEDDINGS_DIR="${ANCHOR_OUTPUT_DIR}/embeddings"

echo ""
echo "[`date`] Done."
echo ""
echo "Anchor embeddings : ${EMBEDDINGS_DIR}"
echo "Anchor AE results : ${ANCHOR_OUTPUT_DIR}/phase2/"
echo ""
echo "Now fill in src/ablation_aug_ae.sub before submitting Section 2:"
echo "  ANCHOR_CKPT       = ${ANCHOR_CKPT}"
echo "  ANCHOR_EMBEDDINGS = ${EMBEDDINGS_DIR}"
