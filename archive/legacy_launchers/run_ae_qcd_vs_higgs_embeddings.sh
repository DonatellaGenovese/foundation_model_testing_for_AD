#!/bin/bash
#SBATCH -A IscrC_ELLIF-HE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=ae_qcd_higgs_emb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "========================================" 
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Move to project directory
cd /leonardo/home/userexternal/dgenoves/foundation_model_testing_for_AD

# Load environment
source ~/.bashrc
conda activate collidenv

# Print environment info
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# MODIFY THIS: Set path to your trained contrastive model checkpoint
ENCODER_CKPT="logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/best.ckpt"

if [ ! -f "$ENCODER_CKPT" ]; then
    echo "ERROR: Encoder checkpoint not found at: $ENCODER_CKPT"
    echo "Please modify this script to point to your trained model!"
    exit 1
fi

# Run autoencoder training on EMBEDDINGS
echo "Training Autoencoder for Anomaly Detection (QCD vs Higgs)"
echo "Mode: EMBEDDINGS from contrastive learning"
echo "Encoder: $ENCODER_CKPT"

python src/train_anomaly_detection.py \
  experiment=anomaly_qcd_vs_higgs_embeddings \
  encoder_ckpt="$ENCODER_CKPT" \
  seed=42

echo "========================================"
echo "Job finished at $(date)"
echo "=========================================="
