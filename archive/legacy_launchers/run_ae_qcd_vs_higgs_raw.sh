#!/bin/bash
#SBATCH -A IscrC_ELLIF-HE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=ae_qcd_higgs_raw
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

# Run autoencoder training on RAW FEATURES
# Training ONLY on QCD, testing on Higgs as anomalies
echo "Training Autoencoder for Anomaly Detection (QCD vs Higgs)"
echo "Mode: RAW FEATURES (baseline)"

python src/train_anomaly_detection.py \
  experiment=anomaly_qcd_vs_higgs_raw \
  seed=42

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
