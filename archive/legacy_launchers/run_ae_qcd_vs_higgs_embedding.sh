#!/bin/bash
#SBATCH --job-name=ae_emb_qcd_higgs
#SBATCH --output=logs/ae_embedding_%j.out
#SBATCH --error=logs/ae_embedding_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ELLIF-HE
#SBATCH --time=02:00:00
#SBATCH --mem=64GB

# Load environment
source ~/.bashrc
conda activate collidenv

# Navigate to project directory
cd /leonardo/home/userexternal/dgenoves/foundation_model_testing_for_AD

# Run embedding-based anomaly detection
python src/train_anomaly_embedding.py \
    experiment=anomaly_qcd_vs_higgs_embedding \
    ckpt_path=/leonardo/home/userexternal/dgenoves/foundation_model_testing_for_AD/logs/train/runs/2026-03-04_16-03-01/checkpoints/epoch_022.ckpt \
    seed=42

echo "✅ Embedding-based anomaly detection complete!"
