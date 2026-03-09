#!/bin/bash
#SBATCH -A IscrC_ELLIF-HE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=vanilla_supcon_15class
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# move to project directory
cd /leonardo/home/userexternal/dgenoves/foundation_model_testing_for_AD

# load environment
source ~/.bashrc
conda activate collidenv

# Print environment info
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# run vanilla supcon training (15 classes)
python src/train.py \
  experiment=vanillasupcon_15class_pretrain \
  paths=fm_testing_cineca \
  data=collide2v_minicineca \
  seed=42 \
  logger.mlflow.run_name=vanillasupcon_15class_training_$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Job finished at $(date)"
