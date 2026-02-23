#!/bin/bash
#SBATCH -A IscrC_ELLIF-HE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=aug_supcon
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# move to project
cd /leonardo/home/userexternal/dgenoves/foundation_model_testing_for_AD

# load env
source ~/.bashrc
conda activate collidenv

# run augmented supcon training with probe evaluation
python src/train.py \
  experiment=aug_supcon_6class \
  paths=fm_testing_cineca \
  data=collide2v_minicineca \
  logger.mlflow.run_name=aug_supcon_full_training \
  eval_after_training=true

echo "Job finished at $(date)"
