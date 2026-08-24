#!/bin/bash
#SBATCH -A IscrC_ELLIF-HE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=augmented_supcon_optuna
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# move to project
cd /leonardo/home/userexternal/dgenoves/foundation_model_testing_for_AD

# load env
source ~/.bashrc
conda activate collidenv

# run Optuna sweep for augmented SupCon
python src/train.py -m \
  experiment=aug_supcon_6class \
  hparams_search=augmented_supcon_probes_optuna \
  paths=fm_testing_cineca \
  data=collide2v_minicineca \
  trainer.max_epochs=20

echo "Job finished at $(date)"
