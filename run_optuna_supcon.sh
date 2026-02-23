#!/bin/bash
#SBATCH -A IscrC_ELLIF-HE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=vanillasupcon_optuna
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# move to project
cd /leonardo/home/userexternal/dgenoves/foundation_model_testing_for_AD

# load env
source ~/.bashrc
conda activate collidenv

# run Optuna sweep
python src/train.py -m \
  experiment=vanillasupcon_6class_pretrain \
  hparams_search=vanillasupcon_probes_optuna \
  paths=fm_testing_cineca data=collide2v_minicineca

echo "Job finished at $(date)"