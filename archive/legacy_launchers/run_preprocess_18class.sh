#!/bin/bash
#SBATCH -A IscrC_ELLIF-HE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --job-name=preprocess_18class
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "========================================" 
echo "Preprocessing 18-class dataset (QCD + Higgs)"
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
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================"

# Run preprocessing through train.py with minimal epochs
# The datamodule will automatically preprocess if data is missing
echo "Running preprocessing for 18-class dataset..."
echo "Classes: QCD_inclusive + 14 SM processes + VBFHbb + HH_4b + ggHtautau"

python src/train.py \
  experiment=anomaly_qcd_vs_higgs_raw \
  preprocess.enabled=true \
  preprocess.mode=fit_and_apply \
  trainer.max_epochs=1 \
  data.train_val_test_split_per_class=[50000,10000,10000] \
  trainer.accelerator=cpu

echo "========================================"
echo "Preprocessing job finished at $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Verify preprocessed data in: /leonardo_scratch/fast/IscrC_ELLIF-HE/donatellagenovese/CollideMini_processed/v2_18class_highlevel/preprocessed/"
echo "2. Launch autoencoder training: sbatch run_ae_qcd_vs_higgs_raw.sh"
