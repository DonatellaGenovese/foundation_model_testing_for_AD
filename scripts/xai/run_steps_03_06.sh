#!/usr/bin/env bash
# Stage 4, step 6: assignments (03), Wasserstein ranking (04), K +- 2 check (05) and
# Spearman correlations (06), for HH->4b and Z'->n(mumu).
#
#   bash scripts/xai/run_steps_03_06.sh
#
# Paths come from scripts/xai/paths.sh; export any of them to override.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/xai/paths.sh

for spec in "hh4b 13 $MH" "hvdilep 20 $MV"; do
  read -r t S M <<< "$spec"
  K7=$XP/k7_${t}_pca64_d256_seed3
  echo "### ${t}: step 03"
  python scripts/xai/03_assign_flagged.py --matched-npz "$M" --signal-label "$S" --gmm-path "$GMM" \
      --ae-checkpoint "$AE" --output-dir "$K7/03_assign_matched" --fpr 0.10 --ylim 0.95 $PCA
  echo "### ${t}: step 04"
  python scripts/xai/04_profile_and_rank.py --matched-npz "$M" --signal-label "$S" --gmm-path "$GMM" \
      --ae-checkpoint "$AE" --output-dir "$XP/rank_k7_sm/$t" --min-frac 0.05 --fpr 0.10 $PCA
  echo "### ${t}: step 05"
  python scripts/xai/05_robustness_kpm2.py --embeddings-dir "$EMB" --matched-npz "$M" --signal-label "$S" \
      --ae-checkpoint "$AE" --k 7 --gmm-dir "$(dirname "$GMM")" --output-dir "$K7/05_robustness" \
      --fpr 0.10 --min-frac 0.05 $PCA
  echo "### ${t}: step 06"
  python scripts/xai/06_ae_mechanism.py --matched-npz "$M" --gmm-path "$GMM" --ae-checkpoint "$AE" \
      --profile-meta "$XP/rank_k7_sm/$t/profile_meta.json" --output-dir "$K7/06_ae_mechanism" $PCA
done
