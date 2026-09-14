#!/usr/bin/env bash
# Stage 4, step 7: the figures and Table 6 of Section 4.3, written under the names the
# paper uses into paper/figures/xai/ (and paper/sections/xai/ for the table).
#
#   bash scripts/xai/make_figures.sh
#
# Needs step 6. Paths come from scripts/xai/paths.sh; export any of them to override.
# FIG and TABLE redirect the outputs (defaults: the paper's own files).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/xai/paths.sh

R=$XP/rank_k7_sm
D=${FIG:-paper/figures/xai}
TABLE=${TABLE:-paper/sections/xai/wasserstein_side_by_side.tex}
H=$XP/k7_hh4b_pca64_d256_seed3
V=$XP/k7_hvdilep_pca64_d256_seed3
mkdir -p "$D"

# Fig. 2, component profiles: the script reads <run-dir>/04_profile/
mkdir -p "$R/hh4b/04_profile"
cp "$R/hh4b/profile_meta.json" "$R/hh4b/04_profile/"
ln -sf "$MH" "$R/hh4b/04_profile/matched_sm_hh4b.npz"
python scripts/xai/plot_04_profiles.py --run-dir "$R/hh4b" --assignments "$H/03_assign_matched/assignments.npz" \
    --mark '$HH \to 4b$:4,5' --mark '$Z^{\prime} \to n(\mu\mu)$:2' --output "$D/component_profiles_K7.pdf"

# Spearman figure (only C2 is shown for Z')
python scripts/xai/plot_06_convergence.py --run-dir "$H" --output "$D/spearman_hh4b_K7.pdf"
python scripts/xai/plot_06_convergence.py --run-dir "$V" --components 2 --output "$D/spearman_hvdilep_K7.pdf"

# localisation, top observable, Table 6
cp "$H/03_assign_matched/plots/flagged_assignment.pdf" "$D/localisation_hh4b_K7.pdf"
cp "$V/03_assign_matched/plots/flagged_assignment.pdf" "$D/localisation_hvdilep_K7.pdf"
python paper/figures/make_top_observable.py --xp "$XP" --output "$D/top_observable_K7.pdf"
python paper/make_wasserstein_table.py --xp "$R" --output "$TABLE"

# supplementary: per-component distributions
cp "$R/hh4b/plots/hh4b_vs_sm_k5.pdf"              "$D/dist8_hh4b_C5_K7.pdf"
cp "$R/hh4b/plots/hh4b_vs_sm_k4.pdf"              "$D/dist8_hh4b_C4_K7.pdf"
cp "$R/hvdilep/plots/hv_zp1000_mumu_vs_sm_k2.pdf" "$D/dist8_hvdilep_C2_K7.pdf"
cp "$R/hvdilep/plots/hv_zp1000_mumu_vs_sm_k5.pdf" "$D/dist8_hvdilep_C5_K7.pdf"
cp "$R/hh4b/plots/physics_per_component.pdf"      "$D/physics_per_component_K7.pdf"
echo "figures in $D, Table 6 in $TABLE"
