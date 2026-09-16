# Paths of the interpretability stage (README stage 4). Source it:
#
#   source scripts/xai/paths.sh
#
# The defaults are the locations of the paper's runs on CERN EOS. Any variable already
# set in the environment wins, so a different site only needs to export NE, XP and FMD
# (or any single path) before sourcing. To re-run the chain, export XP (and EMB) to a directory of
# your own: PUB, the published outputs, is only ever read.

: "${NE:=/eos/user/d/dgenoves/anomaly_pipeline/new_exp}"             # encoder and AE runs
: "${PUB:=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper}"          # published outputs (read only)
: "${XP:=$PUB}"                                                     # interpretability outputs
: "${FMD:=/eos/user/d/dgenoves/foundation_model_testing_data}"       # datasets
: "${RUN:=vcreg_12class_nosparse_dmodel256_cern}"

: "${EMB:=$NE/xai_embeddings_smnorm/$RUN/encoder_seed_3/embeddings}"
: "${ENC:=$NE/$RUN/seed_3/checkpoints/epoch_014.ckpt}"
: "${AE:=$NE/ad_results/$RUN/encoder_seed_3/mse_normal/checkpoints/ae-epochepoch=49.ckpt}"
: "${GMM:=$XP/k_selection_v3/vcreg_d256_seed3_diag_pca64/gmm_K7.pkl}"
: "${MH:=$XP/vcreg_d256_seed3_smnorm/04_profile/matched_sm_hh4b.npz}"
: "${MV:=$XP/case_HVdilep_Zp1000_piD2_mumu_d256_seed3/matched_sm_HVdilep_Zp1000_piD2_mumu.npz}"
: "${LISTS:=$PUB/event_lists/xai_event_lists.npz}"   # events of the published embeddings
: "${GMM_REF:=$PUB/k_selection_v3/vcreg_d256_seed3_diag_pca64/gmm_K7.pkl}"   # published K = 7 numbering

PCA="--pca-dim 64 --pca-embeddings-dir $EMB --pca-seed 3"
export NE PUB XP FMD RUN EMB ENC AE GMM MH MV LISTS GMM_REF PCA
