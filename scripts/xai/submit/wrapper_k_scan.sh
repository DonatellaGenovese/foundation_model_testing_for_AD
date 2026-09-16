#!/bin/bash
# HTCondor wrapper — mixtures K = 5..12 and choice of K, in one space.
#   SPACE    pca64 (the paper's mixture) or raw (unprojected)
#   THREADS  BLAS threads (the paper's runs: 8 for pca64, 16 for raw)
#   XP, EMB  as in scripts/xai/paths.sh
set -euo pipefail
PROJECT_DIR=/afs/cern.ch/user/d/dgenoves/foundation_model_testing_for_AD
IMAGE=/eos/user/d/dgenoves/fm_testing.sif

echo "[$(date)] K scan (${SPACE}) — host $(hostname)"
apptainer exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs "${IMAGE}" bash -lc "
  set -euo pipefail
  cd ${PROJECT_DIR}
  export PROJECT_ROOT=${PROJECT_DIR} ${XP:+XP=${XP}} ${EMB:+EMB=${EMB}}
  export OMP_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS}
  export MKL_CBWR=AVX2 OPENBLAS_CORETYPE=Haswell PYTHONHASHSEED=0 PYTHONWARNINGS=ignore TQDM_DISABLE=1
  source scripts/xai/paths.sh
  if [ \"\$(realpath -m \$XP)\" = \"\$(realpath -m \$PUB)\" ]; then
    echo 'XP is the published outputs directory: export XP to a directory of your own.' >&2; exit 1
  fi
  if [ '${SPACE}' = pca64 ]; then PCA_VAR=64; TAG=diag_pca64; else PCA_VAR=0; TAG=diag; fi
  G=\$XP/k_selection_v3/vcreg_d256_seed3_\$TAG
  python scripts/xai/select_k_interpretable.py --embeddings-dir \$EMB --matched-npz \$MH --output-dir \$G \
      --k-values 5 6 7 8 9 10 11 12 --n-init 5 --n-restarts 12 --cov-type diag --pca-var \$PCA_VAR
  python -u scripts/xai/select_k_profiles.py --embeddings-dir \$EMB --matched-npz \$MH --gmm-dir \$G \
      --output-dir \$XP/k_profiles/vcreg_d256_seed3_${SPACE} --k-values 3 4 5 6 7 8 9 10 11 12 \
      --pca-dim \$PCA_VAR --n-perm 200 --seed 3
  if [ '${SPACE}' = pca64 ]; then
    cp \$G/gmm_K7.pkl \$G/gmm_K7_as_fitted.pkl
    python scripts/xai/align_mixture.py --gmm \$G/gmm_K7_as_fitted.pkl --reference \$GMM_REF --out \$G/gmm_K7.pkl
  fi
"
echo "[$(date)] done"
