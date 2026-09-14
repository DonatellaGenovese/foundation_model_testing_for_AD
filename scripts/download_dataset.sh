#!/usr/bin/env bash
# Download the vectorised and preprocessed 12-class dataset from its CERNBox public
# link and unpack it into data/ (git-ignored), so stage 1 can be skipped.
#
#   bash scripts/download_dataset.sh                 # everything, about 44 GB
#   bash scripts/download_dataset.sh preprocessed    # only the preprocessed tree, 18 GB
#   bash scripts/download_dataset.sh vectorized      # only the vectorised tree, 26 GB
#
# Runs take the data from there with  paths.eos_data_dir=<repo>/data.
# Note that the data loader walks the vectorised tree before every run (it skips shards
# that already exist), so training needs both trees, not the preprocessed one alone.
set -euo pipefail

SHARE="${XAI_SHARE_URL:-https://cernbox.cern.ch/s/KEq2Nv7PjB5W44y}"
WHICH="${1:-all}"

case "${WHICH}" in
    all)          FILES=(collide2v_12class_vectorized_train.tar collide2v_12class_vectorized_val_test.tar
                         collide2v_12class_preprocessed_train.tar collide2v_12class_preprocessed_val_test.tar) ;;
    vectorized)   FILES=(collide2v_12class_vectorized_train.tar collide2v_12class_vectorized_val_test.tar) ;;
    preprocessed) FILES=(collide2v_12class_preprocessed_train.tar collide2v_12class_preprocessed_val_test.tar) ;;
    *) echo "usage: $0 [all|vectorized|preprocessed]"; exit 1 ;;
esac

TOKEN="${SHARE##*/}"
HOST="${SHARE%%/s/*}"
BASE="${HOST}/remote.php/dav/public-files/${TOKEN}"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DL="${ROOT}/data/.download"
mkdir -p "${DL}"

for f in "${FILES[@]}" SHA256SUMS; do
    if [ -f "${DL}/${f}" ] && [ "${f}" != SHA256SUMS ]; then
        echo "have ${f}"
    else
        echo "downloading ${f}"
        curl -fL --retry 3 -o "${DL}/${f}.part" "${BASE}/${f}"
        mv "${DL}/${f}.part" "${DL}/${f}"
    fi
done

# SHA256SUMS covers every archive of the share, so only check what was downloaded.
(cd "${DL}" && sha256sum --ignore-missing -c SHA256SUMS)

for f in "${FILES[@]}"; do
    echo "unpacking ${f}"
    tar -xf "${DL}/${f}" -C "${ROOT}"
done
echo
echo "Unpacked into ${ROOT}/data/v2_12class_nosparse_highlevel"
echo "Use it with, for example:"
echo "  python src/train.py experiment=fm_testing_12class_nosparse_dmodel256_cern seed=7 \\"
echo "      paths.eos_data_dir=${ROOT}/data"
echo "The archives in ${DL} can be deleted afterwards."
