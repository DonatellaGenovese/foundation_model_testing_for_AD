#!/usr/bin/env bash
# Download the inputs of notebooks/xai_reproduce_load.ipynb from their CERNBox public
# link, verify them and unpack them into data/ (git-ignored), where the notebook looks
# for them.
#
#   bash scripts/download_xai_data.sh
#
# About 4 GB: the saved embeddings and K = 7 mixture, the encoder and autoencoder
# checkpoints, the SM + HH->4b test sets, and the Z'->n(mumu) test set with its parquet.
set -euo pipefail

# Public link of the folder holding the archives; override with XAI_SHARE_URL.
SHARE="${XAI_SHARE_URL:-https://cernbox.cern.ch/s/KEq2Nv7PjB5W44y}"
FILES=(xai_embeddings.tar xai_models.tar xai_testsets.tar xai_case_zprime.tar SHA256SUMS)

if [ -z "${SHARE}" ]; then
    echo "Set the public link first, e.g."
    echo "  XAI_SHARE_URL=https://cernbox.cern.ch/s/<token> bash $0"
    exit 1
fi

# Files inside a public folder share are served over WebDAV under public-files/<token>.
TOKEN="${SHARE##*/}"
HOST="${SHARE%%/s/*}"
BASE="${HOST}/remote.php/dav/public-files/${TOKEN}"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DL="${ROOT}/data/.download"
mkdir -p "${DL}"

for f in "${FILES[@]}"; do
    if [ -f "${DL}/${f}" ]; then
        echo "have ${f}"
    else
        echo "downloading ${f}"
        if ! curl -fL --retry 3 -o "${DL}/${f}.part" "${BASE}/${f}"; then
            rm -f "${DL}/${f}.part"
            echo "Could not fetch ${BASE}/${f}"
            echo "Open ${SHARE} in a browser, check the file names, and download them into ${DL}."
            exit 1
        fi
        mv "${DL}/${f}.part" "${DL}/${f}"
    fi
done

# SHA256SUMS covers every archive of the share, so only check what was downloaded.
(cd "${DL}" && sha256sum --ignore-missing -c SHA256SUMS)

# The archives hold paths starting with data/, so they unpack into the repository root.
for f in "${FILES[@]}"; do
    [[ "${f}" == *.tar ]] && tar -xf "${DL}/${f}" -C "${ROOT}"
done
echo "Inputs unpacked into ${ROOT}/data; the archives stay in ${DL} and can be deleted."
