#!/usr/bin/env bash
# Download the published checkpoints from their CERNBox public link and unpack them
# into data/weights/ (git-ignored).
#
#   bash scripts/download_weights.sh                # encoders + autoencoders, 2.3 GB
#   bash scripts/download_weights.sh encoders       # encoders only, 2.24 GB
#   bash scripts/download_weights.sh autoencoders   # autoencoders only, 0.04 GB
#
# Names carry what identifies a run, so no manifest is needed:
#   data/weights/encoders/<model>_d<dim>_seed<seed>.ckpt        e.g. vcreg_d256_seed3.ckpt
#   data/weights/autoencoders/<model>_d<dim>_seed<seed>_ae.ckpt
# The epoch each run stopped at, and the architecture, are inside the checkpoint
# (`epoch`, `hyper_parameters`). Models: supcon, simclr, vcreg, vicreg, ce (supervised
# baseline, not probed). Seeds differ per model: vcreg 0-4, supcon and simclr
# 7/42/137/1337/31337, vicreg 7/42/12345/1337/31337, ce 0-4.
set -euo pipefail

SHARE="${XAI_SHARE_URL:-https://cernbox.cern.ch/s/KEq2Nv7PjB5W44y}"
WHICH="${1:-all}"
case "${WHICH}" in
    all)          FILES=(weights_encoders.tar weights_autoencoders.tar) ;;
    encoders)     FILES=(weights_encoders.tar) ;;
    autoencoders) FILES=(weights_autoencoders.tar) ;;
    *) echo "usage: $0 [all|encoders|autoencoders]"; exit 1 ;;
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
    tar -xf "${DL}/${f}" -C "${ROOT}"
done
echo "Checkpoints unpacked into ${ROOT}/data/weights"
