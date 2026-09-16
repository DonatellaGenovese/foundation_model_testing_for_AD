#!/usr/bin/env python3
"""Step 1 of the interpretability chain: encode the events the paper's embeddings came from.

The paper's embeddings were extracted through the training DataLoader, and the train sample
it reads depends on how the shards happen to be split across loader workers: a different
run reads different events, in different proportions per class, and the mixture and the
choice of K move with them. So the reproducible step 1 is not a new extraction but the
encoding of a fixed list. xai_event_lists.npz names every event of the published
embeddings by shard and row, in the published order; this script encodes exactly those
events with the frozen encoder and writes <split>_embeddings.npz in the format the later
steps read.

On an A100 inside fm_testing.sif, with the float32 matmul precision the paper's run used
("high"), val and test come out bit-identical to the published embeddings; train differs
on 150 of 1,433,738 events by at most 2.3e-3 (events the loader had encoded in a short
final batch), which leaves every result of the chain, Table 6 included, unchanged. Other
hardware gives values that differ at the 1e-3 level.

Usage (inside fm_testing.sif, on a GPU):
    python scripts/xai/encode_event_list.py --lists data/xai_event_lists.npz \\
        --data $FMD/v2_nosparse_higgs_smnorm_highlevel/preprocessed --ckpt $ENC --output-dir $EMB
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA initialises

import numpy as np
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(Path(ROOT) / "scripts" / "xai"))

import torch  # noqa: E402

from common.constants import CLASS_FOLDERS  # noqa: E402
from scripts.run_encoder_seeds_anomaly import load_encoder  # noqa: E402

MODEL_CLASS = "src.models.collide2v_vicreg.COLLIDE2VVCRegLitModule"
LABEL_OF = {folder: label for label, folder in CLASS_FOLDERS.items()}


def encode_split(model, lists, split, data: Path, batch_size: int, device: str):
    files = [str(f) for f in lists[f"{split}_files"]]
    index = lists[f"{split}_file"].astype(np.int64)
    row = lists[f"{split}_row"].astype(np.int64)
    labels = lists[f"{split}_label"].astype(np.int64)

    folder_labels = np.array([LABEL_OF[f.split("/")[0]] for f in files], dtype=np.int64)
    if not np.array_equal(folder_labels[index], labels):
        raise SystemExit(f"{split}: the list's labels disagree with its shard folders")
    xs = [np.load(data / split / f, mmap_mode="r") for f in files]

    n = len(index)
    print(f"{split}: encoding {n:,} events from {len(files)} shards on {device}", flush=True)
    emb = None
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            fi, ro = index[s:s + batch_size], row[s:s + batch_size]
            batch = np.stack([xs[f][r] for f, r in zip(fi, ro)]).astype(np.float32, copy=False)
            e = model.encoder.get_embeddings(torch.from_numpy(batch).to(device)).cpu().numpy()
            if emb is None:
                emb = np.empty((n, e.shape[1]), dtype=np.float32)
            emb[s:s + len(e)] = e
            if (s // batch_size) % 500 == 0:
                print(f"  {s + len(e):>9,}/{n:,}  {time.time() - t0:6.0f}s", flush=True)
    return emb, labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists", type=Path, required=True, help="xai_event_lists.npz")
    ap.add_argument("--data", type=Path, required=True,
                    help="The smnorm dataset's preprocessed/ folder, holding train/ val/ test/")
    ap.add_argument("--ckpt", type=Path, required=True, help="Encoder checkpoint")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=512, help="The extraction config's batch size")
    ap.add_argument("--precision", default="high", choices=["high", "highest"],
                    help="float32 matmul precision; 'high' (TF32 on Ampere) is what the paper's run used")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace embeddings already in --output-dir (by default they are left alone, "
                         "so the published ones cannot be overwritten by mistake)")
    a = ap.parse_args()

    existing = [s for s in a.splits if (a.output_dir / f"{s}_embeddings.npz").exists()]
    if existing and not a.overwrite:
        print(f"Embeddings already present in {a.output_dir} ({', '.join(existing)}) — nothing to do. "
              "Pass --overwrite to replace them, or choose another --output-dir.")
        return 0

    torch.set_float32_matmul_precision(a.precision)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lists = np.load(a.lists)
    model = load_encoder(a.ckpt, MODEL_CLASS)
    model.eval().to(device)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        emb, labels = encode_split(model, lists, split, a.data, a.batch_size, device)
        np.savez_compressed(a.output_dir / f"{split}_embeddings.npz", embeddings=emb, labels=labels)
        print(f"saved {a.output_dir / f'{split}_embeddings.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
