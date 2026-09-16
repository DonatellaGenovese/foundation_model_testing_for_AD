#!/usr/bin/env python3
"""Encode every event of one split with the frozen encoder, in a fixed order.

Reads the preprocessed shards directly — classes in label order, files sorted, rows in
order — instead of the training DataLoader, whose file shuffle, per-worker quota and
shuffle buffer make the event set and order change from run to run. Each embedding is
saved with the (file, row) it came from, so any subset or ordering can be rebuilt
exactly later.

With --events, only the listed (file, row) pairs are encoded, in the listed order: this
is the deterministic extraction of a given event list.

Usage (inside fm_testing.sif, on a GPU):
    python scripts/xai/deterministic/encode_shards.py --split test --out /eos/.../all_test
    python scripts/xai/deterministic/encode_shards.py --split train \\
        --events train_events.npz --out /eos/.../train
"""
from __future__ import annotations

import argparse
import json
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
DATA = "/eos/user/d/dgenoves/foundation_model_testing_data/v2_nosparse_higgs_smnorm_highlevel/preprocessed"
CKPT = ("/eos/user/d/dgenoves/anomaly_pipeline/new_exp/vcreg_12class_nosparse_dmodel256_cern/"
        "seed_3/checkpoints/epoch_014.ckpt")


def shard_list(data: Path, split: str) -> list[tuple[int, str]]:
    """(label, path of the _x.npy shard) for the 15 classes, in label then file order."""
    out = []
    for label in range(15):
        d = data / split / CLASS_FOLDERS[label]
        for f in sorted(p for p in os.listdir(d) if p.endswith("_x.npy")):
            out.append((label, str(d / f)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path(DATA))
    ap.add_argument("--ckpt", type=Path, default=Path(CKPT))
    ap.add_argument("--events", type=Path, default=None,
                    help="npz with file_id and row arrays: encode only these, in this order")
    ap.add_argument("--batch-size", type=int, default=512, help="The extraction config's batch size")
    ap.add_argument("--npz", type=Path, default=None,
                    help="Also write <npz> with 'embeddings' and 'labels', the format of the published "
                         "<split>_embeddings.npz that the interpretability steps read")
    ap.add_argument("--precision", default="high", choices=["high", "highest"],
                    help="float32 matmul precision; 'high' (TF32 on Ampere) is what the paper's run used")
    a = ap.parse_args()

    torch.set_float32_matmul_precision(a.precision)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shards = shard_list(a.data, a.split)
    files = [p for _, p in shards]
    labels_of = np.array([l for l, _ in shards], dtype=np.int64)
    xs = [np.load(p, mmap_mode="r") for p in files]
    sizes = np.array([len(x) for x in xs], dtype=np.int64)

    if a.events is not None:
        ev = np.load(a.events)
        file_id, row = ev["file_id"].astype(np.int64), ev["row"].astype(np.int64)
    else:
        file_id = np.repeat(np.arange(len(files)), sizes)
        row = np.concatenate([np.arange(n) for n in sizes])
    n = len(file_id)
    print(f"{a.split}: {len(files)} shards, {sizes.sum():,} events available, encoding {n:,} on {device}"
          f" (precision {a.precision}, batch {a.batch_size})", flush=True)

    model = load_encoder(a.ckpt, MODEL_CLASS)
    model.eval().to(device)
    emb = None
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, a.batch_size):
            fi, ro = file_id[s:s + a.batch_size], row[s:s + a.batch_size]
            batch = np.stack([xs[f][r] for f, r in zip(fi, ro)]).astype(np.float32, copy=False)
            e = model.encoder.get_embeddings(torch.from_numpy(batch).to(device)).cpu().numpy()
            if emb is None:
                emb = np.empty((n, e.shape[1]), dtype=np.float32)
            emb[s:s + len(e)] = e
            if (s // a.batch_size) % 500 == 0:
                print(f"  {s + len(e):>9,}/{n:,}  {time.time() - t0:6.0f}s", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "embeddings.npy", emb)
    np.save(a.out / "labels.npy", labels_of[file_id])
    np.save(a.out / "file_id.npy", file_id.astype(np.int32))
    np.save(a.out / "row.npy", row.astype(np.int32))
    meta = {
        "split": a.split, "n_events": int(n), "files": files, "ckpt": str(a.ckpt),
        "batch_size": a.batch_size, "precision": a.precision, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "host": os.uname().nodename,
        "events_file": str(a.events) if a.events else None,
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    if a.npz is not None:
        a.npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.npz, embeddings=emb, labels=labels_of[file_id])
        print(f"saved {a.npz}")
    print(f"saved {a.out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
