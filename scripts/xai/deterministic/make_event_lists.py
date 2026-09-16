#!/usr/bin/env python3
"""Write the event lists of the published interpretability embeddings in a portable form.

match_events.py names each published event by `file_id`, its position in the sorted shard
list that encode_shards.py read (recovery/all_<split>/meta.json), and `row`. This
replaces the position with the shard's name relative to preprocessed/<split>/, stored in
the output itself, so the list no longer depends on how a directory is listed or where
the dataset lives.

Output, one npz, for each split:
    <split>_files   shard names, "<class folder>/<file>_x.npy"
    <split>_file    index into <split>_files, one per event, in the published order
    <split>_row     row inside that shard
    <split>_label   class label (15-class numbering), checked here against the published
                    embeddings and re-checked by encode_event_list.py

Usage:
    python scripts/xai/deterministic/make_event_lists.py \\
        --recovery /eos/user/d/dgenoves/xai_deterministic/recovery \\
        --published-embeddings /eos/.../encoder_seed_3/embeddings --out xai_event_lists.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.constants import CLASS_FOLDERS  # noqa: E402

LABEL_OF = {folder: label for label, folder in CLASS_FOLDERS.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recovery", type=Path, required=True)
    ap.add_argument("--published-embeddings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    arrays = {}
    for split in ("train", "val", "test"):
        ev = np.load(a.recovery / f"{split}_events.npz")
        paths = json.load(open(a.recovery / f"all_{split}" / "meta.json"))["files"]
        names = ["/".join(Path(p).parts[-2:]) for p in paths]
        file_id, row = ev["file_id"].astype(np.int64), ev["row"].astype(np.int64)

        used = np.unique(file_id)
        remap = np.full(len(names), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        files = np.array([names[i] for i in used])
        label_of_file = np.array([LABEL_OF[n.split("/")[0]] for n in files], dtype=np.int64)
        index = remap[file_id]
        labels = label_of_file[index]

        published = np.load(a.published_embeddings / f"{split}_embeddings.npz")["labels"]
        if not np.array_equal(labels, published):
            raise SystemExit(f"{split}: labels from the shard names disagree with the published embeddings")

        arrays[f"{split}_files"] = files
        arrays[f"{split}_file"] = index.astype(np.int16)
        arrays[f"{split}_row"] = row.astype(np.int32)
        arrays[f"{split}_label"] = labels.astype(np.int8)
        print(f"{split}: {len(index):,} events from {len(files)} shards; labels match the published embeddings")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    print(f"saved {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
