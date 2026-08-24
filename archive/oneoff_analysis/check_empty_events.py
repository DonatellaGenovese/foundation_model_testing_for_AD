#!/usr/bin/env python3
"""
Check that no event has all PT slots == 0 across jets/electrons/muons/photons.
Runs on all classes in train/val/test for each dataset.

Usage:
    python3 scripts/check_empty_events.py [dataset_label ...]
"""
import sys
import numpy as np
import json
from pathlib import Path

EOS_DATA = Path("/eos/user/d/dgenoves/foundation_model_testing_data")

DATASETS = sys.argv[1:] or [
    "v2_12class_nosparse_highlevel",
    "v2_nosparse_higgs_allsm_highlevel",
    "v2_nosparse_higgs_smstats_highlevel",
]


def get_pt_cols(feature_map: dict) -> list[int]:
    pt_cols = []
    for cfg in feature_map.values():
        topk = cfg.get("topk")
        if topk is None:
            continue
        start  = cfg["start"]
        n_cols = len(cfg["columns"])
        pt_cols += [start + i * n_cols for i in range(topk)]
    return sorted(pt_cols)


for label in DATASETS:
    data_dir = EOS_DATA / label / "preprocessed"
    print(f"\n{'='*70}")
    print(f"Dataset: {label}")
    print(f"{'='*70}")

    with open(data_dir / "feature_map.json") as f:
        fm = json.load(f)
    pt_cols = get_pt_cols(fm)
    print(f"PT columns checked: {len(pt_cols)} (jets+electrons+muons+photons topk slots)\n")

    total_events = 0
    total_empty  = 0

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            files = sorted(cls_dir.glob("*_x.npy"))
            if not files:
                print(f"  {split}/{cls_dir.name}: NO FILES")
                continue
            cls_events = 0
            cls_empty  = 0
            for f in files:
                X = np.load(f, mmap_mode="r")
                empty = int((X[:, pt_cols] == 0.0).all(axis=1).sum())
                cls_events += len(X)
                cls_empty  += empty
            status = "✓ OK" if cls_empty == 0 else f"✗ EMPTY={cls_empty}"
            print(f"  {split}/{cls_dir.name:50s} {cls_events:>8d} events  {status}")
            total_events += cls_events
            total_empty  += cls_empty

    verdict = "PASS ✓" if total_empty == 0 else f"FAIL ✗  ({total_empty} empty events)"
    print(f"\n  Total: {total_events:,} events — {verdict}")
