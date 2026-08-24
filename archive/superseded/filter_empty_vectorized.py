"""
Post-process existing vectorized .npy shards to remove events where all
reconstructed-object slots have PT == 0 (no particles reconstructed).

Must run BEFORE preprocessing — the filter relies on the raw fill=0 padding
convention, which is no longer reliable after normalization.

Usage (single dataset):
    python scripts/filter_empty_vectorized.py \
        --vec-dir /eos/.../dataset/vectorized

Usage (all datasets under a root):
    python scripts/filter_empty_vectorized.py \
        --root /eos/user/d/dgenoves/foundation_model_testing_data

After this script completes, re-run preprocessing for the affected datasets.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
def build_filter_mask(X: np.ndarray, feature_map: dict) -> np.ndarray:
    """True = event has at least one non-padding particle (keep it)."""
    keep = np.zeros(len(X), dtype=bool)
    for cfg in feature_map.values():
        topk = cfg.get("topk")
        if topk is None:
            continue
        start  = cfg["start"]
        n_cols = len(cfg["columns"])
        pt_cols = [start + i * n_cols for i in range(topk)]
        keep |= (X[:, pt_cols] != 0.0).any(axis=1)
    return keep


def filter_shard(x_path: Path, y_path: Path, feature_map: dict) -> tuple[int, int]:
    """Filter one shard in-place. Returns (n_before, n_removed)."""
    X = np.load(x_path)
    y = np.load(y_path)
    mask = build_filter_mask(X, feature_map)
    n_removed = int((~mask).sum())
    if n_removed > 0:
        np.save(x_path, X[mask])
        np.save(y_path, y[mask])
    return len(X), n_removed


def process_vec_dir(vec_dir: Path, dry_run: bool = False) -> dict:
    fm_path = vec_dir / "feature_map.json"
    if not fm_path.exists():
        print(f"  [SKIP] no feature_map.json in {vec_dir}", file=sys.stderr)
        return {}

    with open(fm_path) as f:
        feature_map = json.load(f)

    shards = sorted(vec_dir.rglob("*_x.npy"))
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Processing {vec_dir.name}  ({len(shards)} shards)")

    total_before = total_removed = 0
    for x_path in shards:
        y_path = x_path.with_name(x_path.name.replace("_x.npy", "_y.npy"))
        if not y_path.exists():
            print(f"  [WARN] missing y shard for {x_path.name}", file=sys.stderr)
            continue
        if dry_run:
            X = np.load(x_path)
            mask = build_filter_mask(X, feature_map)
            n_b, n_r = len(X), int((~mask).sum())
        else:
            n_b, n_r = filter_shard(x_path, y_path, feature_map)
        total_before   += n_b
        total_removed  += n_r

    pct = total_removed / total_before * 100 if total_before else 0
    print(f"  → removed {total_removed:,} / {total_before:,} events  ({pct:.2f}%)")
    return {"shards": len(shards), "before": total_before, "removed": total_removed}


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vec-dir", type=Path,
                       help="Path to a single vectorized/ directory")
    group.add_argument("--root", type=Path,
                       help="Root directory; all */vectorized/ sub-dirs are processed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed without modifying files")
    args = parser.parse_args()

    if args.vec_dir:
        dirs = [args.vec_dir]
    else:
        dirs = sorted(args.root.glob("*/vectorized"))
        if not dirs:
            print(f"No */vectorized/ directories found under {args.root}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(dirs)} vectorized directories under {args.root}")

    summary = {}
    for d in dirs:
        key = f"{d.parent.name}/{d.name}" if d.name == "vectorized" else d.name
        summary[key] = process_vec_dir(d, dry_run=args.dry_run)

    print("\n=== Summary ===")
    grand_before = grand_removed = 0
    for name, stats in summary.items():
        if not stats:
            continue
        pct = stats["removed"] / stats["before"] * 100 if stats["before"] else 0
        print(f"  {name}: {stats['removed']:,} / {stats['before']:,} removed  ({pct:.2f}%)")
        grand_before  += stats["before"]
        grand_removed += stats["removed"]
    pct_total = grand_removed / grand_before * 100 if grand_before else 0
    print(f"\n  TOTAL: {grand_removed:,} / {grand_before:,} removed  ({pct_total:.2f}%)")

    if args.dry_run:
        print("\n[DRY-RUN] No files were modified.")
    else:
        print("\nDone. Re-run preprocessing for all affected datasets before training.")


if __name__ == "__main__":
    main()
