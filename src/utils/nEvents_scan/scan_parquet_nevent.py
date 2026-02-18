#!/usr/bin/env python3
"""
Scan all parquet files under each process folder and record their number of events.

Usage:
    # Use config (recommended):
    python scan_parquet_nevent.py data_sources=local4090
    python scan_parquet_nevent.py data_sources=cern
    python scan_parquet_nevent.py data_sources=cineca

    # Legacy mode (still works):
    python scan_parquet_nevent.py --local
    python scan_parquet_nevent.py

Outputs:
    file_event_counts.json or file_event_counts_local.json
"""

import os
import json
import sys
from omegaconf import OmegaConf
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import hydra
from pathlib import Path


def count_events(path: str) -> tuple[str, int | str]:
    """Return the number of events in a Parquet file."""
    try:
        pf = pq.ParquetFile(path)
        # Try standard num_rows first
        nrows = pf.metadata.num_rows
        if nrows is None:
            # Try custom metadata fallback (nEvents from your writer)
            meta = pf.metadata.metadata or {}
            if b"nEvents" in meta:
                nrows = int(meta[b"nEvents"])
            else:
                # Sum row groups if both missing (rare)
                nrows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        return os.path.basename(path), nrows
    except Exception as e:
        return os.path.basename(path), f"error: {e}"


def scan_dataset(base_dir: str, process_to_folder: dict, output_json: str, n_workers: int = 8):
    """Iterate over all folders and scan parquet files in parallel."""
    summary = {}
    for process_name, folder_name in process_to_folder.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"⚠️  Missing folder: {folder_path}, skipping")
            continue

        files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if f.endswith(".parquet")]
        print(f"🔍 Scanning {folder_name} ({len(files)} files)")

        class_meta = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for fname, nrows in tqdm(ex.map(count_events, files), total=len(files)):
                if isinstance(nrows, int):
                    class_meta[fname] = nrows
                else:
                    print(f"  ⚠️  {folder_name}/{fname}: {nrows}")

        summary[folder_name] = class_meta
        print(f"  ✅ {folder_name}: {len(class_meta)} files, {sum(class_meta.values()):,} events")

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print("=" * 80)
    print(f"✅ Saved {output_json} ({len(summary)} datasets)")
    print(f"   Total files: {sum(len(v) for v in summary.values())}")
    print(f"   Total events: {sum(sum(v.values()) for v in summary.values()):,}")
    print("=" * 80)


def load_config_for_scan(config_name: str = "local"):
    """
    Load data sources config for scanning.

    Args:
        config_name: 'local4090' or 'cern' or 'localA6000' or 'cineca'

    Returns:
        dict with base_dir, process_to_folder, output_json, n_workers
    """
    # Load data sources config
    config_path = Path(f"configs/data_sources/{config_name}.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Available: configs/data_sources/local4090.yaml, cern.yaml, localA6000.yaml, or cineca.yaml"
        )

    ds_cfg = OmegaConf.load(config_path)

    # Load data config to get process_to_folder mapping
    if config_name == "local4090":
        data_cfg_path = "configs/data/collide2v_mini4090.yaml"
    elif config_name == "localA6000":
        data_cfg_path = "configs/data/collide2v_miniA6000.yaml"
    elif config_name == "cineca":
        data_cfg_path = "configs/data/collide2v_minicineca.yaml"
    else:
        data_cfg_path = "configs/data/collide2v_basic.yaml"

    data_cfg = OmegaConf.load(data_cfg_path)

    return {
        "base_dir": ds_cfg.data_sources.raw_data_base_dir,
        "process_to_folder": data_cfg.process_to_folder,
        "output_json": ds_cfg.data_sources.event_counts_json,
        "n_workers": 8 if config_name == "local4090" or config_name == "localA6000" else 16,
        "environment": config_name
    }


if __name__ == "__main__":
    # Parse arguments
    args = sys.argv[1:]

    # Check for config-based mode
    use_config = False
    config_name = "local4090"

    for arg in args:
        if arg.startswith("data_sources="):
            use_config = True
            config_name = arg.split("=")[1]
            break

    # =========================================================================
    # CONFIG MODE (New, recommended)
    # =========================================================================
    if use_config:
        print("=" * 80)
        print(f"📋 CONFIG MODE - Using data_sources={config_name}")
        print("=" * 80)

        try:
            cfg = load_config_for_scan(config_name)

            print(f"Environment: {cfg['environment']}")
            print(f"Base dir:    {cfg['base_dir']}")
            print(f"Output:      {cfg['output_json']}")
            print(f"Classes:     {len(cfg['process_to_folder'])}")
            print("=" * 80)
            print()

            scan_dataset(
                cfg['base_dir'],
                cfg['process_to_folder'],
                cfg['output_json'],
                cfg['n_workers']
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            print()
            print("Usage:")
            print("  python scan_parquet_nevent.py data_sources=local4090")
            print("  python scan_parquet_nevent.py data_sources=cern")
            print("  python scan_parquet_nevent.py data_sources=cineca")
            sys.exit(1)
