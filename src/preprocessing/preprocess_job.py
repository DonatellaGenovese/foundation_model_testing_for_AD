import argparse
import json
import sys
import os
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.preprocessing.preprocess import PreprocessingPipeline

"""
Preprocess job script for applying preprocessing to data based on a given manifest file.
"""

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("🟢 preprocess_job.py starting...", flush=True)

# ---------------------------------------------------------------------
# Parse manifest path before Hydra gets control
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--manifest-path", type=str, required=True)
args, _ = parser.parse_known_args()

# Make manifest path absolute now (before Hydra may change cwd)
args.manifest_path = os.path.abspath(args.manifest_path)

# ⚠️ Prevent Hydra from seeing our CLI arguments
sys.argv = [sys.argv[0]]


def main():
    # Load the subset manifest for this job
    with open(args.manifest_path, "r") as f:
        full_manifest = json.load(f)

    # Extract embedded paths and preprocess config
    paths = full_manifest.pop("_paths")
    pre_cfg_dict = full_manifest.pop("_preprocess_cfg")
    subset_manifest = full_manifest

    preprocess_cfg = OmegaConf.create(pre_cfg_dict)

    # Instantiate preprocessing pipeline (no class_order/process_to_folder needed for apply)
    pipeline = PreprocessingPipeline(
        paths=paths,
        preprocess_cfg=preprocess_cfg,
        process_to_folder={},
        class_order=[],
        device="cpu",
    )

    # Load already-fitted normalization stats
    stats = pipeline._load_stats_json()
    print(f"🟢 Loaded stats from {pipeline.stats_out}")

    # Apply only to this subset
    pipeline.apply_manifest(subset_manifest, stats)
    print("✅ preprocess_job.py completed subset.")


if __name__ == "__main__":
    main()
