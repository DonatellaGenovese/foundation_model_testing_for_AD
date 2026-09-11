#!/usr/bin/env python3
"""
Build the additional-signal dataset, normalised with the SM-only statistics.

Same principle as `preprocess_smnorm.py`, with one difference: those signals were
already vectorised as part of the allsm tree, whereas these processes have never
been touched, so this script vectorises them first and then applies the existing
statistics to the result.

The statistics come from `v2_12class_nosparse_highlevel`, fitted on the 12 SM
classes only. They are copied into the output directory and the preprocessing
runs in `apply_only` mode, so nothing is refitted: the new signals are placed on
the scale the encoder was trained with, and never contribute to defining it. The
script fails if the statistics change, which is the only way `apply_only` can go
wrong silently.

QCD_inclusive is included because it is the reference population for every metric
downstream — the AE's normality and the denominator of the false-positive rate.

Usage:
    python3 scripts/prepare_newsig_smnorm.py --dry-run
    python3 scripts/prepare_newsig_smnorm.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import OmegaConf

from src.data.utils import (
    get_all_cols,
    load_global_filelist,
    make_split_manifest,
    vectorize_to_local,
)
from src.preprocessing.preprocess import PreprocessingPipeline
from src.train_full_anomaly_pipeline import _compose_cfg

DATA        = Path("/eos/user/d/dgenoves/foundation_model_testing_data")
STATS_LABEL = "v2_12class_nosparse_highlevel"          # SM-only statistics, no signal
DST_LABEL   = "v3_nosparse_newsig_smnorm_highlevel"    # what we are building; v3, see config
EXPERIMENT  = "new_exp/anomaly_newsig_smnorm"

# Order in which the file manifest is DRAWN -- not the classes that are built.
# make_split_manifest shuffles each folder's files from a single RNG, folder after
# folder, so the files a folder receives depend on every folder drawn before it. The
# published numbers (appendix Table 9) come from a tree whose manifest was drawn over
# these six folders. Drawing over only the two built here hands HH_bbtautau different
# test files -- RS22000103/129/173 instead of RS22000011/024/076 -- and so different
# events and different numbers. Drawing in the original order and keeping two folders
# reproduces the published selection file for file (checked against the stored v2
# manifest). The four middle entries only advance the RNG; they are never vectorised.
# A process added later must be APPENDED here: that leaves every earlier draw intact.
MANIFEST_RNG_ORDER = ["QCD_HT50toInf", "VVV_incl", "VH_incl", "tttt_incl", "ttH_incl",
                      "HH_bbtautau"]


def build_manifest(folders: list, split_counts) -> dict:
    """Train/val/test file lists for `folders`, drawn in MANIFEST_RNG_ORDER."""
    missing = [f for f in folders if f not in MANIFEST_RNG_ORDER]
    if missing:
        raise SystemExit(f"{missing} not in MANIFEST_RNG_ORDER: append them there, or "
                         f"which files they receive is undefined.")
    full = make_split_manifest(load_global_filelist(), list(split_counts),
                               MANIFEST_RNG_ORDER, seed=42)
    return {f: full[f] for f in folders}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Keep events with no reconstructed object. Off by default, "
                         "matching the SM tree the encoder was trained on — mixing the "
                         "two would treat signal and background differently.")
    ap.add_argument("--skip-vectorize", action="store_true",
                    help="Reuse an existing vectorised tree and only apply the statistics")
    a = ap.parse_args()

    # A tree built without the filter is a different dataset, so it gets its own
    # label: reusing the same one would silently overwrite the filtered tree.
    dst_label = DST_LABEL.replace("nosparse", "sparse") if a.keep_empty else DST_LABEL
    cfg = _compose_cfg("anomaly_detection.yaml", [f"experiment={EXPERIMENT}"])
    classnames = list(cfg.data.to_classify)
    p2f = OmegaConf.to_container(cfg.data.process_to_folder, resolve=True)
    folder = {c: p2f[c] for c in classnames}

    stats_src   = DATA / STATS_LABEL / "preprocessed" / "norm_stats.json"
    vec_dir     = DATA / dst_label / "vectorized"
    preproc_dir = DATA / dst_label / "preprocessed"
    tmp_root    = Path("/tmp/fm_testing_tmp") / dst_label

    stats = json.loads(stats_src.read_text())
    meta = stats.get("_meta", {})

    print(f"processes     : {len(classnames)}")
    for i, c in enumerate(classnames):
        print(f"   [{i}] {c:16s} -> {folder[c]}")
    print(f"splits        : {list(cfg.data.train_val_test_split_per_class)}")
    print(f"SM-only stats : {stats_src}")
    print(f"   fit events : {meta.get('num_examples_fit'):,}  (no signal)")
    print(f"vectorised    : {vec_dir}")
    print(f"output        : {preproc_dir}")

    if not stats_src.exists():
        print(f"MISSING statistics: {stats_src}")
        return 1
    if a.dry_run:
        print("\n[dry-run] stopping before writing")
        return 0

    if not a.skip_vectorize:
        print("\n=== vectorising ===")
        dm = hydra.utils.instantiate(cfg.data)
        manifest = build_manifest([dm.folder[c] for c in dm.classnames],
                                  dm.train_val_test_split_per_class)
        # Written to disk as well as passed in. vectorize_to_local does not save a
        # manifest it is handed, and every later datamodule.prepare_data() -- which the
        # inference jobs call -- reuses split_manifest.json when present and otherwise
        # draws a fresh one in to_classify order, adding the wrong files to this tree.
        manifest_path = Path(dm.paths["eos_vec_dir"]) / "split_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        vectorize_to_local(
            base_dir=dm.paths["dataset_dir"],
            config=dm.datasets_config,
            class_names=dm.classnames,
            folder_map=dm.folder,
            labels_map=dm.labels,
            all_cols=get_all_cols(dm.datasets_config),
            vlen=dm.vlen,
            tmp_vec_dir=dm.paths["tmp_vec_dir"],
            eos_vec_dir=dm.paths["eos_vec_dir"],
            split_counts=dm.train_val_test_split_per_class,
            read_batch_size=512,
            drop_empty_events=not a.keep_empty,
            split_manifest=manifest,
        )
    else:
        print("\n=== vectorising skipped ===")

    print("\n=== applying SM-only statistics ===")
    # apply_only reads norm_stats.json from the *output* dir, so seed it there.
    preproc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stats_src, preproc_dir / "norm_stats.json")
    print(f"copied SM-only stats into {preproc_dir/'norm_stats.json'}")

    pre_cfg = OmegaConf.to_container(cfg.preprocess, resolve=True)
    pre_cfg["enabled"] = True
    pre_cfg["mode"] = "apply_only"
    pre_cfg["force"] = True        # the stats file exists on purpose; do not skip

    PreprocessingPipeline(
        paths={
            "eos_vec_dir":     str(vec_dir),
            "tmp_preproc_dir": str(tmp_root / "preprocessed"),
            "eos_preproc_dir": str(preproc_dir),
        },
        preprocess_cfg=pre_cfg,
        process_to_folder=folder,
        class_order=classnames,
        device="cpu",
    ).run()

    after = json.loads((preproc_dir / "norm_stats.json").read_text()).get("_meta", {})
    print(f"\nnorm_stats after run: fit events = {after.get('num_examples_fit'):,}")
    if after.get("num_examples_fit") != meta.get("num_examples_fit"):
        print("WARNING: statistics were refitted — apply_only did not hold")
        return 1
    print("Statistics unchanged: the new signals were normalised with the SM-only scale.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
