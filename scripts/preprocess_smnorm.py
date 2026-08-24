#!/usr/bin/env python3
"""
Build the `smnorm` dataset: all 15 classes normalised with the SM-only statistics.

Why this exists. Both dataset variants currently in use fit their normalisation
statistics on a sample that includes the BSM signals:

    v2_12class_nosparse_highlevel          531,750 fit events — 12 SM, no signal
    v2_nosparse_higgs_smstats_highlevel    182,146 fit events — QCD + 3 signals
    v2_nosparse_higgs_allsm_highlevel      663,549 fit events — 12 SM + 3 signals

So the HH->4b proxy helped define the feature scale it is later evaluated
against, which contradicts both "statistics are fitted once on the SM training
data" (App. A.1) and "the signal proxy is held out entirely" (Sec. 3.4).

This script writes a dataset where the 12 SM classes *and* the 3 signals are
normalised with the SM-only statistics, by running the pipeline in `apply_only`
mode against a copy of the 12-class norm_stats.json.

The vectorised input is read in place from the allsm tree — nothing is copied,
so the 7 GB of vectorised data is not duplicated.

Usage:
    python3 scripts/preprocess_smnorm.py --dry-run
    python3 scripts/preprocess_smnorm.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omegaconf import OmegaConf

from src.preprocessing.preprocess import PreprocessingPipeline
from src.train_full_anomaly_pipeline import _compose_cfg

DATA        = Path("/eos/user/d/dgenoves/foundation_model_testing_data")
SRC_LABEL   = "v2_nosparse_higgs_allsm_highlevel"      # provides vectorised input (15 classes)
STATS_LABEL = "v2_12class_nosparse_highlevel"          # provides SM-only statistics
DST_LABEL   = "v2_nosparse_higgs_smnorm_highlevel"     # what we are building
EXPERIMENT  = "anomaly_qcd_vs_higgs_embedding_vcreg_nosparse_allsm_dmodel128_cern"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    vec_dir     = DATA / SRC_LABEL / "vectorized"
    stats_src   = DATA / STATS_LABEL / "preprocessed" / "norm_stats.json"
    preproc_dir = DATA / DST_LABEL / "preprocessed"
    tmp_dir     = Path("/tmp/fm_testing_tmp") / DST_LABEL / "preprocessed"

    cfg = _compose_cfg("anomaly_detection.yaml", [f"experiment={EXPERIMENT}"])
    classnames = list(cfg.data.to_classify)
    p2f = OmegaConf.to_container(cfg.data.process_to_folder, resolve=True)
    folder = {c: p2f[c] for c in classnames}

    stats = json.loads(stats_src.read_text())
    meta = stats.get("_meta", {})

    print(f"vectorised input : {vec_dir}")
    print(f"SM-only stats    : {stats_src}")
    print(f"   fit events    : {meta.get('num_examples_fit'):,}  (no signal)")
    print(f"   features      : {meta.get('num_features_expanded')}")
    print(f"output           : {preproc_dir}")
    print(f"classes          : {len(classnames)}")

    if not vec_dir.is_dir():
        print(f"MISSING vectorised input: {vec_dir}")
        return 1
    if a.dry_run:
        print("\n[dry-run] stopping before writing")
        return 0

    # apply_only reads norm_stats.json from the *output* dir, so seed it there.
    preproc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stats_src, preproc_dir / "norm_stats.json")
    print(f"\ncopied SM-only stats into {preproc_dir/'norm_stats.json'}")

    pre_cfg = OmegaConf.to_container(cfg.preprocess, resolve=True)
    pre_cfg["enabled"] = True
    pre_cfg["mode"] = "apply_only"
    pre_cfg["force"] = True        # stats file exists on purpose; do not skip

    PreprocessingPipeline(
        paths={
            "eos_vec_dir":     str(vec_dir),
            "tmp_preproc_dir": str(tmp_dir),
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
    print("Statistics unchanged: signals were normalised with the SM-only scale.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
