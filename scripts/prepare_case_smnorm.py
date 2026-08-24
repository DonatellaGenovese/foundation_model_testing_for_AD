#!/usr/bin/env python3
"""
Build the CASE held-out-signal dataset, normalised with the SM-only statistics.

Same contract as `prepare_newsig_smnorm.py` — vectorise processes that have never
been touched, then apply the existing 12-SM-class statistics in `apply_only` mode so
the new signals land on the scale the encoder was trained with and never contribute
to defining it. The script fails if the statistics change, which is the only way
`apply_only` can go wrong silently.

ONE THING IS DIFFERENT, AND IT MATTERS. Every CASE process has exactly one parquet
file of 5,000 events, whereas the proxy signals have hundreds (VVV_incl 210,
ttH_incl 1103). `make_split_manifest` assigns *whole files* greedily, filling train,
then val, then test:

    for fname, n in items:
        buckets[split_names[split_idx]].append(fname)
        acc += n
        if acc >= targets_abs[split_idx]: split_idx += 1

With a single file, the first iteration puts it in `train` and the loop ends — val
and test come out empty. Every downstream step reads the *test* split, so the
automatic manifest would silently produce a dataset with no signal in it at all.

So the manifest is built here instead, and passed to `vectorize_to_local` as a dict
(it accepts one, and skips its own construction when given). Each signal's single
file goes to test; QCD_inclusive, which has 13k files, gets real files in all three
splits because it is the reference population for the false-positive rate.

Usage:
    python3 scripts/prepare_case_smnorm.py --dry-run
    python3 scripts/prepare_case_smnorm.py
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import OmegaConf

from src.data.utils import get_all_cols, vectorize_to_local
from src.preprocessing.preprocess import PreprocessingPipeline
from src.train_full_anomaly_pipeline import _compose_cfg

DATA        = Path("/eos/user/d/dgenoves/foundation_model_testing_data")
STATS_LABEL = "v2_12class_nosparse_highlevel"        # SM-only statistics, no signal
DST_LABEL   = "v2_nosparse_case_smnorm_highlevel"
EXPERIMENT  = "new_exp/anomaly_case_smnorm"
NORMAL      = "QCD_inclusive"
QCD_FILES_PER_SPLIT = 3                              # ~30k events per split


def build_manifest(base_dir: Path, folder_map: dict, classnames: list,
                   event_db: dict) -> dict:
    """Every class gets files in every split; only `test` is ever read downstream.

    Returns {folder: {"train": [...], "val": [...], "test": [...]}} with bare
    filenames, the shape vectorize_to_local expects.

    WHY THE SIGNALS APPEAR IN ALL THREE SPLITS. The datamodule refuses to build
    unless `has_enough_events` passes, and that function loops over every
    (split, class) pair and returns False the moment one of them has no `_x.npy`
    files. Leaving the signals out of train and val — the natural choice, since only
    test is used — therefore makes the datamodule raise before any inference runs.
    Each signal has a single 5,000-event file, so it is listed in all three splits;
    the same events are vectorised three times and only the test copy is read, by
    infer_new_signals.py, which loads `test_embeddings.npz` alone. Duplicating rows
    in a split nothing trains on is harmless; an unbuildable datamodule is not.

    QCD files are drawn from the intersection of what is on disk and what the event
    database knows, because `has_enough_events` raises KeyError on a file it cannot
    look up, and the database holds 8,967 of the 13,126 QCD parquets.
    """
    manifest = {}
    for cname in classnames:
        folder = folder_map[cname]
        on_disk = sorted(p.name for p in (base_dir / folder).glob("*.parquet"))
        if not on_disk:
            raise FileNotFoundError(f"no parquet under {base_dir/folder}")
        known = set(event_db.get(folder, {}))
        files = [f for f in on_disk if f in known]
        if not files:
            raise RuntimeError(
                f"none of the {len(on_disk)} parquets under {folder} are in the event "
                f"count database; has_enough_events would raise KeyError"
            )
        if cname == NORMAL:
            k = QCD_FILES_PER_SPLIT
            manifest[folder] = {"train": files[:k], "val": files[k:2 * k],
                                "test": files[2 * k:3 * k]}
        else:
            manifest[folder] = {"train": files, "val": files, "test": files}
    return manifest


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

    base_dir    = Path(cfg.paths.dataset_dir)
    stats_src   = DATA / STATS_LABEL / "preprocessed" / "norm_stats.json"
    vec_dir     = DATA / dst_label / "vectorized"
    preproc_dir = DATA / dst_label / "preprocessed"
    tmp_root    = Path("/tmp/fm_testing_tmp") / dst_label

    if not stats_src.exists():
        print(f"MISSING statistics: {stats_src}")
        return 1
    stats = json.loads(stats_src.read_text())
    meta = stats.get("_meta", {})

    from src.data.utils import load_global_filelist
    event_db = load_global_filelist()
    manifest = build_manifest(base_dir, folder, classnames, event_db)

    print(f"source        : {base_dir}")
    print(f"processes     : {len(classnames)}")
    for i, c in enumerate(classnames):
        m = manifest[folder[c]]
        print(f"   [{i}] {c:30s} train={len(m['train']):>2} val={len(m['val']):>2} "
              f"test={len(m['test']):>2}")
    print(f"SM-only stats : {stats_src}")
    print(f"   fit events : {meta.get('num_examples_fit'):,}  (no signal)")
    print(f"output        : {preproc_dir}")

    if a.dry_run:
        print("\n[dry-run] stopping before writing")
        return 0

    if not a.skip_vectorize:
        print("\n=== vectorising ===")
        dm = hydra.utils.instantiate(cfg.data)
        vec_dir.mkdir(parents=True, exist_ok=True)
        # Keep a copy on disk for provenance; vectorize_to_local uses the dict.
        (vec_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
        vectorize_to_local(
            base_dir=str(base_dir),
            config=dm.datasets_config,
            class_names=dm.classnames,
            folder_map=dm.folder,
            labels_map=dm.labels,
            all_cols=get_all_cols(dm.datasets_config),
            vlen=dm.vlen,
            tmp_vec_dir=dm.paths["tmp_vec_dir"],
            eos_vec_dir=str(vec_dir),
            split_counts=dm.train_val_test_split_per_class,
            read_batch_size=512,
            split_manifest=manifest,
            drop_empty_events=not a.keep_empty,
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
    print("Statistics unchanged: the CASE signals were normalised with the SM-only scale.")

    print("\n=== events per class in the test split ===")
    for c in classnames:
        d = preproc_dir / "test" / folder[c]
        n = sum(1 for _ in d.glob("*_x.npy")) if d.exists() else 0
        print(f"   {c:30s} {n} shard(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
