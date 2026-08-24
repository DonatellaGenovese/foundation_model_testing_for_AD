#!/usr/bin/env python3
"""
Extract full-SM embeddings for the XAI pipeline.

The embeddings that ship with the anomaly-detection runs are not usable here:
`extract_and_save_embeddings` filters `to_classify` down to
`normal_classes | anomaly_classes`, which for the standard AD config is QCD plus
the three Higgs signals. Fitting the GMM on those means decomposing QCD, not the
SM — while Sec. 3.3 and `scripts/xai/README.md` both require all 12 SM classes.

This script runs the extraction against `new_exp/xai_embeddings_vcreg_smnorm`,
whose `normal_classes` lists all 12 SM processes, so train/val/test carry the
full SM plus the three signals.

Normalisation: the config reads the *smnorm* dataset, whose statistics are
fitted on the 12 SM classes only. This matches the AD run under `ad_results/`
(rerun 2026-08-03/04 with `new_exp/anomaly_embedding_vcreg_smnorm`), so the AE
checkpoint and its val-calibrated threshold apply to these embeddings. The older
*allsm* extraction is NOT interchangeable: its statistics were fitted on 12 SM +
3 signals, so the signal proxy helped define the feature scale it is evaluated
against — the embeddings produced that way were deleted on 2026-08-04.

The AE is deliberately not retrained here: it is a function on embedding vectors
and the encoder is unchanged, so the existing AE checkpoint applies as-is.

Usage:
    python3 scripts/xai/extract_xai_embeddings.py --seed 3 --dmodel 256
    python3 scripts/xai/extract_xai_embeddings.py --seed 3 --dmodel 128 --dry-run
"""

import argparse
import collections
import functools
import gc
import sys
import typing
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import omegaconf
import torch

torch.serialization.add_safe_globals([
    functools.partial,
    torch.optim.AdamW, torch.optim.Adam,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    omegaconf.ListConfig, omegaconf.DictConfig, omegaconf.dictconfig.DictConfig,
    omegaconf.nodes.AnyNode, omegaconf.base.Metadata, omegaconf.base.ContainerMetadata,
    collections.defaultdict, typing.Any,
    list, dict, int,
])
torch.set_float32_matmul_precision("high")

from scripts.run_encoder_seeds_anomaly import find_best_ckpt, load_encoder
from src.train_full_anomaly_pipeline import extract_and_save_embeddings

NEW_EXP    = Path("/eos/user/d/dgenoves/anomaly_pipeline/new_exp")
MODEL_CLASS = "src.models.collide2v_vicreg.COLLIDE2VVCRegLitModule"

# d_model-agnostic: the encoder architecture is restored from the checkpoint.
EXPERIMENT = "new_exp/xai_embeddings_vcreg_smnorm"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dmodel", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    run        = f"vcreg_12class_nosparse_dmodel{a.dmodel}_cern"
    experiment = EXPERIMENT
    seed_dir   = NEW_EXP / run / f"seed_{a.seed}"
    # "smnorm" is in the path on purpose: the allsm embeddings that previously
    # lived under xai_embeddings/ are not interchangeable with these.
    out_dir    = NEW_EXP / "xai_embeddings_smnorm" / run / f"encoder_seed_{a.seed}" / "embeddings"

    ckpt = find_best_ckpt(seed_dir)

    print(f"seed        : {a.seed}")
    print(f"d_model     : {a.dmodel}")
    print(f"encoder ckpt: {ckpt}")
    print(f"experiment  : {experiment}")
    print(f"output      : {out_dir}")

    if (out_dir / "train_embeddings.npz").exists():
        print("\nEmbeddings already present — nothing to do.")
        return 0
    if a.dry_run:
        print("\n[dry-run] stopping before extraction")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_encoder(ckpt, MODEL_CLASS)
    extract_and_save_embeddings(
        model=model,
        phase2_experiment=experiment,
        output_dir=out_dir,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
