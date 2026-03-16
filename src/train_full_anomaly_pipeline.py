"""
Full anomaly detection pipeline: SupCon pretraining + Autoencoder.

Phase 1: Train VanillaSupCon encoder on 15 classes (contrastive pretraining).
Phase 2: Freeze encoder, extract embeddings for 18 classes (15 + 3 Higgs),
         train autoencoder on QCD only, validate MSE separation vs Higgs.

The pipeline returns ae/separation_ratio as the final metric, making it
ready for end-to-end Optuna optimization in a future step.

Usage:
    python src/train_full_anomaly_pipeline.py \
        --phase1-experiment vanillasupcon_15class_pretrain \
        --phase2-experiment anomaly_qcd_vs_higgs_embedding \
        --output-dir /path/to/output

    # With hyperparameter overrides (for future Optuna integration):
    python src/train_full_anomaly_pipeline.py \
        --phase1-experiment vanillasupcon_15class_pretrain \
        --phase2-experiment anomaly_qcd_vs_higgs_embedding \
        --output-dir /path/to/output \
        --phase1-override model.temperature=0.07 trainer.max_epochs=30 \
        --phase2-override model.compression=8
"""

import argparse
import json
import os
import functools
import collections
import typing
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import omegaconf
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import hydra
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

# Allow safe unpickling from trusted checkpoints (mirrors train.py)
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
torch.set_float32_matmul_precision('high')

from src.train import train as train_supcon
from src.extract_embeddings import extract_embeddings_from_loader
from src.train_anomaly_embedding import EmbeddingDataModule
from src.models.autoencoder import AutoencoderLitModule
from src.utils import RankedLogger, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)

CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


def _compose_cfg(config_name: str, overrides: list, output_dir: Path = None) -> DictConfig:
    """
    Compose a Hydra config, clearing any previous GlobalHydra state.

    When using hydra.compose (not @hydra.main), HydraConfig is never initialised,
    so ${hydra:runtime.output_dir} and ${hydra:runtime.cwd} are unresolvable.
    We patch them by injecting concrete paths as overrides before composition.
    """
    cwd = str(Path.cwd())
    path_overrides = [f"paths.work_dir={cwd}"]
    if output_dir is not None:
        path_overrides.append(f"paths.output_dir={output_dir}")
    # path_overrides go first so user overrides can still win
    all_overrides = path_overrides + overrides

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=all_overrides)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: SupCon pretraining
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(experiment: str, output_dir: Path, extra_overrides: list = []):
    """
    Train the SupCon encoder.

    Returns:
        best_ckpt_path: path to the best Phase 1 checkpoint
        model: encoder LightningModule with best weights loaded
        metric_dict: training metrics dict
    """
    log.info("=" * 80)
    log.info("PHASE 1: SUPCON PRETRAINING")
    log.info("=" * 80)

    cfg = _compose_cfg("train.yaml", [f"experiment={experiment}"] + extra_overrides,
                       output_dir=output_dir)

    # train() is wrapped with @task_wrapper; returns (metric_dict, object_dict)
    metric_dict, obj_dict = train_supcon(cfg)

    trainer: Trainer = obj_dict["trainer"]
    model = obj_dict["model"]

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if not best_ckpt or not Path(best_ckpt).exists():
        raise RuntimeError("Phase 1 training produced no valid checkpoint.")

    log.info(f"Phase 1 best checkpoint : {best_ckpt}")
    log.info(f"Phase 1 val/con_loss    : {metric_dict.get('val/con_loss', 'N/A')}")

    # Load best weights (trainer.fit leaves model weights at the last step)
    device = next(model.parameters()).device
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return best_ckpt, model, metric_dict


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save_embeddings(
    model,
    phase2_experiment: str,
    output_dir: Path,
    use_projections: bool = False,
) -> Path:
    """
    Instantiate the 18-class datamodule from the Phase 2 experiment config,
    run the frozen encoder over all splits, and save embeddings to disk.

    The Phase 1 datamodule (15 classes) is NOT reused here because the 3 Higgs
    anomaly classes must be present in val/test for Phase 2 evaluation.
    """
    log.info("=" * 80)
    log.info("EMBEDDING EXTRACTION (18-class datamodule)")
    log.info("=" * 80)

    cfg = _compose_cfg("anomaly_detection.yaml", [f"experiment={phase2_experiment}"],
                       output_dir=output_dir)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")
    try:
        datamodule.setup("test")
    except Exception as e:
        log.warning(f"Test split setup failed: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_dim = None

    split_loaders = {
        "train": datamodule.train_dataloader(),
        "val":   datamodule.val_dataloader(),
        "test":  datamodule.test_dataloader() if hasattr(datamodule, "test_dataloader") else None,
    }

    for split_name, loader in split_loaders.items():
        if loader is None:
            continue
        data = extract_embeddings_from_loader(
            model=model,
            dataloader=loader,
            device=device,
            use_projections=use_projections,
            desc=f"Extracting {split_name}",
        )
        np.savez_compressed(
            output_dir / f"{split_name}_embeddings.npz",
            embeddings=data["embeddings"],
            labels=data["labels"],
        )
        embedding_dim = data["embeddings"].shape[1]
        log.info(f"  {split_name}: {data['embeddings'].shape}")

    metadata = {
        "embedding_dim": int(embedding_dim),
        "use_projections": use_projections,
        "splits": [k for k, v in split_loaders.items() if v is not None],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Embeddings saved to: {output_dir}")
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Autoencoder training
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(
    phase2_experiment: str,
    embedding_dir: Path,
    output_dir: Path,
    extra_overrides: list = [],
) -> float:
    """
    Train the autoencoder on frozen QCD embeddings.
    Val/test include QCD + Higgs to track MSE separation.

    Returns:
        Best ae/separation_ratio (the metric to maximise).
    """
    log.info("=" * 80)
    log.info("PHASE 2: AUTOENCODER TRAINING ON EMBEDDINGS")
    log.info("=" * 80)

    cfg = _compose_cfg("anomaly_detection.yaml",
                       [f"experiment={phase2_experiment}"] + extra_overrides,
                       output_dir=output_dir)

    embedding_datamodule = EmbeddingDataModule(
        embedding_dir=str(embedding_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.get("num_workers", 0),
        normal_classes=cfg.normal_classes,
        anomaly_classes=cfg.get("anomaly_classes", []),
    )
    embedding_datamodule.setup()

    input_dim = embedding_datamodule.embedding_dim
    compression = cfg.model.get("compression", 4)
    depth = cfg.model.get("depth", 2)
    bottleneck = input_dim // compression

    log.info(f"AE: input={input_dim} → bottleneck={bottleneck} (compression={compression}, depth={depth})")

    ae_model = AutoencoderLitModule(
        input_dim=input_dim,
        compression=compression,
        depth=depth,
        dropout=cfg.model.dropout,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        normal_classes_labels=cfg.normal_classes,
        anomaly_classes_labels=cfg.get("anomaly_classes", None),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="ae-{epoch:02d}-{ae/separation_ratio:.4f}",
        monitor="ae/separation_ratio",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="ae/separation_ratio",
        patience=cfg.trainer.get("early_stopping_patience", 20),
        mode="max",
        verbose=True,
    )

    loggers = instantiate_loggers(cfg.get("logger"))

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        logger=loggers,
        callbacks=[checkpoint_callback, early_stop],
        deterministic=True,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
    )

    trainer.fit(ae_model, datamodule=embedding_datamodule)
    trainer.test(ae_model, datamodule=embedding_datamodule,
                 ckpt_path="best", weights_only=False)

    ae_model.plot_reconstruction_errors(output_dir / "plots")

    score = checkpoint_callback.best_model_score
    separation_ratio = float(score) if score is not None else float("nan")
    log.info(f"Phase 2 best separation ratio : {separation_ratio:.4f}")
    log.info(f"Phase 2 best checkpoint       : {checkpoint_callback.best_model_path}")

    return separation_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full_anomaly_pipeline(
    phase1_experiment: str,
    phase2_experiment: str,
    output_dir: Path,
    seed: int = 42,
    use_projections: bool = False,
    phase1_overrides: list = [],
    phase2_overrides: list = [],
    phase1_ckpt: str = None,
) -> float:
    """
    Run the complete two-phase anomaly detection pipeline.

    Args:
        phase1_experiment : Hydra experiment name for Phase 1 (SupCon).
        phase2_experiment : Hydra experiment name for Phase 2 (AE).
        output_dir        : Root directory for all outputs.
        seed              : Global random seed.
        use_projections   : Use projection head (True) or encoder h (False, recommended).
        phase1_overrides  : Extra Hydra key=value overrides for Phase 1.
        phase2_overrides  : Extra Hydra key=value overrides for Phase 2.
        phase1_ckpt       : If provided, skip Phase 1 training and load this checkpoint.

    Returns:
        ae/separation_ratio — the Optuna objective for future optimisation.
    """
    import importlib

    L.seed_everything(seed, workers=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 ─────────────────────────────────────────────────────────────────
    p1_output = output_dir / "phase1"
    p1_output.mkdir(parents=True, exist_ok=True)

    if phase1_ckpt:
        # Skip training — load existing checkpoint directly
        log.info("=" * 80)
        log.info(f"PHASE 1: SKIPPED — loading checkpoint: {phase1_ckpt}")
        log.info("=" * 80)
        cfg_p1 = _compose_cfg("train.yaml", [f"experiment={phase1_experiment}"] + phase1_overrides,
                              output_dir=p1_output)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        module_path, class_name = cfg_p1.model._target_.rsplit(".", 1)
        ModelClass = getattr(importlib.import_module(module_path), class_name)
        model = ModelClass.load_from_checkpoint(phase1_ckpt, map_location=device, weights_only=False)
        model.eval()
        best_ckpt = phase1_ckpt
        p1_metrics = {}
    else:
        best_ckpt, model, p1_metrics = run_phase1(
            experiment=phase1_experiment,
            output_dir=p1_output,
            extra_overrides=phase1_overrides,
        )

    # Embedding extraction ────────────────────────────────────────────────────
    embedding_dir = output_dir / "embeddings"
    extract_and_save_embeddings(
        model=model,
        phase2_experiment=phase2_experiment,
        output_dir=embedding_dir,
        use_projections=use_projections,
    )

    # Phase 2 ─────────────────────────────────────────────────────────────────
    p2_output = output_dir / "phase2"
    p2_output.mkdir(parents=True, exist_ok=True)
    separation_ratio = run_phase2(
        phase2_experiment=phase2_experiment,
        embedding_dir=embedding_dir,
        output_dir=p2_output,
        extra_overrides=phase2_overrides,
    )

    # Summary ─────────────────────────────────────────────────────────────────
    log.info("=" * 80)
    log.info("FULL ANOMALY PIPELINE COMPLETE")
    log.info("=" * 80)
    log.info(f"  Phase 1 checkpoint   : {best_ckpt}")
    log.info(f"  val/con_loss (P1)    : {p1_metrics.get('val/con_loss', 'N/A')}")
    log.info(f"  AE separation ratio  : {separation_ratio:.4f}")

    summary = {
        "phase1_ckpt": str(best_ckpt),
        "embedding_dir": str(embedding_dir),
        "p1_val_con_loss": float(p1_metrics.get("val/con_loss", float("nan"))),
        "ae_separation_ratio": separation_ratio,
    }
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return separation_ratio


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full anomaly detection pipeline: SupCon pretraining + AE"
    )
    parser.add_argument(
        "--phase1-experiment",
        default="vanillasupcon_15class_pretrain",
        help="Hydra experiment name for Phase 1 (SupCon pretraining)",
    )
    parser.add_argument(
        "--phase2-experiment",
        default="anomaly_qcd_vs_higgs_embedding",
        help="Hydra experiment name for Phase 2 (AE on embeddings)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for all pipeline outputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phase1-ckpt",
        default=None,
        help="Skip Phase 1 training and load this existing checkpoint directly",
    )
    parser.add_argument(
        "--use-projections",
        action="store_true",
        default=False,
        help="Use projection head embeddings instead of encoder h (not recommended)",
    )
    parser.add_argument(
        "--phase1-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra overrides for Phase 1, e.g. model.temperature=0.07",
    )
    parser.add_argument(
        "--phase2-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra overrides for Phase 2, e.g. model.compression=8",
    )
    args = parser.parse_args()

    # Ensure project root is cwd (Hydra expects this)
    os.chdir(Path(__file__).parent.parent)

    separation_ratio = run_full_anomaly_pipeline(
        phase1_experiment=args.phase1_experiment,
        phase2_experiment=args.phase2_experiment,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        use_projections=args.use_projections,
        phase1_overrides=args.phase1_override or [],
        phase2_overrides=args.phase2_override or [],
        phase1_ckpt=args.phase1_ckpt,
    )

    print(f"\nFinal ae/separation_ratio: {separation_ratio:.4f}")


if __name__ == "__main__":
    main()
