"""
Full anomaly detection pipeline: contrastive encoder pretraining + Autoencoder.

Phase 1: Train a contrastive encoder on 12 SM classes (nosparse).
Phase 2: Freeze encoder, extract embeddings, train autoencoder on normal
         classes only, evaluate anomaly detection vs 3 Higgs signals.

Usage:
    python src/train_full_anomaly_pipeline.py \
        --phase1-experiment vcreg_12class_nosparse_dmodel256_cern \
        --phase2-experiment anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern \
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/full_pipeline/seed_0

    # With hyperparameter overrides:
    python src/train_full_anomaly_pipeline.py \
        --phase1-experiment vcreg_12class_nosparse_dmodel256_cern \
        --phase2-experiment anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern \
        --output-dir /path/to/output \
        --phase1-override model.temperature=0.07 trainer.max_epochs=30 \
        --phase2-override model.compression=8
"""

import argparse
import json
import os
import re
import functools
import collections
import typing
from typing import List, Optional
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
        torch.optim.lr_scheduler.OneCycleLR,
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
    log.info("EMBEDDING EXTRACTION (normal + anomaly classes only)")
    log.info("=" * 80)

    cfg = _compose_cfg("anomaly_detection.yaml", [f"experiment={phase2_experiment}"],
                       output_dir=output_dir)

    # Filter to_classify to only the classes needed for AE training/evaluation:
    # normal_classes (QCD) + anomaly_classes (Higgs).
    needed_indices = set(cfg.normal_classes) | set(cfg.get("anomaly_classes", []))
    all_classes = list(cfg.data.to_classify)
    sorted_needed = sorted(needed_indices)
    needed_classes = [all_classes[i] for i in sorted_needed if i < len(all_classes)]
    log.info(f"Extracting embeddings for {len(needed_classes)} classes (normal + anomaly): {needed_classes}")

    # After filtering to_classify, the datamodule assigns labels 0..N-1 (re-indexed).
    # Build a remap so saved labels match the original class indices expected by
    # EmbeddingDataModule (e.g. QCD=0, VBFHbb=15, HH_4b=16, ggHtautau=17).
    # re-indexed 0 → original sorted_needed[0], re-indexed 1 → sorted_needed[1], ...
    label_remap = {new_idx: orig_idx for new_idx, orig_idx in enumerate(sorted_needed)}
    log.info(f"Label remap (re-indexed → original): {label_remap}")

    with omegaconf.open_dict(cfg):
        cfg.data.to_classify = needed_classes

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

    os.makedirs(output_dir, exist_ok=True)
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
        # Remap labels from re-indexed back to original class indices
        labels = data["labels"].copy()
        for new_idx, orig_idx in label_remap.items():
            labels[data["labels"] == new_idx] = orig_idx

        np.savez_compressed(
            output_dir / f"{split_name}_embeddings.npz",
            embeddings=data["embeddings"],
            labels=labels,
        )
        embedding_dim = data["embeddings"].shape[1]
        log.info(f"  {split_name}: {data['embeddings'].shape}, labels: { {int(orig): int((labels==orig).sum()) for orig in sorted_needed} }")

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
    monitor: str = "ae/val_drift_metric",
    val_signal_classes: Optional[List[int]] = None,
) -> dict:
    """
    Train the autoencoder on frozen QCD embeddings.

    Args:
        val_signal_classes: anomaly classes visible during validation monitoring.
            None        → inherit from anomaly_classes (all signals)
            []          → no signals in val (Strategy 1: unsupervised drift)
            [15,16,17]  → all signals in val (Strategy 2)
            [17]        → single signal in val (Strategy 3)
    """
    log.info("=" * 80)
    log.info("PHASE 2: AUTOENCODER TRAINING ON EMBEDDINGS")
    log.info("=" * 80)

    # mode=max for AUROC/separation monitors; mode=min for drift (lower = better)
    _max_monitors = {"ae/val_auroc", "ae/separation_ratio"}
    checkpoint_mode = "max" if (monitor in _max_monitors or "auroc" in monitor) else "min"
    log.info(f"Checkpoint monitor: {monitor}  (mode={checkpoint_mode})")

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
        val_signal_classes=val_signal_classes,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="ae-epoch{epoch:02d}",
        monitor=monitor,
        mode=checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=cfg.trainer.get("early_stopping_patience", 20),
        mode=checkpoint_mode,
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

    # Collect comprehensive per-signal test metrics from model (populated by on_test_epoch_end)
    per_signal_metrics: dict = dict(getattr(ae_model, "_test_metrics", {}))

    _sep = trainer.callback_metrics.get("ae/separation_ratio", float("nan"))
    separation_ratio = float(_sep.item() if hasattr(_sep, "item") else _sep)

    drift_metric = per_signal_metrics.get(
        "drift_metric",
        float(trainer.callback_metrics.get("ae/drift_metric", float("nan")))
    )
    drift_per_fpr = {
        f"drift_fpr{tag}": per_signal_metrics.get(
            f"drift_fpr{tag}",
            float(trainer.callback_metrics.get(f"ae/drift_fpr{tag}", float("nan")))
        )
        for tag in ["01", "05", "10"]
    }

    # val drift restored from best checkpoint via on_load_checkpoint
    val_drift_metric = ae_model._val_drift_metric

    best_ckpt_path = checkpoint_callback.best_model_path
    best_epoch = None
    if best_ckpt_path:
        m = re.search(r"epoch=(\d+)", Path(best_ckpt_path).stem)
        if m:
            best_epoch = int(m.group(1))

    log.info(f"Phase 2 best separation ratio : {separation_ratio:.4f}")
    log.info(f"Phase 2 val drift metric      : {val_drift_metric:.4f}  (HPO objective)")
    log.info(f"Phase 2 test drift metric     : {drift_metric:.4f}  (final report only)")
    for k, v in drift_per_fpr.items():
        log.info(f"Phase 2 {k}               : {v:.4f}")
    log.info(f"Phase 2 best checkpoint       : {best_ckpt_path}  (epoch={best_epoch})")

    return {
        "separation_ratio": separation_ratio,
        "val_drift_metric": val_drift_metric,
        "drift_metric": drift_metric,
        **drift_per_fpr,
        "best_epoch": best_epoch,
        "per_signal": per_signal_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Raw AE baseline (no Phase 1 encoder)
# ─────────────────────────────────────────────────────────────────────────────

def run_raw_ae_baseline(
    experiment: str,
    output_dir: Path,
    extra_overrides: list = [],
    monitor: str = "ae/val_drift_metric",
    val_signal_classes: Optional[List[int]] = None,
    score_classes: Optional[List[int]] = None,
) -> dict:
    """
    Train AE directly on raw input features — no Phase 1 encoder.

    Uses RawNpyDataModule (TensorDataset in-memory) for fast, reliable loading
    that avoids ShuffleBuffer/IterableDataset issues with COLLIDE2V.

    Args:
        experiment       : Hydra experiment name (e.g. anomaly_qcd_vs_higgs_raw_cern)
        val_signal_classes: same semantics as run_phase2()
    """
    from src.data.raw_npy_datamodule import RawNpyDataModule

    log.info("=" * 80)
    log.info("RAW AE BASELINE: AUTOENCODER ON RAW INPUT FEATURES")
    log.info("=" * 80)

    _max_monitors = {"ae/val_auroc", "ae/separation_ratio"}
    checkpoint_mode = "max" if (monitor in _max_monitors or "auroc" in monitor) else "min"
    log.info(f"Checkpoint monitor: {monitor}  (mode={checkpoint_mode})")

    cfg = _compose_cfg("anomaly_detection.yaml",
                       [f"experiment={experiment}"] + extra_overrides,
                       output_dir=output_dir)

    normal_classes  = list(cfg.normal_classes)
    anomaly_classes = list(cfg.get("anomaly_classes", []))
    split           = list(cfg.data.train_val_test_split_per_class)
    n_train, n_val, n_test = split[0], split[1], split[2]

    preprocessed_dir = Path(cfg.paths.eos_data_dir) / cfg.data.label / "preprocessed"
    log.info(f"Preprocessed dir: {preprocessed_dir}")

    # Build class_folders from to_classify + process_to_folder so any class indices work
    to_classify = list(cfg.data.to_classify)
    process_to_folder = dict(cfg.data.get("process_to_folder", {}))
    class_folders = {
        i: process_to_folder.get(name, name)
        for i, name in enumerate(to_classify)
    }

    datamodule = RawNpyDataModule(
        preprocessed_dir=preprocessed_dir,
        normal_classes=normal_classes,
        anomaly_classes=anomaly_classes,
        class_folders=class_folders,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        batch_size=cfg.data.batch_size,
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    input_dim = datamodule.input_dim
    compression = cfg.model.get("compression", 16)
    depth = cfg.model.get("depth", 3)
    bottleneck = input_dim // compression

    log.info(f"Raw AE: input={input_dim} → bottleneck={bottleneck} (compression={compression}, depth={depth})")

    ae_model = AutoencoderLitModule(
        input_dim=input_dim,
        compression=compression,
        depth=depth,
        dropout=cfg.model.dropout,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        normal_classes_labels=list(cfg.normal_classes),
        anomaly_classes_labels=list(cfg.get("anomaly_classes", [])) or None,
        val_signal_classes=val_signal_classes,
        score_classes=score_classes,
    )

    os.makedirs(output_dir / "checkpoints", exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="ae-epoch{epoch:02d}",
        monitor=monitor,
        mode=checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=cfg.trainer.get("early_stopping_patience", 20),
        mode=checkpoint_mode,
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

    trainer.fit(ae_model, datamodule=datamodule)
    trainer.test(ae_model, datamodule=datamodule, ckpt_path="best", weights_only=False)

    ae_model.plot_reconstruction_errors(output_dir / "plots")

    per_signal_metrics: dict = dict(getattr(ae_model, "_test_metrics", {}))

    _sep = trainer.callback_metrics.get("ae/separation_ratio", float("nan"))
    separation_ratio = float(_sep.item() if hasattr(_sep, "item") else _sep)

    drift_metric = per_signal_metrics.get(
        "drift_metric",
        float(trainer.callback_metrics.get("ae/drift_metric", float("nan")))
    )
    drift_per_fpr = {
        f"drift_fpr{tag}": per_signal_metrics.get(
            f"drift_fpr{tag}",
            float(trainer.callback_metrics.get(f"ae/drift_fpr{tag}", float("nan")))
        )
        for tag in ["01", "05", "10"]
    }
    val_drift_metric = ae_model._val_drift_metric

    best_ckpt_path = checkpoint_callback.best_model_path
    best_epoch = None
    if best_ckpt_path:
        m = re.search(r"epoch=(\d+)", Path(best_ckpt_path).stem)
        if m:
            best_epoch = int(m.group(1))

    log.info(f"Raw AE separation ratio  : {separation_ratio:.4f}")
    log.info(f"Raw AE val drift metric  : {val_drift_metric:.4f}")
    log.info(f"Raw AE test drift metric : {drift_metric:.4f}")
    for k, v in drift_per_fpr.items():
        log.info(f"Raw AE {k}            : {v:.4f}")
    log.info(f"Raw AE best checkpoint   : {best_ckpt_path}  (epoch={best_epoch})")

    return {
        "separation_ratio": separation_ratio,
        "val_drift_metric": val_drift_metric,
        "drift_metric": drift_metric,
        **drift_per_fpr,
        "best_epoch": best_epoch,
        "per_signal": per_signal_metrics,
    }


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
    embeddings_dir: str = None,
    drift_monitor: str = "ae/val_drift_metric",
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
        embeddings_dir    : If provided, skip embedding extraction and load from this directory.
                            All embeddings must already exist (train/val/test_embeddings.npz).
                            Useful for AE ablation where the encoder is fixed.

    Returns:
        ae/separation_ratio — the Optuna objective for future optimisation.
    """
    import importlib

    L.seed_everything(seed, workers=True)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1 ─────────────────────────────────────────────────────────────────
    p1_output = output_dir / "phase1"
    os.makedirs(p1_output, exist_ok=True)

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
    if embeddings_dir is not None:
        embedding_dir = Path(embeddings_dir)
        log.info("=" * 80)
        log.info(f"EMBEDDING EXTRACTION: SKIPPED — reusing embeddings from {embedding_dir}")
        log.info("=" * 80)
        if not embedding_dir.exists():
            raise FileNotFoundError(f"--embeddings-dir not found: {embedding_dir}")
    else:
        embedding_dir = output_dir / "embeddings"
        extract_and_save_embeddings(
            model=model,
            phase2_experiment=phase2_experiment,
            output_dir=embedding_dir,
            use_projections=use_projections,
        )

    # Phase 2 ─────────────────────────────────────────────────────────────────
    p2_output = output_dir / "phase2"
    os.makedirs(p2_output, exist_ok=True)
    p2_metrics = run_phase2(
        phase2_experiment=phase2_experiment,
        embedding_dir=embedding_dir,
        output_dir=p2_output,
        extra_overrides=phase2_overrides,
        monitor=drift_monitor,
    )
    separation_ratio  = p2_metrics["separation_ratio"]
    val_drift_metric  = p2_metrics["val_drift_metric"]
    drift_metric      = p2_metrics["drift_metric"]
    drift_per_fpr     = {k: p2_metrics[k] for k in ("drift_fpr01", "drift_fpr05", "drift_fpr10")}

    # Linear probe metrics from Phase 1
    linear_probe_accuracy    = float(p1_metrics.get("linear_probe_accuracy",    float("nan")))
    linear_probe_f1_macro    = float(p1_metrics.get("linear_probe_f1_macro",    float("nan")))
    linear_probe_auroc_macro = float(p1_metrics.get("linear_probe_auroc_macro", float("nan")))
    # Per-class metrics: keys are linear_probe_f1_<process> / linear_probe_auroc_<process>
    per_class_probe = {
        k: float(v) for k, v in p1_metrics.items()
        if (k.startswith("linear_probe_f1_") or k.startswith("linear_probe_auroc_"))
        and not k.endswith("_macro")
    }

    # Summary ─────────────────────────────────────────────────────────────────
    log.info("=" * 80)
    log.info("FULL ANOMALY PIPELINE COMPLETE")
    log.info("=" * 80)
    log.info(f"  Phase 1 checkpoint        : {best_ckpt}")
    log.info(f"  val/con_loss (P1)         : {p1_metrics.get('val/con_loss', 'N/A')}")
    log.info(f"  Linear probe accuracy     : {linear_probe_accuracy:.4f}")
    log.info(f"  Linear probe F1 (macro)   : {linear_probe_f1_macro:.4f}")
    log.info(f"  Linear probe AUROC (macro): {linear_probe_auroc_macro:.4f}")
    for k, v in sorted(per_class_probe.items()):
        log.info(f"  {k:<50}: {v:.4f}")
    log.info(f"  AE separation ratio       : {separation_ratio:.4f}")
    log.info(f"  AE val drift metric (HPO) : {val_drift_metric:.4f}")
    log.info(f"  AE test drift metric      : {drift_metric:.4f}")

    summary = {
        "phase1_ckpt": str(best_ckpt),
        "embedding_dir": str(embedding_dir),
        "p1_val_con_loss": float(p1_metrics.get("val/con_loss", float("nan"))),
        "linear_probe_accuracy": linear_probe_accuracy,
        "linear_probe_f1_macro": linear_probe_f1_macro,
        "linear_probe_auroc_macro": linear_probe_auroc_macro,
        **per_class_probe,
        "ae_separation_ratio": separation_ratio,
        "ae_val_drift_metric": val_drift_metric,
        "ae_drift_metric": drift_metric,
        **{f"ae_{k}": v for k, v in drift_per_fpr.items()},
    }
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "separation_ratio": separation_ratio,
        "val_drift_metric": val_drift_metric,
        "drift_metric": drift_metric,
        **drift_per_fpr,
        "linear_probe_accuracy": linear_probe_accuracy,
        "linear_probe_f1_macro": linear_probe_f1_macro,
        "linear_probe_auroc_macro": linear_probe_auroc_macro,
        **per_class_probe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full anomaly detection pipeline: SupCon pretraining + AE"
    )
    parser.add_argument(
        "--phase1-experiment",
        default="vcreg_12class_nosparse_dmodel256_cern",
        help="Hydra experiment name for Phase 1 (contrastive encoder training)",
    )
    parser.add_argument(
        "--phase2-experiment",
        default="anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern",
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
    parser.add_argument(
        "--embeddings-dir",
        default=None,
        help="Skip embedding extraction and reuse embeddings from this directory. "
             "Useful for AE ablation where the encoder is fixed (saves time and disk).",
    )
    parser.add_argument(
        "--drift-monitor",
        default="ae/val_drift_metric",
        choices=["ae/val_drift_metric", "ae/val_drift_fpr01", "ae/val_drift_fpr05", "ae/val_drift_fpr10"],
        help="Metric to monitor for best checkpoint selection (default: average over all FPRs)",
    )
    args = parser.parse_args()

    # Ensure project root is cwd (Hydra expects this)
    os.chdir(Path(__file__).parent.parent)

    metrics = run_full_anomaly_pipeline(
        phase1_experiment=args.phase1_experiment,
        phase2_experiment=args.phase2_experiment,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        use_projections=args.use_projections,
        phase1_overrides=args.phase1_override or [],
        phase2_overrides=args.phase2_override or [],
        phase1_ckpt=args.phase1_ckpt,
        embeddings_dir=args.embeddings_dir,
        drift_monitor=args.drift_monitor,
    )

    print(f"\nFinal ae/separation_ratio      : {metrics['separation_ratio']:.4f}")
    print(f"Final ae/val_drift_metric (HPO): {metrics['val_drift_metric']:.4f}")
    print(f"Final ae/drift_metric (test)   : {metrics['drift_metric']:.4f}")

    # Save metrics to JSON for seed aggregation
    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_to_save = {**metrics, "seed": args.seed}
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
