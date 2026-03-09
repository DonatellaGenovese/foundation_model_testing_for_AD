"""
Train Autoencoder for Anomaly Detection on Pretrained Embeddings.

Extracts embeddings from a pretrained encoder, then trains an autoencoder
on QCD embeddings only to detect Higgs variants as anomalies.

Usage:
    python src/train_anomaly_embedding.py \
        experiment=anomaly_qcd_vs_higgs_embedding \
        ckpt_path=/path/to/encoder.ckpt
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os
import subprocess
import sys

import hydra
import rootutils
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from lightning import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.autoencoder import AutoencoderLitModule
from src.models.collide2v_vanillasupcon import COLLIDE2VVanillaSupConLitModule
from src.utils import RankedLogger, extras, instantiate_callbacks, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)


class EmbeddingDataModule(LightningDataModule):
    """Simple DataModule for loading precomputed embeddings."""
    
    def __init__(
        self,
        embedding_dir: str,
        batch_size: int = 512,
        num_workers: int = 0,
        normal_classes: List[int] = [0],
        anomaly_classes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.embedding_dir = Path(embedding_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normal_classes = normal_classes
        self.anomaly_classes = anomaly_classes if anomaly_classes else []
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.embedding_dim = None
    
    def _filter_dataset(self, embeddings: torch.Tensor, labels: torch.Tensor, split_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter dataset to only include normal + anomaly classes."""
        # Keep only samples from normal class or anomaly classes
        keep_classes = self.normal_classes + self.anomaly_classes
        mask = torch.zeros(len(labels), dtype=torch.bool)
        for cls in keep_classes:
            mask |= (labels == cls)
        
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        
        log.info(f"{split_name}: Filtered {len(embeddings)} → {len(filtered_embeddings)} samples (classes: {keep_classes})")
        
        # Count samples per class
        for cls in keep_classes:
            count = (filtered_labels == cls).sum().item()
            log.info(f"  Class {cls}: {count} samples")
        
        return filtered_embeddings, filtered_labels
    
    def setup(self, stage: str = None):
        """Load embeddings from disk."""
        
        # Load train embeddings
        train_file = self.embedding_dir / "train_embeddings.npz"
        if train_file.exists():
            train_data = np.load(train_file)
            train_embeddings = torch.from_numpy(train_data['embeddings']).float()
            train_labels = torch.from_numpy(train_data['labels']).long()
            
            # TRAIN: normal class only
            train_mask = torch.zeros(len(train_labels), dtype=torch.bool)
            for cls in self.normal_classes:
                train_mask |= (train_labels == cls)
            self.train_dataset = TensorDataset(train_embeddings[train_mask], train_labels[train_mask])
            self.embedding_dim = train_embeddings.shape[1]
        else:
            raise FileNotFoundError(f"Train embeddings not found: {train_file}")
        
        # Load val embeddings
        val_file = self.embedding_dir / "val_embeddings.npz"
        if val_file.exists():
            val_data = np.load(val_file)
            val_embeddings = torch.from_numpy(val_data['embeddings']).float()
            val_labels = torch.from_numpy(val_data['labels']).long()
            
            # Filter to only normal + anomaly classes
            val_embeddings, val_labels = self._filter_dataset(val_embeddings, val_labels, "Val")
            
            self.val_dataset = TensorDataset(val_embeddings, val_labels)
            log.info(f"Loaded val embeddings: {val_embeddings.shape}")
        else:
            log.warning(f"Val embeddings not found: {val_file}")
        
        # Load test embeddings
        test_file = self.embedding_dir / "test_embeddings.npz"
        if test_file.exists():
            test_data = np.load(test_file)
            test_embeddings = torch.from_numpy(test_data['embeddings']).float()
            test_labels = torch.from_numpy(test_data['labels']).long()
            
            # Filter to only normal + anomaly classes
            test_embeddings, test_labels = self._filter_dataset(test_embeddings, test_labels, "Test")
            
            self.test_dataset = TensorDataset(test_embeddings, test_labels)
            log.info(f"Loaded test embeddings: {test_embeddings.shape}")
        else:
            log.warning(f"Test embeddings not found: {test_file}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def extract_embeddings_if_needed(cfg: DictConfig) -> Path:
    """
    Extract embeddings using the separate extraction script if not already present.
    
    Args:
        cfg: Hydra config
        
    Returns:
        Path to embedding directory
    """
    embedding_dir = Path(cfg.embedding_output_dir)
    metadata_file = embedding_dir / "metadata.json"
    
    # Check if embeddings already exist
    if metadata_file.exists():
        log.info(f"✅  Found existing embeddings in: {embedding_dir}")
        return embedding_dir
    
    # Extract embeddings
    log.info("❌ Embeddings not found. Extracting from encoder checkpoint...")
    log.info(f"Checkpoint: {cfg.ckpt_path}")
    
    # Run extraction script
    cmd = [
        sys.executable,
        "src/extract_embeddings.py",
        f"experiment={cfg.experiment_name if hasattr(cfg, 'experiment_name') else 'anomaly_qcd_vs_higgs_embedding'}",
        f"ckpt_path={cfg.ckpt_path}",
        f"embedding_output_dir={cfg.embedding_output_dir}",
        f"seed={cfg.get('seed', 42)}",
    ]
    
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if not metadata_file.exists():
        raise RuntimeError("Embedding extraction failed!")
    
    log.info(f"✅ Embeddings extracted successfully to: {embedding_dir}")
    return embedding_dir


@hydra.main(version_base="1.3", config_path="../configs", config_name="anomaly_detection.yaml")
def main(cfg: DictConfig):
    """
    Main training loop for embedding-based anomaly detection.
    
    Args:
        cfg: Hydra config with:
            - ckpt_path: Path to encoder checkpoint
            - embedding_output_dir: Where to save/load embeddings
            - model: Autoencoder hyperparameters
    """
    
    # Set seed
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
        log.info(f"Set seed to {cfg.seed}")
    
    log.info("="*80)
    log.info("EMBEDDING-BASED ANOMALY DETECTION WITH AUTOENCODER")
    log.info("="*80)
    
    # Check checkpoint path
    if not cfg.get("ckpt_path"):
        raise ValueError("Must provide ckpt_path for encoder checkpoint")
    
    if not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")
    
    log.info(f"Encoder checkpoint: {cfg.ckpt_path}")
    
    # Extract embeddings if needed
    embedding_dir = extract_embeddings_if_needed(cfg)
    
    # Create embedding datamodule
    log.info("Loading embeddings...")
    log.info(f"Filtering to: normal_classes={cfg.normal_classes}, anomaly_classes={cfg.get('anomaly_classes', [])}")
    embedding_datamodule = EmbeddingDataModule(
        embedding_dir=str(embedding_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.get("num_workers", 0),
        normal_classes=cfg.normal_classes,
        anomaly_classes=cfg.get("anomaly_classes", []),
    )
    
    # Get embedding dimension
    input_dim = embedding_datamodule.embedding_dim
    log.info(f"Embedding dimension: {input_dim}")
    
    # Create autoencoder
    log.info("Creating autoencoder for embeddings...")
    ae_model = AutoencoderLitModule(
        input_dim=input_dim,
        bottleneck_dim=cfg.model.bottleneck_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        normal_classes=cfg.normal_classes,
        anomaly_classes=cfg.get("anomaly_classes", None),
    )
    
    log.info(f"Autoencoder architecture:")
    log.info(f"  Input dim (embedding): {input_dim}")
    log.info(f"  Bottleneck dim: {cfg.model.bottleneck_dim}")
    log.info(f"  Hidden dims: {cfg.model.hidden_dims}")
    log.info(f"  Training on normal class: {cfg.normal_classes}")
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.paths.output_dir) / "checkpoints",
        filename="ae_emb-{epoch:02d}-{ae/separation_ratio:.4f}",
        monitor="ae/separation_ratio",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="ae/separation_ratio",
        patience=20,  # More patience for embedding training
        mode="max",
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    
    # Logger
    loggers = instantiate_loggers(cfg.get("logger"))
    
    # Trainer
    log.info("Instantiating trainer...")
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator='auto',
        devices=1,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        deterministic=True if cfg.get("seed") else False,
    )
    
    # Train
    log.info("Starting training on embeddings (normal class only)...")
    trainer.fit(ae_model, datamodule=embedding_datamodule)
    
    # Test
    log.info("Testing on full dataset (normal + anomalies)...")
    trainer.test(ae_model, datamodule=embedding_datamodule, ckpt_path="best", weights_only=False)
    
    # Plot results
    output_dir = Path(cfg.paths.output_dir) / "plots"
    log.info(f"Saving plots to: {output_dir}")
    ae_model.plot_reconstruction_errors(output_dir)
    
    log.info("="*80)
    log.info("EMBEDDING-BASED ANOMALY DETECTION COMPLETE")
    log.info("="*80)
    
    # Print summary
    log.info("\nSummary:")
    log.info(f"  Encoder checkpoint: {cfg.ckpt_path}")
    log.info(f"  Embedding dimension: {input_dim}")
    log.info(f"  Best AE checkpoint: {checkpoint_callback.best_model_path}")
    log.info(f"  Best Separation Ratio: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
