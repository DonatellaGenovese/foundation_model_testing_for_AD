"""
Train Autoencoder for Anomaly Detection on Raw Input Features.

Trains an autoencoder on QCD (normal class) and detects Higgs variants as anomalies.

Usage:
    python src/train_anomaly_detection.py \
        experiment=anomaly_qcd_vs_higgs_raw \
        data.normal_classes=[list of your normal classes] \
        data.anomaly_classes=[list of your anomaly classes]
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import hydra
import rootutils
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, IterableDataset
from lightning import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.autoencoder import AutoencoderLitModule
from src.utils import RankedLogger, extras, instantiate_callbacks, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)


class FilteredIterableDataset(IterableDataset):
    """Filter an IterableDataset to only yield samples whose label is in keep_classes."""

    def __init__(self, base_dataset: IterableDataset, keep_classes: List[int]):
        super().__init__()
        self.base_dataset = base_dataset
        self.keep_classes = set(keep_classes)

    def __iter__(self):
        for features, label in self.base_dataset:
            if label.item() in self.keep_classes:
                yield features, label


class FilteredDataModuleWrapper(LightningDataModule):
    """
    Wrapper around a datamodule that filters datasets to only include
    specified classes (normal + anomaly classes).
    """
    
    def __init__(
        self,
        base_datamodule: LightningDataModule,
        normal_classes: List[int],
        anomaly_classes: List[int],
    ):
        super().__init__()
        self.base_dm = base_datamodule
        self.normal_classes = normal_classes
        self.anomaly_classes = anomaly_classes
        self.keep_classes = self.normal_classes + self.anomaly_classes
        
        # Copy the vlen from the original datamodule
        self.vlen = base_datamodule.vlen
    
    def _filter_dataset(self, dataset, split_name: str):
        """Filter dataset to only include normal + anomaly classes."""
        if dataset is None:
            return None

        # IterableDataset (e.g. ShuffleBuffer, LocalVectorDataset): wrap with streaming filter
        if isinstance(dataset, IterableDataset):
            log.info(f"{split_name}: Wrapping IterableDataset with class filter (keep: {self.keep_classes})")
            return FilteredIterableDataset(dataset, self.keep_classes)

        # Map-style dataset: collect labels, build index subset
        all_labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            all_labels.append(label)
        all_labels = torch.tensor(all_labels)

        mask = torch.zeros(len(all_labels), dtype=torch.bool)
        for cls in self.keep_classes:
            mask |= (all_labels == cls)

        keep_indices = torch.where(mask)[0].tolist()
        log.info(f"{split_name}: Filtered {len(dataset)} → {len(keep_indices)} samples (classes: {self.keep_classes})")

        filtered_labels = all_labels[mask]
        for cls in self.keep_classes:
            count = (filtered_labels == cls).sum().item()
            log.info(f"  Class {cls}: {count} samples")

        return Subset(dataset, keep_indices)
    
    def prepare_data(self):
        self.base_dm.prepare_data()
    
    def setup(self, stage: str = None):
        self.base_dm.setup(stage)

        # COLLIDE2V uses shuffled_train/shuffled_val/teststream (data_train etc. are None).
        # Fall back to data_train/data_val/data_test, then train_dataset/val_dataset/test_dataset.
        def _get(*attrs):
            for attr in attrs:
                v = getattr(self.base_dm, attr, None)
                if v is not None:
                    return v
            return None

        # Prefer raw streams (trainstream/valstream) over ShuffleBuffer wrappers:
        # wrapping FilteredIterableDataset around ShuffleBuffer forces reading all 18
        # classes through a 100k buffer before filtering. The raw streams already have
        # file-level shuffle (shuffle_file_order=True), which is sufficient.
        raw_train = _get("trainstream", "shuffled_train", "data_train", "train_dataset")
        raw_val   = _get("valstream",   "shuffled_val",   "data_val",   "val_dataset")
        raw_test  = _get("teststream",  "data_test",      "test_dataset")

        if raw_train is not None:
            self.train_dataset = self._filter_dataset(raw_train, "Train")
        if raw_val is not None:
            self.val_dataset   = self._filter_dataset(raw_val,   "Val")
        if raw_test is not None:
            self.test_dataset  = self._filter_dataset(raw_test,  "Test")
    
    def train_dataloader(self):
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            return None
        # IterableDataset (ShuffleBuffer) handles shuffling internally
        shuffle = not isinstance(self.train_dataset, IterableDataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.base_dm.hparams.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.base_dm.hparams.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.base_dm.hparams.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="anomaly_detection.yaml")
def main(cfg: DictConfig):
    """
    Main training loop for anomaly detection.
    
    Args:
        cfg: Hydra config with:
            - data: DataModule config
            - model: Autoencoder hyperparameters
    """
    
    # Set seed
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
        log.info(f"Set seed to {cfg.seed}")
    
    log.info("="*80)
    log.info("ANOMALY DETECTION WITH AUTOENCODER")
    log.info("="*80)
    
    # Instantiate datamodule from config 
    # We use the COLLIDE2VDataModule to construct dataset
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    base_datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # IMPORTANT: Call prepare_data() before setup() to trigger preprocessing if enabled
    log.info("Preparing data (preprocessing if needed)...")
    base_datamodule.prepare_data()
        
    # Wrap datamodule with filtering to only include normal + anomaly classes
    log.info(f"Filtering to: normal_classes={cfg.normal_classes}, anomaly_classes={cfg.get('anomaly_classes', [])}")
    datamodule = FilteredDataModuleWrapper(
        base_datamodule=base_datamodule,
        normal_classes=cfg.normal_classes,
        anomaly_classes=cfg.get("anomaly_classes", []),
    )
    datamodule.setup('fit')
    
    # Get input dimension from raw features
    input_dim = datamodule.vlen
    log.info(f"Input dimension (raw features): {input_dim}")
    
    # Create autoencoder
    log.info("Creating autoencoder...")
    ae_model = AutoencoderLitModule(
        input_dim=input_dim,
        compression=cfg.model.compression,
        depth=cfg.model.depth,
        dropout=cfg.model.dropout,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        normal_classes_labels=cfg.normal_classes,
        anomaly_classes_labels=cfg.get("anomaly_classes", None),
    )

    log.info(f"Autoencoder architecture:")
    log.info(f"  Input dim: {input_dim}")
    log.info(f"  Bottleneck dim: {input_dim // cfg.model.compression}")
    log.info(f"  Depth: {cfg.model.depth}")
    log.info(f"  Training on normal classes: {cfg.normal_classes}")
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.paths.output_dir) / "checkpoints",
        filename="ae-epoch{epoch:02d}",
        monitor="ae/val_loss_normal",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="ae/val_loss_normal",
        patience=10,
        mode="min",
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
        deterministic= bool(cfg.get("seed"))
    )
    
    # Train
    log.info("Starting training (on normal class only)...")
    trainer.fit(ae_model, datamodule=datamodule)
    
    # Test - setup test split
    datamodule.setup('test')
    
    log.info("Testing on filtered dataset (normal + anomalies only)...")
    # Use weights_only=False for PyTorch 2.6+ compatibility with OmegaConf in checkpoints
    trainer.test(ae_model, datamodule=datamodule, ckpt_path="best", weights_only=False)
    
    # Plot results
    output_dir = Path(cfg.paths.output_dir) / "plots"
    log.info(f"Saving plots to: {output_dir}")
    ae_model.plot_reconstruction_errors(output_dir)
    
    log.info("="*80)
    log.info("ANOMALY DETECTION TRAINING COMPLETE")
    log.info("="*80)
    
    # Print summary
    log.info("\nSummary:")
    log.info(f"  Input dimension: {input_dim}")
    log.info(f"  Best checkpoint: {checkpoint_callback.best_model_path}")
    log.info(f"  Best Separation Ratio: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
