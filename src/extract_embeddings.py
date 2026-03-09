"""
Extract embeddings from a trained encoder for anomaly detection.

This script loads a pretrained encoder checkpoint (supports all model types)
and extracts embeddings for all data splits (train/val/test). The embeddings 
are saved to disk for use in downstream tasks like training an autoencoder 
for anomaly detection.

Supported Models:
- COLLIDE2VVanillaSupConLitModule (SupCon with projection head)
- COLLIDE2VAugmentedSupConLitModule (SupCon with augmentation)
- COLLIDE2VTransformerLitModule (standard classifier)
- COLLIDE2VTinyMLPLitModule (MLP classifier)

Usage:
    python src/extract_embeddings.py \
        ckpt_path=/path/to/checkpoint.ckpt \
        experiment=anomaly_qcd_vs_higgs_raw \
        output_dir=/path/to/save/embeddings \
        use_projections=true  # For SupCon models: use projection head or encoder embeddings
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import os
import importlib

import hydra
import rootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from lightning import LightningModule, LightningDataModule, seed_everything
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@torch.no_grad()
def extract_embeddings_from_loader(
    model: LightningModule,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    use_projections: bool = False,
    desc: str = "Extracting"
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for all batches in a dataloader.
    
    Supports multiple model types:
    - SupCon models: Can extract from encoder or projection head
    - Transformer/MLP models: Extract from encoder only
    
    Args:
        model: Pretrained LightningModule (any model with get_embeddings method)
        dataloader: DataLoader to extract from
        device: Device to run on
        use_projections: If True and model has projection_head, use projections;
                        otherwise use encoder embeddings
        desc: Description for progress bar
        
    Returns:
        Dictionary with 'embeddings', 'labels' as numpy arrays
    """
    model.eval()

    # Check what the model supports
    has_projection = hasattr(model, 'projection_head') and model.projection_head is not None
    has_encoder = hasattr(model, 'encoder') and model.encoder is not None
    has_get_embeddings = hasattr(model, 'get_embeddings')
    
    # Determine extraction strategy
    if use_projections and has_projection:
        log.info("Using projection head for embeddings (normalized projections)")
        extract_fn = lambda x: F.normalize(model.projection_head(model.encoder.get_embeddings(x)), dim=-1, p=2)
    elif has_encoder:
        log.info("Using encoder embeddings (without projection)")
        extract_fn = lambda x: model.encoder.get_embeddings(x)
    elif has_get_embeddings:
        log.info("Using model's get_embeddings method")
        extract_fn = lambda x: model.get_embeddings(x)
    else:
        raise RuntimeError(
            "Model does not support embedding extraction. "
            "Model must have either 'encoder.get_embeddings', 'projection_head', or 'get_embeddings' method."
        )
    
    all_embeddings = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc=desc):
        x = batch[0].to(device)  # Input features
        y = batch[1]  # Labels (kept on CPU)
        
        # Extract embeddings using the appropriate method
        embeddings = extract_fn(x)
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(y.numpy())
    
    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    log.info(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    return {
        'embeddings': embeddings,
        'labels': labels,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="anomaly_detection.yaml")
def main(cfg: DictConfig):
    """
    Main entry point for embedding extraction.
    
    Args:
        cfg: Hydra config with:
            - ckpt_path: Path to encoder checkpoint
            - data: DataModule config
            - embedding_output_dir: Where to save embeddings
            - use_projections: Whether to use projection head (SupCon models only)
    """
    
    # Set seed
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
        log.info(f"Set seed to {cfg.seed}")
    
    log.info("="*80)
    log.info("EMBEDDING EXTRACTION FOR ANOMALY DETECTION")
    log.info("="*80)
    
    # Check checkpoint path
    if not cfg.get("ckpt_path"):
        raise ValueError("Must provide ckpt_path in config or command line")
    
    ckpt_path = cfg.ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Setup output directory
    output_dir = Path(cfg.get("embedding_output_dir", "embeddings"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Embeddings will be saved to: {output_dir}")
    
    # Load device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")
    
    # Instantiate datamodule FIRST (needed for model setup)
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Prepare data
    log.info("Preparing data (preprocessing if needed)...")
    datamodule.prepare_data()
    datamodule.setup('fit')
    
    # Load model checkpoint - import class from config
    log.info(f"Loading checkpoint from: {ckpt_path}")
    
    # Get model class from config
    if not cfg.get("model_class"):
        raise ValueError(
            "Must provide model_class in config (e.g., model_class='src.models.collide2v_vanillasupcon.COLLIDE2VVanillaSupConLitModule')"
        )
    
    model_class_path = cfg.model_class
    log.info(f"Importing model class: {model_class_path}")
    
    # Dynamic import: split module path and class name
    module_path, class_name = model_class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)
    
    log.info(f"Loading {class_name} from checkpoint...")
    model = ModelClass.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        weights_only=False
    )
    model.eval()
    model.to(device)
    
    # Detect model type and capabilities
    model_type = model.__class__.__name__
    has_projection = hasattr(model, 'projection_head') and model.projection_head is not None
    has_encoder = hasattr(model, 'encoder') and model.encoder is not None
    has_get_embeddings = hasattr(model, 'get_embeddings')
    
    log.info(f"Model type: {model_type}")
    log.info(f"Has encoder: {has_encoder}")
    log.info(f"Has projection_head: {has_projection}")
    log.info(f"Has get_embeddings: {has_get_embeddings}")
    
    # Determine embedding dimension
    use_projections = cfg.get("use_projections", True) and has_projection
    
    if use_projections:
        embedding_dim = model.hparams.projection_dim
        log.info(f"Using projection embeddings (dim: {embedding_dim})")
    elif has_encoder and hasattr(model.hparams, 'd_model'):
        embedding_dim = model.hparams.d_model
        log.info(f"Using encoder embeddings (dim: {embedding_dim})")
    else:
        x_sample = next(iter(datamodule.train_dataloader()))[0][:1].to(device)
        with torch.no_grad():
          if use_projections:
            emb = F.normalize(model.projection_head(model.encoder.get_embeddings(x_sample)), dim=-1)
          else:
            emb = model.encoder.get_embeddings(x_sample)
        embedding_dim = emb.shape[1]
    
    # Extract embeddings for each split
    splits = {
        'train': datamodule.train_dataloader(),
        'val': datamodule.val_dataloader(),
    }
    
    # Also get test dataloader if available
    try:
        datamodule.setup('test')
        splits['test'] = datamodule.test_dataloader()
    except Exception as e:
        log.warning(f"Test split not available: {e}")
    
    embeddings_data = {}
    
    for split_name, dataloader in splits.items():
        log.info(f"\nExtracting {split_name} embeddings...")
        
        data = extract_embeddings_from_loader(
            model=model,
            dataloader=dataloader,
            device=device,
            use_projections=use_projections,
            desc=f"Extracting {split_name}"
        )
        
        embeddings_data[split_name] = data
        
        # Save to disk
        split_file = output_dir / f"{split_name}_embeddings.npz"
        np.savez_compressed(
            split_file,
            embeddings=data['embeddings'],
            labels=data['labels'],
        )
        log.info(f"✅ Saved {split_name} embeddings to: {split_file}")
        
        # Print statistics
        unique_labels, counts = np.unique(data['labels'], return_counts=True)
        log.info(f"  Classes in {split_name}: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            log.info(f"    Class {label}: {count} samples")
    
    # Save metadata
    metadata = {
        'ckpt_path': str(ckpt_path),
        'embedding_dim': int(embedding_dim),
        'model_type': model_type,
        'use_projections': use_projections,
        'num_splits': len(embeddings_data),
        'splits': list(embeddings_data.keys()),
    }
    
    metadata_file = output_dir / "metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    log.info(f"✅ Saved metadata to: {metadata_file}")
    
    log.info("="*80)
    log.info("EMBEDDING EXTRACTION COMPLETE")
    log.info("="*80)
    log.info(f"\nAll embeddings saved to: {output_dir}")
    log.info(f"Model type: {model_type}")
    log.info(f"Embedding dimension: {embedding_dim}")
    log.info(f"Total samples extracted: {sum(len(d['embeddings']) for d in embeddings_data.values())}")


if __name__ == "__main__":
    main()
