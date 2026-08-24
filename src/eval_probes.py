"""
Evaluate a pretrained checkpoint with a Linear Probe.

This script loads a pretrained contrastive model and evaluates the quality
of learned embeddings using:
1. Linear Probe: Trainable linear classifier on frozen embeddings

Usage:
    python src/eval_probes.py ckpt_path=/path/to/checkpoint.ckpt
    python src/eval_probes.py experiment=vcreg_12class_nosparse_dmodel256_cern ckpt_path=/path/to/checkpoint.ckpt
"""

from typing import Any, Dict, List, Tuple, Union
from pathlib import Path
import json
import math

import hydra
import rootutils
import torch
import numpy as np
from torch.utils.data import DataLoader
from lightning import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.probes import LinearProbe
from src.models.collide2v_augmented_supcon import COLLIDE2VAugmentedSupConLitModule
from src.models.collide2v_augmented_selfsupcon import COLLIDE2VAugmentedSelfSupConLitModule
from src.models.collide2v_vicreg import (
    COLLIDE2VVICRegLitModule,
    COLLIDE2VVCRegLitModule,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def extract_embeddings(
    encoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    max_samples: int = None,
    stratified: bool = True,
    num_classes: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from encoder for all samples in dataloader.
    
    Args:
        encoder: Trained encoder network
        dataloader: DataLoader to extract embeddings from
        device: Device to run on
        max_samples: Maximum number of samples to extract (None = all)
        stratified: If True, sample evenly from all classes
        num_classes: Number of classes (required if stratified=True)
    
    Returns:
        embeddings: [N, d_model] numpy array
        labels: [N] numpy array of class labels
    """
    encoder.eval()
    encoder.to(device)
    
    # Force complete determinism for reproducible embeddings
    with torch.no_grad():
        # Disable dropout completely (belt and suspenders approach)
        for module in encoder.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
                module.p = 0.0
        
        # Verify model state
        if hasattr(encoder, 'training'):
            assert not encoder.training, f"Encoder not in eval mode! training={encoder.training}"
    
    all_embeddings = []
    all_labels = []
    
    total_samples = 0
    
    # If stratified sampling, collect all data first, then sample
    if stratified and max_samples is not None and num_classes is not None:
        log.info(f"Using stratified sampling: {max_samples // num_classes} samples per class")
        
        # Use local random generator for reproducibility (independent of global state)
        rng = np.random.RandomState(42)
        
        # Collect all embeddings and labels
        temp_embeddings = []
        temp_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                features = features.to(device)
                embeddings = encoder.get_embeddings(features)
                temp_embeddings.append(embeddings.cpu().numpy())
                temp_labels.append(labels.cpu().numpy())
        
        all_embeddings_full = np.concatenate(temp_embeddings, axis=0)
        all_labels_full = np.concatenate(temp_labels, axis=0)
        
        # Stratified sampling
        samples_per_class = max_samples // num_classes
        selected_indices = []
        
        log.info(f"Sampling from each class:")
        for class_idx in range(num_classes):
            class_mask = all_labels_full == class_idx
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) > 0:
                # Sample up to samples_per_class from this class
                n_to_sample = min(samples_per_class, len(class_indices))
                sampled = rng.choice(class_indices, size=n_to_sample, replace=False)
                selected_indices.extend(sampled)
                log.info(f"  Class {class_idx}: {n_to_sample} samples (from {len(class_indices)} available)")
            else:
                log.warning(f"  Class {class_idx}: No samples found!")
        
        selected_indices = np.array(selected_indices)
        embeddings = all_embeddings_full[selected_indices]
        labels = all_labels_full[selected_indices]
        
        log.info(f"Stratified sampling complete: {len(embeddings)} total samples")
        
    else:
        # Original sequential sampling
        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                features = features.to(device)
                
                # Get embeddings
                embeddings = encoder.get_embeddings(features)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                total_samples += len(labels)
                
                if max_samples is not None and total_samples >= max_samples:
                    break
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        if max_samples is not None:
            embeddings = embeddings[:max_samples]
            labels = labels[:max_samples]
    
    return embeddings, labels


def visualize_embeddings_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "UMAP Projection of Embeddings",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
):
    """
    Create UMAP visualization of embeddings colored by class.
    
    Args:
        embeddings: [N, d_model] array of embeddings
        labels: [N] array of class labels
        class_names: List of class names
        output_path: Path to save figure
        title: Plot title
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric for UMAP
    """
    try:
        import umap
    except ImportError:
        log.error("UMAP not installed. Install with: pip install umap-learn")
        return
    
    log.info(f"Computing UMAP projection with {len(embeddings)} samples...")
    
    # Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        n_components=2,
    )
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each class
    unique_labels = np.unique(labels)
    
    # Use a colormap that supports more than 10 classes
    if len(unique_labels) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    elif len(unique_labels) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    else:
        # For more than 20 classes, use a continuous colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    log.info(f"Plotting {len(unique_labels)} classes:")
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        n_samples = mask.sum()
        # Convert numpy int64 to Python int for OmegaConf ListConfig indexing
        label_idx = int(label)
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
        
        log.info(f"  {class_name:30s}: {n_samples:6d} samples")
        
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[idx]],
            label=class_name,
            alpha=0.6,
            s=10,
            edgecolors='none'
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Use multiple columns for legend if many classes
    ncol = 1 if len(unique_labels) <= 8 else 2
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=ncol)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    log.info(f"UMAP visualization saved to: {output_path}")

    # ── Per-class subplot grid ────────────────────────────────────────────────
    # Each subplot shows all points in grey and highlights one class in color.
    n_classes = len(unique_labels)
    ncols = math.ceil(math.sqrt(n_classes))
    nrows = math.ceil(n_classes / ncols)

    fig2, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).flatten()

    for idx, label in enumerate(unique_labels):
        ax = axes[idx]
        mask = labels == label
        label_idx = int(label)
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"

        # All other points in light grey
        ax.scatter(
            embedding_2d[~mask, 0], embedding_2d[~mask, 1],
            c='lightgrey', alpha=0.3, s=4, edgecolors='none'
        )
        # Highlighted class in color
        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=[colors[idx]], alpha=0.7, s=6, edgecolors='none', label=class_name
        )
        ax.set_title(class_name, fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for ax in axes[n_classes:]:
        ax.set_visible(False)

    fig2.suptitle(f"{title} — per class", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    subplots_path = output_path.with_name(output_path.stem + "_subplots.png")
    plt.savefig(subplots_path, dpi=300, bbox_inches='tight')
    plt.close()

    log.info(f"UMAP per-class subplots saved to: {subplots_path}")


def evaluate_with_probes(
    cfg: DictConfig,
    model: Union[
        COLLIDE2VAugmentedSupConLitModule,
        COLLIDE2VAugmentedSelfSupConLitModule,
        COLLIDE2VVICRegLitModule,
        COLLIDE2VVCRegLitModule,
    ],
    datamodule: LightningDataModule,
    call_source: str = 'standalone',
) -> Dict[str, float]:
    """
    Evaluate trained model with a Linear Probe.
    
    Args:
        cfg: Hydra configuration
        model: Trained contrastive model
        datamodule: DataModule for data loading
        
    Returns:
        Dictionary with probe evaluation results
    """
    
    # CRITICAL: Force deterministic behavior for reproducible probe evaluation
    import os
    if hasattr(cfg, 'seed') and cfg.seed is not None:
        # Set environment variables for deterministic CUDA operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(cfg.seed)
        
        # Enable PyTorch deterministic mode
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        log.info(f"Enforced deterministic mode for probe evaluation (seed={cfg.seed})")
    
    log.info("\n" + "="*80)
    log.info(f"STARTING PROBE EVALUATION (called from: {call_source})")
    log.info("="*80)
    
    # Set model to eval mode
    model.eval()
    
    # Ensure datamodule is set up
    if not hasattr(datamodule, 'train_dataloader') or datamodule.train_dataloader is None:
        datamodule.setup('fit')

    # Get encoder (should be already loaded from checkpoint via on_load_checkpoint hook)
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError(
            "Encoder not found in model. This should not happen if the checkpoint "
            "was loaded correctly. Make sure the checkpoint was saved after training "
            "and contains the encoder weights."
        )
    
    # Move model to device after setup (ensures all components are on the same device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    log.info(f"Model moved to device: {device}")
    
    # Get eval config (with defaults if not present)
    eval_cfg = cfg.get("eval", None)
    if eval_cfg is None:
        log.warning("No eval config found! Using defaults.")
        eval_cfg = OmegaConf.create({
            "linear_probe": {"max_epochs": 50, "lr": 0.001, "weight_decay": 0.0},

            "output_dir": Path(cfg.paths.output_dir) / "probe_evaluation"
        })
    
    # Create output directory
    output_dir = Path(eval_cfg.get("output_dir", "./probe_evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # A `debug_comparison/` dump used to be written here: first test batch plus
    # datamodule metadata, prefixed train_path/eval_path so the two entry points could
    # be diffed by hand. Removed — nothing read it, it landed in the repo root (the
    # path was relative to cwd, and the wrapper cd's to PROJECT_DIR), concurrent probe
    # jobs raced on the same filenames, and it cost an extra dataloader pass over the
    # test split on every run.

    results = {}
    
    # Get class names for visualizations
    class_names = None
    if hasattr(datamodule, 'class_names'):
        class_names = datamodule.class_names
    elif hasattr(cfg.data, 'to_classify'):
        class_names = cfg.data.to_classify
    
    # ========================================
    # 0. CACHE TRAINING DATA ONCE FOR DETERMINISTIC PROBE TRAINING
    # ========================================
    log.info("\n" + "="*80)
    log.info("0. CACHING TRAINING DATA FOR DETERMINISTIC PROBE TRAINING")
    log.info("="*80)
    
    # CRITICAL: Cache training data ONCE at the beginning with proper seeding
    # This ensures all subsequent probe training uses the exact same data
    cached_train_data = None
    if hasattr(cfg, 'seed') and cfg.seed is not None:
        # Seed everything before getting training data
        seed_everything(cfg.seed, workers=True)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        
        log.info(f"Seeding with {cfg.seed} before caching training data...")
        
        # Get max training samples for probe (optional limit)
        max_probe_train_samples = eval_cfg.get("max_probe_train_samples", None)
        
        # Get the train dataloader and cache ALL data in deterministic order
        train_loader = datamodule.train_dataloader()
        all_features = []
        all_labels = []
        sample_count = 0
        
        log.info("Extracting training data...")
        for batch_idx, batch in enumerate(train_loader):
            features, labels = batch
            all_features.append(features)
            all_labels.append(labels)
            sample_count += len(labels)
            
            # Limit samples if specified
            if max_probe_train_samples and sample_count >= max_probe_train_samples:
                break
            
            # Log progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                log.info(f"  Cached {sample_count} samples...")
        
        # Concatenate into single tensors
        cached_train_data = (
            torch.cat(all_features, dim=0),
            torch.cat(all_labels, dim=0)
        )
        
        # Truncate if needed
        if max_probe_train_samples and len(cached_train_data[1]) > max_probe_train_samples:
            cached_train_data = (
                cached_train_data[0][:max_probe_train_samples],
                cached_train_data[1][:max_probe_train_samples]
            )
        
        log.info(f"✓ Cached {len(cached_train_data[1])} training samples for probe evaluation")
        log.info(f"  Features shape: {cached_train_data[0].shape}")
        log.info(f"  Labels shape: {cached_train_data[1].shape}")
        log.info(f"  Unique labels: {torch.unique(cached_train_data[1]).tolist()}")
        log.info(f"  First 20 labels: {cached_train_data[1][:20].tolist()}")
    
    # ========================================
    # 1. EMBEDDING VISUALIZATION - BEFORE PROBE TRAINING (if enabled)
    # ========================================

    if eval_cfg.get("save_visualizations", False):
        log.info("\n" + "="*80)
        log.info("1. GENERATING EMBEDDING VISUALIZATIONS (BEFORE PROBE)")
        log.info("="*80)
        
        viz_output_dir = output_dir / "visualizations"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract embeddings from test set
        max_viz_samples = eval_cfg.get("max_visualization_samples", 10000)
        log.info(f"Extracting embeddings from test set (max {max_viz_samples} samples)...")
        
        test_embeddings, test_labels = extract_embeddings(
            encoder=encoder,
            dataloader=datamodule.test_dataloader(),
            device=device,
            max_samples=max_viz_samples,
            stratified=True,
            num_classes=datamodule.num_classes,
        )
        
        log.info(f"Extracted {len(test_embeddings)} embeddings with shape {test_embeddings.shape}")
        
        # DIAGNOSTIC: Log detailed embedding statistics
        log.info("="*80)
        log.info("EMBEDDING STATISTICS:")
        log.info(f"  Shape: {test_embeddings.shape}")
        log.info(f"  Mean: {test_embeddings.mean():.6f}")
        log.info(f"  Std: {test_embeddings.std():.6f}")
        log.info(f"  Min: {test_embeddings.min():.6f}")
        log.info(f"  Max: {test_embeddings.max():.6f}")
        log.info(f"  First 5 sample norms: {np.linalg.norm(test_embeddings[:5], axis=1)}")
        log.info(f"  Label distribution: {np.bincount(test_labels)}")
        log.info(f"  First 20 labels: {test_labels[:20]}")
        log.info("="*80)
        
        # DEBUG: Save embeddings for reproducibility analysis
        embeddings_debug_path = viz_output_dir / "embeddings_debug.npz"
        np.savez(
            embeddings_debug_path,
            embeddings=test_embeddings,
            labels=test_labels
        )
        log.info(f"[DEBUG] Saved embeddings to: {embeddings_debug_path}")
        # A second copy of these same embeddings went to debug_comparison/ in the repo
        # root; removed with the rest of that dump. This one, in the job's output dir,
        # is the copy that is kept.

        # Create UMAP visualization on embeddings
        visualize_embeddings_umap(
            embeddings=test_embeddings,
            labels=test_labels,
            class_names=class_names if class_names else [f"Class {i}" for i in range(datamodule.num_classes)],
            output_path=viz_output_dir / "umap_embeddings_before_probe.png",
            title="UMAP Projection of Encoder Embeddings (Before Probe)",
            n_neighbors=eval_cfg.get("umap_n_neighbors", 15),
            min_dist=eval_cfg.get("umap_min_dist", 0.1),
            metric=eval_cfg.get("umap_metric", "cosine"),
        )
        
        log.info("Embedding visualizations (before probe) complete!")
    
    # ========================================
    # 2. LINEAR PROBE EVALUATION
    # ========================================
    log.info("\n" + "="*80)
    log.info("2. LINEAR PROBE EVALUATION")
    log.info("="*80)
    
    # Get class names
    class_names = None
    if hasattr(datamodule, 'class_names'):
        class_names = datamodule.class_names
    elif hasattr(cfg.data, 'to_classify'):
        class_names = cfg.data.to_classify
    
    # Ensure deterministic behavior for probe training
    # CRITICAL: Seed BEFORE creating LinearProbe and re-setup datamodule
    if hasattr(cfg, 'seed') and cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)
        # Also explicitly seed torch for the Linear layer initialization
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        log.info(f"Seeded before probe creation: {cfg.seed}")
        
        # IMPORTANT: Re-setup datamodule to ensure ShuffleBuffer gets correct seed
        # The ShuffleBuffer uses torch.initial_seed() which must be consistent
        log.info("Re-setting up datamodule for deterministic probe training...")
        datamodule.setup('fit')
    
    linear_probe = LinearProbe(
        encoder=encoder,
        embedding_dim=model.hparams.d_model,
        num_classes=datamodule.num_classes,
        class_names=class_names,
        lr=eval_cfg.linear_probe.lr,
        weight_decay=eval_cfg.linear_probe.weight_decay,
    )
    
    # CRITICAL: Manually initialize classifier weights deterministically
    # This ensures the probe starts with identical weights every time
    if hasattr(cfg, 'seed') and cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        # Reinitialize the classifier layer
        torch.nn.init.xavier_uniform_(linear_probe.classifier.weight, gain=1.0)
        torch.nn.init.zeros_(linear_probe.classifier.bias)
        log.info(f"Reinitialized classifier weights with seed={cfg.seed}")
    
    # Logger for linear probe
    linear_logger = TensorBoardLogger(
        save_dir=str(output_dir / "linear_probe"),
        name="",
        version="",
    )
    
    # Force single-threaded dataloading for probe training (ensures determinism)
    # Save original value to restore later
    original_num_workers = datamodule.num_workers if hasattr(datamodule, 'num_workers') else 0
    if hasattr(datamodule, 'num_workers'):
        datamodule.num_workers = 0
        log.info(f"Temporarily set num_workers=0 for deterministic probe training (was {original_num_workers})")
    
    # CRITICAL FIX: Create deterministic wrapper around datamodule
    # The issue is that streaming datasets (trainstream) don't provide deterministic iteration
    # Solution: Use pre-cached training data that was extracted once at the beginning
    class DeterministicDataModule(LightningDataModule):
        """Wrapper that provides deterministic dataloaders using pre-cached training data"""
        def __init__(self, base_dm, cached_train_data):
            super().__init__()
            self._base_dm = base_dm
            self._cached_train_data = cached_train_data
            
        def __getattr__(self, name):
            # Delegate attribute access to base datamodule
            # Only for attributes not in this class
            if name.startswith('_'):
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            return getattr(self._base_dm, name)
        
        def prepare_data(self):
            # Delegate to base
            if hasattr(self._base_dm, 'prepare_data'):
                self._base_dm.prepare_data()
        
        def setup(self, stage=None):
            # Delegate to base
            if hasattr(self._base_dm, 'setup'):
                self._base_dm.setup(stage)
        
        def train_dataloader(self):
            # Use the pre-cached training data
            if self._cached_train_data is None:
                raise RuntimeError("No cached training data provided! This should not happen.")
            
            log.info(f"Using pre-cached training data: {len(self._cached_train_data[1])} samples")
            
            # Create TensorDataset from cached data
            from torch.utils.data import TensorDataset
            dataset = TensorDataset(self._cached_train_data[0], self._cached_train_data[1])
            
            # Create deterministic dataloader with fixed order
            return DataLoader(
                dataset=dataset,
                batch_size=self._base_dm.batch_size_per_device,
                shuffle=False,  # CRITICAL: No shuffling for determinism
                num_workers=0,  # Single-threaded for determinism
                pin_memory=False,
                drop_last=False,
            )
        
        def val_dataloader(self):
            return self._base_dm.val_dataloader()
        
        def test_dataloader(self):
            return self._base_dm.test_dataloader()
    
    # Wrap datamodule for deterministic probe training
    if hasattr(cfg, 'seed') and cfg.seed is not None and cached_train_data is not None:
        deterministic_dm = DeterministicDataModule(datamodule, cached_train_data)
        log.info(f"Created deterministic datamodule wrapper using pre-cached data")
    else:
        deterministic_dm = datamodule
    
    # Trainer for linear probe with deterministic settings
    probe_trainer = Trainer(
        max_epochs=eval_cfg.linear_probe.max_epochs,
        accelerator='auto',
        devices=1,
        logger=linear_logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=True,  # Force deterministic algorithms
    )
    
    log.info("Training linear probe...")
    
    # CRITICAL: Seed one more time RIGHT BEFORE training
    # This ensures the linear probe's weights are initialized deterministically
    if hasattr(cfg, 'seed') and cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        log.info(f"Re-seeded immediately before probe training: {cfg.seed}")
    
    # Use the deterministic datamodule wrapper for training
    probe_trainer.fit(linear_probe, deterministic_dm)
    
    # Restore original num_workers
    if hasattr(datamodule, 'num_workers'):
        datamodule.num_workers = original_num_workers
        log.info(f"Restored num_workers={original_num_workers}")
    
    log.info("Testing linear probe...")
    # Use original datamodule for testing (no shuffling needed)
    linear_results = probe_trainer.test(linear_probe, datamodule)
    
    # Get detailed metrics from linear probe (after testing)
    detailed_metrics = linear_probe.get_detailed_metrics()
    
    linear_accuracy = detailed_metrics['accuracy']
    linear_f1_macro = detailed_metrics['f1_macro']
    linear_auroc_macro = detailed_metrics['auroc_macro']
    linear_f1_per_class = detailed_metrics['f1_per_class']
    linear_auroc_per_class = detailed_metrics['auroc_per_class']
    
    # Plot and save ROC curves
    log.info("Generating ROC curves...")
    roc_output_dir = output_dir / "roc_plots"
    linear_probe.plot_roc_curves(roc_output_dir)
    log.info(f"ROC curves saved to: {roc_output_dir}")
    
    results['linear_probe_accuracy'] = float(linear_accuracy)
    results['linear_probe_f1_macro'] = float(linear_f1_macro)
    results['linear_probe_auroc_macro'] = float(linear_auroc_macro)
    
    # Store per-class results with class names
    if class_names:
        for i, name in enumerate(class_names):
            results[f'linear_probe_f1_{name}'] = float(linear_f1_per_class[i])
            results[f'linear_probe_auroc_{name}'] = float(linear_auroc_per_class[i])
    else:
        results['linear_probe_f1_per_class'] = linear_f1_per_class.tolist()
        results['linear_probe_auroc_per_class'] = linear_auroc_per_class.tolist()
    
    # Print detailed linear probe results
    log.info(f"\n{'='*80}")
    log.info("LINEAR PROBE DETAILED RESULTS")
    log.info(f"{'='*80}")
    log.info(f"Accuracy: {linear_accuracy:.4f}")
    log.info(f"F1 Score (macro): {linear_f1_macro:.4f}")
    log.info(f"AUROC (macro): {linear_auroc_macro:.4f}")
    
    log.info(f"\nPer-Class Metrics:")
    log.info(f"{'-'*80}")
    if class_names:
        for i, name in enumerate(class_names):
            log.info(f"  {name:30s}: F1={linear_f1_per_class[i]:.4f}, AUROC={linear_auroc_per_class[i]:.4f}")
    else:
        for i in range(len(linear_f1_per_class)):
            log.info(f"  Class {i:2d}: F1={linear_f1_per_class[i]:.4f}, AUROC={linear_auroc_per_class[i]:.4f}")
    log.info("")
    
    # ========================================
    # SUMMARY
    # ========================================
    log.info("\n" + "="*80)
    log.info("PROBE EVALUATION SUMMARY")
    log.info("="*80)
    log.info(f"\nLinear Probe:")
    log.info(f"  Accuracy: {linear_accuracy:.4f}")
    log.info(f"  F1 (macro): {linear_f1_macro:.4f}")
    log.info(f"  AUROC (macro): {linear_auroc_macro:.4f}")
    log.info("="*80 + "\n")
    
    # Save detailed results to JSON
    json_path = output_dir / "probe_results.json"
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    log.info(f"Detailed results saved to: {json_path}")
    
    # Save human-readable summary to text file
    summary_path = output_dir / "probe_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("PROBE EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("LINEAR PROBE:\n")
        f.write(f"  Accuracy: {linear_accuracy:.4f}\n")
        f.write(f"  F1 (macro): {linear_f1_macro:.4f}\n")
        f.write(f"  AUROC (macro): {linear_auroc_macro:.4f}\n\n")
        
        if class_names:
            f.write("  Per-Class Metrics:\n")
            for i, name in enumerate(class_names):
                f.write(f"    {name:30s}: F1={linear_f1_per_class[i]:.4f}, AUROC={linear_auroc_per_class[i]:.4f}\n")
        
    
    log.info(f"Summary saved to: {summary_path}")
    
    # ========================================
    # 4. PROBE LOGITS VISUALIZATION (if enabled)
    # ========================================
    if eval_cfg.get("save_visualizations", False):
        log.info("\n" + "="*80)
        log.info("4. GENERATING PROBE LOGITS VISUALIZATIONS (AFTER PROBE)")
        log.info("="*80)
        
        viz_output_dir = output_dir / "visualizations"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract logits from trained linear probe
        max_viz_samples = eval_cfg.get("max_visualization_samples", 10000)
        log.info(f"Extracting logits from trained probe (max {max_viz_samples} samples)...")
        
        # Move linear probe to same device as encoder and set to eval mode
        linear_probe = linear_probe.to(device)
        linear_probe.eval()
        
        # Extract embeddings and get logits from probe
        probe_logits_list = []
        probe_labels_list = []
        
        with torch.no_grad():
            for batch in datamodule.test_dataloader():
                features, labels = batch
                features = features.to(device)
                
                # Get embeddings from encoder
                embeddings = encoder.get_embeddings(features)
                
                # Get logits from trained probe
                logits = linear_probe.classifier(embeddings)
                
                probe_logits_list.append(logits.cpu().numpy())
                probe_labels_list.append(labels.cpu().numpy())
        
        all_probe_logits = np.concatenate(probe_logits_list, axis=0)
        all_probe_labels = np.concatenate(probe_labels_list, axis=0)
        
        # Stratified sampling on logits (use local random generator)
        rng = np.random.RandomState(42)
        samples_per_class = max_viz_samples // datamodule.num_classes
        selected_indices = []
        
        log.info(f"Sampling from each class for logits visualization:")
        for class_idx in range(datamodule.num_classes):
            class_mask = all_probe_labels == class_idx
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) > 0:
                n_to_sample = min(samples_per_class, len(class_indices))
                sampled = rng.choice(class_indices, size=n_to_sample, replace=False)
                selected_indices.extend(sampled)
                log.info(f"  Class {class_idx}: {n_to_sample} samples (from {len(class_indices)} available)")
            else:
                log.warning(f"  Class {class_idx}: No samples found!")
        
        selected_indices = np.array(selected_indices)
        probe_logits = all_probe_logits[selected_indices]
        probe_labels = all_probe_labels[selected_indices]
        
        log.info(f"Extracted {len(probe_logits)} probe logits with shape {probe_logits.shape}")
        
        # Create UMAP visualization on probe logits
        visualize_embeddings_umap(
            embeddings=probe_logits,
            labels=probe_labels,
            class_names=class_names if class_names else [f"Class {i}" for i in range(datamodule.num_classes)],
            output_path=viz_output_dir / "umap_probe_logits_after_training.png",
            title="UMAP Projection of Trained Probe Logits",
            n_neighbors=eval_cfg.get("umap_n_neighbors", 15),
            min_dist=eval_cfg.get("umap_min_dist", 0.1),
            metric=eval_cfg.get("umap_metric", "cosine"),
        )
        
        log.info("Probe logits visualizations complete!")
    
    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for probe evaluation.
    
    Args:
        cfg: Hydra configuration
    """
    
    # Set seed for reproducibility
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
        log.info(f"Set seed to {cfg.seed}")
    
    # Check checkpoint path
    if not cfg.get("ckpt_path"):
        raise ValueError(
            "ckpt_path must be provided! "
            "Use: python src/eval_probes.py ckpt_path=/path/to/checkpoint.ckpt"
        )
    
    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    log.info(f"\n{'='*80}")
    log.info(f"Loading checkpoint: {ckpt_path}")
    log.info(f"{'='*80}\n")
    
    # Load pretrained model — use cfg.model._target_ to select the right class directly.
    _TARGET_TO_CLS = {
        "src.models.collide2v_augmented_supcon.COLLIDE2VAugmentedSupConLitModule":     COLLIDE2VAugmentedSupConLitModule,
        "src.models.collide2v_augmented_selfsupcon.COLLIDE2VAugmentedSelfSupConLitModule": COLLIDE2VAugmentedSelfSupConLitModule,
        "src.models.collide2v_vicreg.COLLIDE2VVICRegLitModule":                        COLLIDE2VVICRegLitModule,
        "src.models.collide2v_vicreg.COLLIDE2VVCRegLitModule":                         COLLIDE2VVCRegLitModule,
    }

    model = None
    load_errors = {}
    try:
        model_target = cfg.model._target_
    except Exception:
        model_target = None

    if model_target and model_target in _TARGET_TO_CLS:
        cls_name = model_target.split(".")[-1]
        cls = _TARGET_TO_CLS[model_target]
        try:
            model = cls.load_from_checkpoint(str(ckpt_path), weights_only=False)
            log.info(f"Loaded checkpoint as {cls_name}.")
        except Exception as e:
            load_errors[cls_name] = str(e)
            log.warning(f"Could not load as {cls_name} (from cfg.model._target_): {e}")

    if model is None:
        # Fallback: try each class in turn
        _candidates = [
            ("COLLIDE2VAugmentedSupConLitModule",      COLLIDE2VAugmentedSupConLitModule),
            ("COLLIDE2VAugmentedSelfSupConLitModule",  COLLIDE2VAugmentedSelfSupConLitModule),
            ("COLLIDE2VVICRegLitModule",               COLLIDE2VVICRegLitModule),
            ("COLLIDE2VVCRegLitModule",                COLLIDE2VVCRegLitModule),
        ]
        for cls_name, cls in _candidates:
            if model is not None:
                break
            try:
                model = cls.load_from_checkpoint(str(ckpt_path), weights_only=False)
                log.info(f"Loaded checkpoint as {cls_name} (fallback detection).")
            except Exception as e:
                load_errors[cls_name] = str(e)
                log.warning(f"Could not load as {cls_name}: {e}")

    if model is None:
        error_summary = "; ".join(f"{k}: {v}" for k, v in load_errors.items())
        log.error(f"Failed to load checkpoint with any known model class. Errors: {error_summary}")
        raise RuntimeError(f"Could not load checkpoint {ckpt_path}. Errors: {error_summary}")

    model.eval()
    torch.save(model.encoder.state_dict(), "encoder_eval.pth")
    
    # Setup datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")


    # Set eos_preproc_dir using the same logic as in training
    if hasattr(cfg, 'paths') and hasattr(cfg.data, 'paths') and hasattr(cfg.data, 'label'):
        preproc_path = f"{cfg.paths.eos_data_dir}/{cfg.data.label}/preprocessed/"
        cfg.data.paths.eos_preproc_dir = preproc_path
        print(f"[MATCH TRAIN] Using preprocessed folder for all splits: {preproc_path}")

    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        seed=cfg.get("seed", None),
    )
    
    log.info("Setting up datamodule...")
    datamodule.setup(stage="fit")
    log.info("Datamodule setup complete")

    # The `_before` half of the debug_comparison/ dump was written here — same test
    # batch and datamodule metadata, taken before evaluate_with_probes so the two could
    # be compared. Removed with the rest of that dump; the setup call above is not
    # diagnostic and stays.
    
    # Evaluate with probes
    results = evaluate_with_probes(cfg, model, datamodule, call_source='eval_probes.py')
    
    log.info("Probe evaluation complete!")


if __name__ == "__main__":
    main()
