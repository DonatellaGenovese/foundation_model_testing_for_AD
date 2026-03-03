"""
Evaluate a pretrained checkpoint with Linear and KNN probes.

This script loads a pretrained contrastive model and evaluates the quality
of learned embeddings using:
1. Linear Probe: Trainable linear classifier on frozen embeddings
2. KNN Probe: Non-parametric K-nearest neighbors on frozen embeddings

Usage:
    python src/eval_probes.py ckpt_path=/path/to/checkpoint.ckpt
    python src/eval_probes.py experiment=vanillasupcon_6class_pretrain ckpt_path=/path/to/checkpoint.ckpt
"""

from typing import Any, Dict, List, Tuple, Union
from pathlib import Path
import json

import hydra
import rootutils
import torch
import numpy as np
from lightning import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.collide2v_vanillasupcon import (
    COLLIDE2VVanillaSupConLitModule,
    LinearProbe,
    KNNProbe,
)
from src.models.collide2v_augmented_supcon import COLLIDE2VAugmentedSupConLitModule
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
    
    all_embeddings = []
    all_labels = []
    
    total_samples = 0
    
    # If stratified sampling, collect all data first, then sample
    if stratified and max_samples is not None and num_classes is not None:
        log.info(f"Using stratified sampling: {max_samples // num_classes} samples per class")
        
        # Set seed for reproducibility
        np.random.seed(42)
        
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
                sampled = np.random.choice(class_indices, size=n_to_sample, replace=False)
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


def evaluate_with_probes(
    cfg: DictConfig,
    model: Union[COLLIDE2VVanillaSupConLitModule, COLLIDE2VAugmentedSupConLitModule],
    datamodule: LightningDataModule,
) -> Dict[str, float]:
    """
    Evaluate trained model with Linear and KNN probes.
    
    Args:
        cfg: Hydra configuration
        model: Trained contrastive model
        datamodule: DataModule for data loading
        
    Returns:
        Dictionary with probe evaluation results
    """
    
    log.info("\n" + "="*80)
    log.info("STARTING PROBE EVALUATION")
    log.info("="*80)
    
    # Set model to eval mode
    model.eval()
    
    # Ensure datamodule is set up
    if not hasattr(datamodule, 'train_dataloader') or datamodule.train_dataloader is None:
        datamodule.setup('fit')

    # Ensure encoder is built even if no training loop was run (baseline case)
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        # Try vanilla supcon method (_build_from_datamodule)
        build_fn = getattr(model, "_build_from_datamodule", None)
        
        if build_fn is not None:
            log.info("Encoder not initialized; building encoder from datamodule (vanilla method).")
            build_fn(datamodule, stage="fit")
        else:
            # Try augmented supcon method (setup with trainer)
            setup_fn = getattr(model, "setup", None)
            if setup_fn is not None:
                log.info("Encoder not initialized; setting up model (augmented method).")
                # Create a proper trainer for setup
                temp_trainer = Trainer(logger=False, enable_checkpointing=False)
                temp_trainer.datamodule = datamodule
                model.trainer = temp_trainer
                setup_fn(stage="fit")
            else:
                raise RuntimeError(
                    "Encoder not initialized and model does not provide _build_from_datamodule or setup. "
                    "Make sure the checkpoint was saved after training."
                )
        
        encoder = model.encoder

        if encoder is None:
            raise RuntimeError("Failed to build encoder from datamodule for probe evaluation.")
    
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
            "knn_probe": {"k_values": [1, 3, 5, 10, 20], "metric": "cosine"},
            "output_dir": Path(cfg.paths.output_dir) / "probe_evaluation"
        })
    
    # Create output directory
    output_dir = Path(eval_cfg.get("output_dir", "./probe_evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # ========================================
    # 1. LINEAR PROBE EVALUATION
    # ========================================
    log.info("\n" + "="*80)
    log.info("1. LINEAR PROBE EVALUATION")
    log.info("="*80)
    
    # Get class names
    class_names = None
    if hasattr(datamodule, 'class_names'):
        class_names = datamodule.class_names
    elif hasattr(cfg.data, 'to_classify'):
        class_names = cfg.data.to_classify
    
    linear_probe = LinearProbe(
        encoder=encoder,
        embedding_dim=model.hparams.d_model,
        num_classes=datamodule.num_classes,
        class_names=class_names,
        lr=eval_cfg.linear_probe.lr,
        weight_decay=eval_cfg.linear_probe.weight_decay,
    )
    
    # Logger for linear probe
    linear_logger = TensorBoardLogger(
        save_dir=str(output_dir / "linear_probe"),
        name="",
        version="",
    )
    
    # Trainer for linear probe
    probe_trainer = Trainer(
        max_epochs=eval_cfg.linear_probe.max_epochs,
        accelerator='auto',
        devices=1,
        logger=linear_logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    log.info("Training linear probe...")
    probe_trainer.fit(linear_probe, datamodule)
    
    log.info("Testing linear probe...")
    linear_results = probe_trainer.test(linear_probe, datamodule)
    
    # Get detailed metrics from linear probe
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
    # 2. KNN PROBE EVALUATION
    # ========================================
    log.info("\n" + "="*80)
    log.info("2. KNN PROBE EVALUATION")
    log.info("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get class names if available
    class_names = None
    if hasattr(datamodule, 'class_names'):
        class_names = datamodule.class_names
    elif hasattr(cfg.data, 'to_classify'):
        class_names = cfg.data.to_classify
    
    knn_results_all = []
    for k in eval_cfg.knn_probe.k_values:
        log.info(f"\n--- Testing k={k} ---")
        
        knn_probe = KNNProbe(
            encoder=encoder,
            k=k,
            metric=eval_cfg.knn_probe.metric,
            device=device
        )
        
        knn_probe.fit(datamodule.train_dataloader())
        knn_results = knn_probe.evaluate(
            datamodule.test_dataloader(),
            class_names=class_names
        )
        
        knn_results_all.append((k, knn_results))
        results[f'knn_probe_k{k}_accuracy'] = float(knn_results['knn_accuracy'])
        results[f'knn_probe_k{k}_f1_macro'] = float(knn_results['knn_f1_macro'])
        results[f'knn_probe_k{k}_auroc_macro'] = float(knn_results['knn_auroc_macro'])
        
        # Store per-class results with class names
        if class_names:
            for i, name in enumerate(class_names):
                results[f'knn_probe_k{k}_f1_{name}'] = float(knn_results['knn_f1_per_class'][i])
                results[f'knn_probe_k{k}_auroc_{name}'] = float(knn_results['knn_auroc_per_class'][i])
        else:
            results[f'knn_probe_k{k}_f1_per_class'] = knn_results['knn_f1_per_class'].tolist()
            results[f'knn_probe_k{k}_auroc_per_class'] = knn_results['knn_auroc_per_class'].tolist()
    
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
    log.info("\nKNN Probe Results:")
    for k, res in knn_results_all:
        log.info(f"  k={k:2d}: Acc={res['knn_accuracy']:.4f}, F1={res['knn_f1_macro']:.4f}, AUROC={res['knn_auroc_macro']:.4f}")
        if class_names and k == knn_results_all[-1][0]:  # Show detailed results for best k (last one)
            log.info(f"    Per-Class Metrics (k={k}):")
            for i, name in enumerate(class_names):
                log.info(f"      {name:30s}: F1={res['knn_f1_per_class'][i]:.4f}, AUROC={res['knn_auroc_per_class'][i]:.4f}")
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
        
        f.write("\n" + "-"*80 + "\n\n")
        f.write("KNN PROBE:\n")
        for k, res in knn_results_all:
            f.write(f"\n  k={k}:\n")
            f.write(f"    Accuracy: {res['knn_accuracy']:.4f}\n")
            f.write(f"    F1 (macro): {res['knn_f1_macro']:.4f}\n")
            f.write(f"    AUROC (macro): {res['knn_auroc_macro']:.4f}\n")
            
            if class_names:
                f.write(f"    Per-Class Metrics:\n")
                for i, name in enumerate(class_names):
                    f.write(f"      {name:30s}: F1={res['knn_f1_per_class'][i]:.4f}, AUROC={res['knn_auroc_per_class'][i]:.4f}\n")
    
    log.info(f"Summary saved to: {summary_path}")
    
    # ========================================
    # 3. EMBEDDING VISUALIZATION (if enabled)
    # ========================================
    if eval_cfg.get("save_visualizations", False):
        log.info("\n" + "="*80)
        log.info("3. GENERATING EMBEDDING VISUALIZATIONS")
        log.info("="*80)
        
        viz_output_dir = output_dir / "visualizations"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract embeddings from test set
        max_viz_samples = eval_cfg.get("max_visualization_samples", 10000)
        log.info(f"Extracting embeddings from test set (max {max_viz_samples} samples)...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_embeddings, test_labels = extract_embeddings(
            encoder=encoder,
            dataloader=datamodule.test_dataloader(),
            device=device,
            max_samples=max_viz_samples,
            stratified=True,
            num_classes=datamodule.num_classes,
        )
        
        log.info(f"Extracted {len(test_embeddings)} embeddings with shape {test_embeddings.shape}")
        
        # Create UMAP visualization
        visualize_embeddings_umap(
            embeddings=test_embeddings,
            labels=test_labels,
            class_names=class_names if class_names else [f"Class {i}" for i in range(datamodule.num_classes)],
            output_path=viz_output_dir / "umap_embeddings.png",
            title="UMAP Projection of Learned Embeddings",
            n_neighbors=eval_cfg.get("umap_n_neighbors", 15),
            min_dist=eval_cfg.get("umap_min_dist", 0.1),
            metric=eval_cfg.get("umap_metric", "cosine"),
        )
        
        log.info("Embedding visualizations complete!")
    
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
    
    # Load pretrained model (vanilla SupCon or augmented SupCon)
    model = None
    vanilla_error = None
    augmented_error = None

    # 1) Try vanilla SupCon checkpoint
    try:
        model = COLLIDE2VVanillaSupConLitModule.load_from_checkpoint(
            str(ckpt_path), weights_only=False
        )
        log.info("Loaded checkpoint as COLLIDE2VVanillaSupConLitModule.")
    except Exception as e:
        vanilla_error = str(e)
        log.warning(f"Could not load as COLLIDE2VVanillaSupConLitModule: {e}")

    # 2) Fallback: try augmented SupCon checkpoint
    if model is None:
        try:
            # AugmentedSupCon builds encoder lazily in setup(), so load with strict=False
            model = COLLIDE2VAugmentedSupConLitModule.load_from_checkpoint(
                str(ckpt_path), weights_only=False, strict=False
            )
            log.info("Loaded checkpoint as COLLIDE2VAugmentedSupConLitModule (strict=False).")
        except Exception as e:
            augmented_error = str(e)
            log.error(
                "Failed to load checkpoint as either vanilla or augmented SupCon model. "
                f"Vanilla error: {vanilla_error}; Augmented error: {augmented_error}"
            )
            raise

    model.eval()
    
    # Setup datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")


    # Set eos_preproc_dir using the same logic as in training
    if hasattr(cfg, 'paths') and hasattr(cfg.data, 'paths') and hasattr(cfg.data, 'label'):
        preproc_path = f"{cfg.paths.eos_data_dir}/{cfg.data.label}/preprocessed/"
        cfg.data.paths.eos_preproc_dir = preproc_path
        print(f"[MATCH TRAIN] Using preprocessed folder for all splits: {preproc_path}")

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    
    # Evaluate with probes
    results = evaluate_with_probes(cfg, model, datamodule)
    
    log.info("Probe evaluation complete!")


if __name__ == "__main__":
    main()
