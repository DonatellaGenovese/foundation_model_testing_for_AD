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

from typing import Any, Dict, List, Tuple
from pathlib import Path
import json

import hydra
import rootutils
import torch
import numpy as np
from lightning import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.collide2v_vanillasupcon import (
    COLLIDE2VVanillaSupConLitModule,
    LinearProbe,
    KNNProbe,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def evaluate_with_probes(
    cfg: DictConfig,
    model: COLLIDE2VVanillaSupConLitModule,
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
    encoder = model.encoder
    
    if encoder is None:
        raise RuntimeError(
            "Encoder not initialized. The model needs to be set up properly. "
            "Make sure the checkpoint was saved after training."
        )
    
    # Ensure datamodule is set up
    if not hasattr(datamodule, 'train_dataloader') or datamodule.train_dataloader is None:
        datamodule.setup('fit')
    
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
    
    # Load pretrained model
    try:
        model = COLLIDE2VVanillaSupConLitModule.load_from_checkpoint(str(ckpt_path))
        model.eval()
    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")
        raise
    
    # Setup datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Evaluate with probes
    results = evaluate_with_probes(cfg, model, datamodule)
    
    log.info("Probe evaluation complete!")


if __name__ == "__main__":
    main()
