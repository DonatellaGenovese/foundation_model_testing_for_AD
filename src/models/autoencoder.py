"""
Autoencoder for Anomaly Detection.

This module implements autoencoders for anomaly detection on:
1. Raw input features (baseline)
2. Learned embeddings from contrastive models

Training Strategy:
- Train ONLY on background (QCD) to learn normal patterns
- Anomalies (Higgs) should have high reconstruction error

Usage:
    # On raw features
    ae = SimpleAutoencoder(input_dim=vlen)
    
    # On embeddings
    ae = SimpleAutoencoder(input_dim=512)  # d_model from encoder
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict

log = logging.getLogger(__name__)


class SimpleAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder.
    
    Architecture:
        Encoder: input → hidden3 → hidden2 → hidden1 → bottleneck
        Decoder: bottleneck → hidden1 → hidden2 → hidden3 → output
    
    Args:
        input_dim: dimension of input features
        compression: compression ratio (input_dim / bottleneck_dim)
        depth: number of encoder compression layers
    """
    
    def __init__(
        self,
        input_dim: int,
        compression: int=16,
        depth: int=3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        #define the bottleneck_dim based on compression rate
        bottleneck_dim = max(4, input_dim // compression)

        hidden_dims = []
        current_dim = input_dim
        
        #construct the hidden Layers based on depth and compression
        for _ in range(depth):
            next_dim = max(bottleneck_dim, current_dim // 2)
            hidden_dims.append(next_dim)
            current_dim = next_dim
        
        #construct the encoder Layers
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        
        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = bottleneck_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer 
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch_size, input_dim]
        
        Returns:
            reconstruction: Reconstructed input [batch_size, input_dim]
            bottleneck: Compressed representation [batch_size, bottleneck_dim]
        """
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction, bottleneck


class AutoencoderLitModule(LightningModule):
    """
    Lightning module for autoencoder training and anomaly detection.
    
    Training:
        - Train ONLY on normal data (QCD)
        - Minimize reconstruction error (MSE)
    
    Inference:
        - Compute reconstruction error on test data
        - High error → anomaly, Low error → normal
    
    Args:
        input_dim: Input dimension
        bottleneck_dim: Bottleneck dimension
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        normal_classes_label: Label for normal class (e.g., QCD = 0)
        anomaly_class_labels: List of anomaly class labels (e.g., Higgs variants)
    """
    
    def __init__(
        self,
        input_dim: int,
        compression: int = 16,
        depth: int = 3,
        dropout: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        normal_classes_labels: list = [0],
        anomaly_classes_labels: Optional[list] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SimpleAutoencoder(
            input_dim=input_dim,
            compression=compression,
            depth=depth,
            dropout=dropout,
        )
        
        # Loss
        self.criterion = nn.MSELoss()
        
        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss_normal = MeanMetric()
        self.val_loss_anomaly = MeanMetric()
        
        # Store reconstruction errors for threshold determination
        self.val_errors_normal = []
        self.val_errors_anomaly = []
        self.val_labels = []
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def compute_reconstruction_error(self, x: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
        """
        Compute reconstruction error per sample.
        
        Args:
            x: Input [batch_size, input_dim]
            reduction: 'none' for per-sample, 'mean' for batch average
        
        Returns:
            errors: Reconstruction errors [batch_size] or scalar
        """
        reconstruction, _ = self.forward(x)
        
        # MSE per sample (mean over features)
        if reduction == 'none':
            errors = torch.mean((x - reconstruction) ** 2, dim=1)
        else:
        # MSE on all the dimensions 
            errors = torch.mean((x - reconstruction) ** 2)
        
        return errors
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features, labels = batch
        
        # Filter: train ONLY on normal class (QCD)
        normal_mask = torch.isin(labels, torch.tensor(self.hparams.normal_classes_labels, device=labels.device))
        
        if not normal_mask.any():
            # Return dummy loss
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        features_normal = features[normal_mask]
        
        # Forward pass
        reconstruction, _ = self.forward(features_normal)
        
        # Loss: reconstruction error on normal samples only
        loss = self.criterion(reconstruction, features_normal)
        
        # Logging
        self.train_loss(loss)
        self.log("ae/train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        features, labels = batch
        
        # Compute reconstruction errors for all samples
        errors = self.compute_reconstruction_error(features, reduction='none')
        
        # Separate normal vs anomaly
        normal_mask = torch.isin(labels, torch.tensor(self.hparams.normal_classes_labels, device=labels.device))
        anomaly_classes = self.hparams.anomaly_class_labels or []
        anomaly_mask = torch.isin(
            labels,
            torch.tensor(anomaly_classes, device=labels.device)
            )
        
        if normal_mask.any():
            normal_errors = errors[normal_mask]
            self.val_loss_normal(normal_errors.mean())
            self.val_errors_normal.extend(normal_errors.cpu().numpy().tolist())
        
        if anomaly_mask.any():
            anomaly_errors = errors[anomaly_mask]
            self.val_loss_anomaly(anomaly_errors.mean())
            self.val_errors_anomaly.extend(anomaly_errors.cpu().numpy().tolist())
        
        # Store labels for test statistics
        self.val_labels.extend(labels.detach().cpu().numpy().tolist())
    
    def on_validation_epoch_end(self):
        # Log average errors
        normal_loss = self.val_loss_normal.compute()
        anomaly_loss = self.val_loss_anomaly.compute()
        
        self.log("ae/val_loss_normal", normal_loss, prog_bar=True)
        self.log("ae/val_loss_anomaly", anomaly_loss, prog_bar=True)
        
        # Compute separation ratio (anomaly / normal)
        if normal_loss > 0:
            separation_ratio = anomaly_loss / normal_loss
            self.log("ae/separation_ratio", separation_ratio, prog_bar=True)
        
        # Reset
        self.val_errors_normal.clear()
        self.val_errors_anomaly.clear()
        self.val_labels.clear()
        self.val_loss_normal.reset()
        self.val_loss_anomaly.reset()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Same as validation but stores for final evaluation."""
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        """
        At test time, print detailed MSE per class (anomalies vs normal)
        """
        
        # Group errors by class label
        errors_by_class = defaultdict(list)
        for error, label in zip(self.val_errors_normal + self.val_errors_anomaly, self.val_labels):
            errors_by_class[label].append(error)
        
        # Print MSE per class
        log.info("\n" + "="*80)
        log.info("RECONSTRUCTION MSE PER CLASS (Test Set)")
        log.info("="*80)
        log.info(f"{'Class':<20} {'Count':<10} {'Mean MSE':<15} {'Std MSE':<15}")
        log.info("-"*80)
        
        # Sort by class label
        for cls in sorted(errors_by_class.keys()):
            errors = errors_by_class[cls]
            mean_mse = np.mean(errors)
            std_mse = np.std(errors)
            count = len(errors)
            
            # Mark normal vs anomaly
            if cls in self.hparams.normal_classes_labels:
                marker = "(NORMAL - trained)"
            elif self.hparams.anomaly_class_labels and cls in self.hparams.anomaly_class_labels:
                marker = "(ANOMALY - unseen)"
            else:
                marker = ""
            
            log.info(f"Class {cls:<15} {count:<10} {mean_mse:<15.6f} {std_mse:<15.6f} {marker}")
        
        log.info("="*80)
        
        # Compute ratios
        if self.val_errors_normal and self.val_errors_anomaly:
            normal_mean = np.mean(self.val_errors_normal)
            anomaly_mean = np.mean(self.val_errors_anomaly)
            ratio = anomaly_mean / normal_mean if normal_mean > 0 else float('inf')
            
            log.info(f"\nNormal (QCD) Mean MSE:   {normal_mean:.6f}")
            log.info(f"Anomaly (Higgs) Mean MSE: {anomaly_mean:.6f}")
            log.info(f"Anomaly/Normal Ratio:     {ratio:.3f}x")
        
        log.info("="*80 + "\n")
        
        # Call parent's on_validation_epoch_end for standard logging
        return self.on_validation_epoch_end()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
    
    def plot_reconstruction_errors(self, output_dir: Path, class_names: Optional[list] = None):
        """
        Plot histogram of reconstruction errors for normal vs anomaly classes.
        
        Args:
            output_dir: Directory to save plot
            class_names: List of class names
        """
        if len(self.val_errors_normal) == 0 or len(self.val_errors_anomaly) == 0:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ========================================
        # HISTOGRAM
        # ========================================
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot histograms
        bins = np.linspace(
            min(min(self.val_errors_normal), min(self.val_errors_anomaly)),
            max(max(self.val_errors_normal), max(self.val_errors_anomaly)),
            60
        )
        
        ax.hist(self.val_errors_normal, bins=bins, alpha=0.6, 
                label=f'Normal (QCD) - n={len(self.val_errors_normal)}', 
                color='blue', edgecolor='darkblue')
        ax.hist(self.val_errors_anomaly, bins=bins, alpha=0.6, 
                label=f'Anomalies (Higgs) - n={len(self.val_errors_anomaly)}', 
                color='red', edgecolor='darkred')
        
        ax.set_xlabel('Reconstruction Error (MSE)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Count', fontsize=13, fontweight='bold')
        
        # Compute statistics
        normal_mean = np.mean(self.val_errors_normal)
        anomaly_mean = np.mean(self.val_errors_anomaly)
        ratio = anomaly_mean / normal_mean if normal_mean > 0 else float('inf')
        
        ax.set_title(f'Anomaly Detection - Reconstruction Errors (Separation Ratio={ratio:.2f}x)', 
                     fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'anomaly_scores_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # METRICS SUMMARY TEXT FILE
        # ========================================
        with open(output_dir / 'anomaly_detection_metrics.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("ANOMALY DETECTION METRICS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("Training Configuration:\n")
            f.write(f"  - Normal class (training): QCD\n")
            f.write(f"  - Anomaly classes (testing): Higgs variants\n")
            f.write(f"  - Normal samples: {len(self.val_errors_normal)}\n")
            f.write(f"  - Anomaly samples: {len(self.val_errors_anomaly)}\n\n")
            
            f.write("Reconstruction Errors:\n")
            f.write(f"  - QCD mean MSE: {normal_mean:.6f} ± {np.std(self.val_errors_normal):.6f}\n")
            f.write(f"  - Higgs mean MSE: {anomaly_mean:.6f} ± {np.std(self.val_errors_anomaly):.6f}\n")
            f.write(f"  - Separation ratio: {ratio:.3f}x\n\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\n✅ Plots and metrics saved to: {output_dir}")
        print(f"   - anomaly_scores_histogram.png")
        print(f"   - anomaly_detection_metrics.txt")
