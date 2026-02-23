"""
Vanilla Supervised Contrastive Learning (SupCon) for COLLIDE2V.

This module implements metric learning to create a discriminative embedding space
where events from the same physics process cluster together.

Key Components:
1. Encoder: TinyTransformer backbone (reused from classification model)
2. Projection Head: 2-layer MLP for contrastive learning (discarded after training)
3. Classification Head: Optional for monitoring (can be removed)
4. SupCon Loss: Pulls same-class samples together, pushes different-class apart
"""

from typing import Any, Dict, Optional, Tuple
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification import (
    Accuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassConfusionMatrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from .components.transformer import TinyTransformer


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Maps encoder output (d_model) to lower-dimensional contrastive space.
    This head is ONLY used during training and discarded for inference.
    
    Architecture:
        Input (512) → Linear → BatchNorm → ReLU → Linear → Output (128)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        """
        Args:
            input_dim: Encoder output dimension (d_model, typically 512)
            hidden_dim: Hidden layer dimension
            output_dim: Final projection dimension (smaller, typically 128)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            # Note: No activation on final layer - embeddings are L2 normalized later
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder embeddings [batch_size, input_dim]
        Returns:
            projections: [batch_size, output_dim]
        """
        return self.net(x)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss (SupCon).
    
    Paper: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
    https://arxiv.org/abs/2004.11362
    
    For each sample i:
    - Positive set P(i): All samples in batch with same class label
    - Negative set: All other samples in batch (different classes)
    - Loss: Pull positives close, push negatives away
    
    Formula:
        L = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(sim(i,p)/τ) / Σ_{k≠i} exp(sim(i,k)/τ) ]
    
    Where:
        - sim(i,j) = cosine similarity between L2-normalized embeddings
        - τ = temperature (lower = harder negative mining)
        - |P(i)| = number of positive pairs for sample i
    """
    
    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        """
        Args:
            temperature: Scaling factor for similarities (typical: 0.07-0.1)
                        Lower = harder discrimination, higher = softer
            base_temperature: Base temperature for loss scaling (from paper)
                             Typically set equal to temperature
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, strict_multi_class: bool = True, debug: bool = False) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: L2-normalized projections [batch_size, embedding_dim]
            labels: Class labels [batch_size]
            strict_multi_class: If True, raise error on single-class batch; if False, return 0 loss
            debug: If True, print debug information
        
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            raise ValueError("SupConLoss requires batch_size >= 2.")
        
        # Ensure labels is proper shape [B, 1]
        labels = labels.contiguous().view(-1, 1)
        
        # DEBUG: Check for single-class batches
        unique_classes = torch.unique(labels)
        num_classes_in_batch = len(unique_classes)
        
        if debug:
            print(f"\n[SupConLoss DEBUG]")
            print(f"  Batch size: {batch_size}")
            print(f"  Num unique classes: {num_classes_in_batch}")
            print(f"  Classes: {unique_classes.cpu().numpy()}")
            print(f"  Features shape: {features.shape}")
            print(f"  Features norm (should be ~1.0): min={features.norm(dim=1).min():.4f}, max={features.norm(dim=1).max():.4f}")
        
        # Create positive pairs mask: mask[i,j] = 1 if labels[i] == labels[j]
        # This is a [B, B] matrix where entry (i,j) is 1 if same class, 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

        # SupCon needs both positives and negatives in-batch to learn class separation.
        # If a batch contains a single class only, contrastive learning degenerates.
        if num_classes_in_batch < 2:
            if strict_multi_class:
                raise ValueError(
                    "SupConLoss received a single-class batch. "
                    "Need at least 2 classes per batch for meaningful contrastive learning."
                )
            else:
                if debug:
                    print(f"  [WARNING] Single-class batch detected! Returning zero loss.")
                # Return zero loss but keep gradient flow
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute similarity matrix: dot product between all pairs
        # Since features are L2-normalized, this gives cosine similarity
        similarity_matrix = torch.matmul(features, features.T)  # [B, B]
        
        # For numerical stability: subtract max from logits
        # This prevents overflow in exp() without changing gradients
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute exp(similarity / temperature)
        exp_logits = torch.exp(logits / self.temperature)
        
        # Create mask to remove self-similarity (diagonal elements)
        # We don't want to compare a sample with itself
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        
        # Apply mask: only keep non-diagonal elements
        mask = mask * logits_mask  # Positive pairs (excluding self)
        
        if debug:
            print(f"  Positive pairs per sample: min={mask.sum(1).min().item():.0f}, max={mask.sum(1).max().item():.0f}, mean={mask.sum(1).float().mean().item():.2f}")
            print(f"  Similarity (before temp): min={similarity_matrix.min().item():.4f}, max={similarity_matrix.max().item():.4f}")
            print(f"  Similarity / T: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        
        # Compute log probability for each positive pair
        # Numerator: similarity to positive
        # Denominator: sum of similarities to all samples (except self)
        log_prob = logits / self.temperature - torch.log(
            (exp_logits * logits_mask).sum(1, keepdim=True) + 1e-9  # Add epsilon for stability
        )
        
        if debug:
            print(f"  Log prob: min={log_prob.min().item():.4f}, max={log_prob.max().item():.4f}")
        
        # Compute mean log-likelihood over positive pairs
        # For each sample, sum over all its positive pairs and normalize by count
        mask_sum = mask.sum(1)  # Number of positive pairs per sample
        
        # Handle edge case: if a sample has no positives in batch
        # (shouldn't happen with proper batching, but add safety)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        # Sum log probabilities over positives and normalize
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        if debug:
            print(f"  Mean log prob pos: min={mean_log_prob_pos.min().item():.4f}, max={mean_log_prob_pos.max().item():.4f}")
        
        # Loss is negative mean log-likelihood (we want to maximize likelihood)
        # Apply temperature scaling as in the original paper
        # NOTE: This returns a SCALAR loss (mean over batch)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        if debug:
            print(f"  Temperature scaling: {self.temperature / self.base_temperature:.4f}")
            print(f"  Final loss (scalar): {loss.item():.4f}")
        
        return loss

class LinearProbe(LightningModule):
    """
    Linear classifier on top of FROZEN embeddings.
    
    Used to evaluate embedding quality: trains ONLY a linear layer.
    Encoder weights remain frozen.
    
    Usage:
        1. Train contrastive model
        2. Freeze encoder
        3. Train this probe → measures linear separability
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_classes: int,
        class_names: Optional[list] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # Freeze encoder
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # Trainable linear classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Additional test metrics
        self.test_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)
        self.test_f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None)
        self.test_auroc_macro = MulticlassAUROC(num_classes=num_classes, average='macro')
        
        self.val_acc_best = MaxMetric()
        
        # Store computed metrics for retrieval after test
        self.cached_test_metrics = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get logits from frozen embeddings."""
        with torch.no_grad():
            embeddings = self.encoder.get_embeddings(x)
        logits = self.classifier(embeddings)
        return logits
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log("probe/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("probe/train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        
        self.log("probe/val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("probe/val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.test_acc(preds, y)
        self.test_f1_per_class(preds, y)
        self.test_f1_macro(preds, y)
        self.test_auroc_per_class(probs, y)
        self.test_auroc_macro(probs, y)
        
        self.log("probe/test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("probe/test_f1_macro", self.test_f1_macro, on_step=False, on_epoch=True)
        self.log("probe/test_auroc_macro", self.test_auroc_macro, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("probe/val_acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        """Compute and cache per-class metrics before automatic reset."""
        # Compute all metrics before they get reset
        accuracy = self.test_acc.compute()
        f1_macro = self.test_f1_macro.compute()
        f1_per_class = self.test_f1_per_class.compute()
        auroc_macro = self.test_auroc_macro.compute()
        auroc_per_class = self.test_auroc_per_class.compute()
        
        # Cache results for later retrieval
        self.cached_test_metrics = {
            'accuracy': accuracy.cpu().numpy(),
            'f1_macro': f1_macro.cpu().numpy(),
            'f1_per_class': f1_per_class.cpu().numpy(),
            'auroc_macro': auroc_macro.cpu().numpy(),
            'auroc_per_class': auroc_per_class.cpu().numpy(),
        }
        
        # Log per-class metrics
        for i in range(len(f1_per_class)):
            self.log(f"probe/test_f1_class_{i}", f1_per_class[i])
            self.log(f"probe/test_auroc_class_{i}", auroc_per_class[i])
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed per-class and aggregate metrics from cached results."""
        if self.cached_test_metrics is None:
            raise RuntimeError("No cached test metrics found. Run trainer.test() first.")
        return self.cached_test_metrics
        
class KNNProbe:
    """
    K-Nearest Neighbors classifier on frozen embeddings.
    
    Non-parametric evaluation: no training, just fit on train embeddings.
    Measures local separability of embedding space.
    
    Usage:
        1. Train contrastive model
        2. Extract embeddings from train set
        3. Fit KNN
        4. Evaluate on test set
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        k: int = 5,
        metric: str = 'cosine',
        device: str = 'cuda'
    ):
        """
        Args:
            encoder: Pretrained encoder (will be set to eval mode)
            k: Number of neighbors
            metric: Distance metric ('cosine', 'euclidean', 'minkowski')
            device: Device for embedding extraction
        """
        self.encoder = encoder
        self.encoder.eval()
        self.device = device
        self.k = k
        self.metric = metric
        self.knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
        
        self.train_embeddings = None
        self.train_labels = None

    @torch.no_grad()
    def extract_embeddings(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings from dataloader."""
        embeddings_list = []
        labels_list = []
        
        self.encoder.to(self.device)
        
        for batch in dataloader:
            x, y = batch
            x = x.to(self.device)
            
            embeddings = self.encoder.get_embeddings(x)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(y.cpu().numpy())
        
        embeddings = np.vstack(embeddings_list)
        labels = np.concatenate(labels_list)
        
        return embeddings, labels

    def fit(self, train_dataloader):
        """Fit KNN on training embeddings."""
        print(f"\n{'='*60}")
        print(f"Extracting training embeddings for KNN (k={self.k}, metric={self.metric})...")
        print(f"{'='*60}")
        
        self.train_embeddings, self.train_labels = self.extract_embeddings(train_dataloader)
        
        print(f"Fitting KNN on {len(self.train_labels)} samples...")
        self.knn.fit(self.train_embeddings, self.train_labels)
        print(f"✓ KNN fitted successfully\n")
    
    def evaluate(self, test_dataloader, class_names: Optional[list] = None) -> Dict[str, Any]:
        """Evaluate KNN on test set with detailed per-class metrics."""
        print(f"{'='*60}")
        print("Extracting test embeddings...")
        print(f"{'='*60}")
        
        test_embeddings, test_labels = self.extract_embeddings(test_dataloader)
        
        print(f"Predicting with KNN (k={self.k})...")
        predictions = self.knn.predict(test_embeddings)
        predict_proba = self.knn.predict_proba(test_embeddings)
        
        # Compute aggregate metrics
        acc = accuracy_score(test_labels, predictions)
        f1_macro = f1_score(test_labels, predictions, average='macro')
        f1_weighted = f1_score(test_labels, predictions, average='weighted')
        
        # Compute per-class metrics
        f1_per_class = f1_score(test_labels, predictions, average=None)
        
        # Compute AUROC (one-vs-rest)
        try:
            auroc_macro = roc_auc_score(test_labels, predict_proba, average='macro', multi_class='ovr')
            auroc_per_class = roc_auc_score(test_labels, predict_proba, average=None, multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not compute AUROC: {e}")
            auroc_macro = 0.0
            auroc_per_class = np.zeros(len(np.unique(test_labels)))
        
        # Confusion matrix
        conf_matrix = confusion_matrix(test_labels, predictions)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"KNN Probe Results (k={self.k}, metric={self.metric})")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"AUROC (macro): {auroc_macro:.4f}")
        
        # Print per-class metrics
        print(f"\nPer-Class Metrics:")
        print(f"{'-'*60}")
        if class_names:
            for i, name in enumerate(class_names):
                print(f"  {name:20s}: F1={f1_per_class[i]:.4f}, AUROC={auroc_per_class[i]:.4f}")
        else:
            for i in range(len(f1_per_class)):
                print(f"  Class {i:2d}: F1={f1_per_class[i]:.4f}, AUROC={auroc_per_class[i]:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(
            test_labels,
            predictions,
            target_names=class_names,
            digits=4
        ))
        
        return {
            'knn_accuracy': acc,
            'knn_f1_macro': f1_macro,
            'knn_f1_weighted': f1_weighted,
            'knn_f1_per_class': f1_per_class,
            'knn_auroc_macro': auroc_macro,
            'knn_auroc_per_class': auroc_per_class,
            'predictions': predictions,
            'labels': test_labels,
            'embeddings': test_embeddings,
            'confusion_matrix': conf_matrix,
        }
    

class COLLIDE2VVanillaSupConLitModule(LightningModule):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        projection_dim: int = 128,
        hidden_projection_dim: int = 256,
        temperature: float = 0.1,
        base_temperature: float = 0.07,
        use_classification_head: bool = True,
        classification_weight: float = 0.1,
        allow_single_class_batches: bool = True,
        optimizer: Any = None,
        scheduler: Optional[Any] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Built in setup() (datamodule-dependent)
        self.encoder = None
        self.projection_head = None
        self.classification_head = None

        # Debug state
        self._built = False
        self._setup_calls = 0
        self._setup_calls_to_log: Optional[float] = None

        # Losses
        self.contrastive_criterion = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        self.classification_criterion = nn.CrossEntropyLoss() if use_classification_head else None

        # Metrics
        self.train_contrastive_loss = MeanMetric()
        self.val_contrastive_loss = MeanMetric()

        if use_classification_head:
            self.train_classification_loss = MeanMetric()
            self.val_classification_loss = MeanMetric()
            self.train_acc = Accuracy(task="multiclass", num_classes=6)  # overwritten in setup()
            self.val_acc = Accuracy(task="multiclass", num_classes=6)
            self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.encoder is None or self.projection_head is None:
            raise RuntimeError("Model components not initialized. setup() did not run correctly.")

        embeddings = self.encoder.get_embeddings(x)
        if self.training and not embeddings.requires_grad:
            raise RuntimeError("Embeddings do not require grad. Check encoder/get_embeddings path.")

        projections = self.projection_head(embeddings)
        projections = F.normalize(projections, dim=-1, p=2)

        logits = None
        if self.hparams.use_classification_head and self.classification_head is not None:
            logits = self.classification_head(embeddings)

        return embeddings, projections, logits

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int = 0) -> Dict[str, torch.Tensor]:
        x, labels = batch
        embeddings, projections, logits = self.forward(x)

        unique_classes = torch.unique(labels).numel()
        is_single_class_batch = unique_classes < 2
        
        # DEBUG: Print on first batch
        debug_loss = (batch_idx == 0 and self.training)

        if is_single_class_batch and self.hparams.allow_single_class_batches:
            contrastive_loss = projections.new_zeros((), requires_grad=True)
            if debug_loss:
                print(f"\n[VanillaSupCon] Single-class batch detected. Returning zero loss.")
        else:
            contrastive_loss = self.contrastive_criterion(
                projections,
                labels,
                strict_multi_class=self.training,
                debug=debug_loss,
            )

        result = {
            "contrastive_loss": contrastive_loss,
            "embeddings": embeddings,
            "projections": projections,
            "num_classes_in_batch": torch.tensor(unique_classes, device=labels.device),
            "is_single_class": torch.tensor(
                1.0 if is_single_class_batch else 0.0,
                device=labels.device,
            ),
        }

        if self.hparams.use_classification_head and logits is not None:
            classification_loss = self.classification_criterion(logits, labels)
            result["classification_loss"] = classification_loss
            result["logits"] = logits
            result["preds"] = logits.argmax(dim=-1)

        return result

    def _log_debug_batch_stats(self, labels: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> None:
        uniq, cnt = torch.unique(labels, return_counts=True)
        self.log("debug/classes_in_batch", float(len(uniq)), on_step=False, on_epoch=True)
        self.log("debug/min_class_count", cnt.min().float(), on_step=False, on_epoch=True)
        self.log("debug/max_class_count", cnt.max().float(), on_step=False, on_epoch=True)
        self.log("debug/emb_std", outputs["embeddings"].std(), on_step=False, on_epoch=True)
        self.log("debug/proj_std", outputs["projections"].std(), on_step=False, on_epoch=True)

        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("debug/lr", lr, on_step=True, on_epoch=False, prog_bar=False)

        # Detect potential "no positive-pair" anchors quickly
        labels_2d = labels.view(-1, 1)
        same = (labels_2d == labels_2d.T).float()
        same.fill_diagonal_(0.0)
        no_pos_ratio = (same.sum(dim=1) == 0).float().mean()
        self.log("debug/no_positive_anchor_ratio", no_pos_ratio, on_step=False, on_epoch=True)

        diff = 1.0 - (labels_2d == labels_2d.T).float()
        diff.fill_diagonal_(0.0)
        no_neg_ratio = (diff.sum(dim=1) == 0).float().mean()
        self.log("debug/no_negative_anchor_ratio", no_neg_ratio, on_step=False, on_epoch=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, labels = batch
        outputs = self.model_step(batch, batch_idx=batch_idx)

        if batch_idx == 0 and self._setup_calls_to_log is not None:
            self.log("debug/setup_calls", self._setup_calls_to_log, on_step=False, on_epoch=True)
            self._setup_calls_to_log = None

        loss = outputs["contrastive_loss"]
        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            self.train_classification_loss(outputs["classification_loss"])
            self.train_acc(outputs["preds"], labels)
            self.log("train/cls_loss", self.train_classification_loss, on_step=False, on_epoch=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.train_contrastive_loss(outputs["contrastive_loss"])
        self.log("train/con_loss", self.train_contrastive_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        
        # Log batch statistics
        self.log("debug/num_classes_in_batch", outputs["num_classes_in_batch"], on_step=True, on_epoch=True)
        self.log("debug/single_class_batch", outputs["is_single_class"], on_step=True, on_epoch=True)

        if batch_idx == 0:
            self._log_debug_batch_stats(labels, outputs)
            # Log projection norms
            self.log("debug/proj_norm_min", outputs["projections"].norm(dim=1).min(), on_step=False, on_epoch=True)
            self.log("debug/proj_norm_max", outputs["projections"].norm(dim=1).max(), on_step=False, on_epoch=True)
            # Log class distribution
            unique_labels, counts = torch.unique(labels, return_counts=True)
            self.log("debug/min_samples_per_class", counts.min().float(), on_step=False, on_epoch=True)
            self.log("debug/max_samples_per_class", counts.max().float(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, labels = batch
        outputs = self.model_step(batch, batch_idx=batch_idx)

        loss = outputs["contrastive_loss"]
        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            self.val_classification_loss(outputs["classification_loss"])
            self.val_acc(outputs["preds"], labels)
            self.log("val/cls_loss", self.val_classification_loss, on_step=False, on_epoch=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_contrastive_loss(outputs["contrastive_loss"])
        self.log("val/con_loss", self.val_contrastive_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        # Log batch statistics (mirror train_step behavior)
        self.log(
            "debug/val_num_classes_in_batch",
            outputs["num_classes_in_batch"].float() if hasattr(outputs["num_classes_in_batch"], "float") else float(outputs["num_classes_in_batch"]),
            on_step=False,
            on_epoch=True,
        )
        self.log("debug/val_single_class_batch", outputs["is_single_class"], on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, labels = batch
        outputs = self.model_step(batch, batch_idx=batch_idx)

        loss = outputs["contrastive_loss"]
        self.log("test/con_loss", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            preds = outputs["preds"]
            test_acc = (preds == labels).float().mean()
            self.log("test/cls_loss", outputs["classification_loss"], on_step=False, on_epoch=True)
            self.log("test/acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/single_class_batch", outputs["is_single_class"], on_step=False, on_epoch=True)
        return {"loss": loss}

    def on_after_backward(self) -> None:
        # Gradient flow diagnostics
        enc_g = [p.grad.norm() for p in self.encoder.parameters() if p.grad is not None] if self.encoder else []
        proj_g = [p.grad.norm() for p in self.projection_head.parameters() if p.grad is not None] if self.projection_head else []

        enc_grad = torch.stack(enc_g).mean() if len(enc_g) > 0 else torch.tensor(0.0, device=self.device)
        proj_grad = torch.stack(proj_g).mean() if len(proj_g) > 0 else torch.tensor(0.0, device=self.device)

        self.log("debug/gradnorm_encoder", enc_grad, on_step=True, on_epoch=False)
        self.log("debug/gradnorm_projection", proj_grad, on_step=True, on_epoch=False)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.use_classification_head:
            acc = self.val_acc.compute()
            self.val_acc_best(acc)
            self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def _build_from_datamodule(self, dm: Any, stage: str = "fit") -> None:
        """Initialize encoder and heads from a COLLIDE2V datamodule.

        Normally called via ``setup()`` when training with a Trainer, but it
        can also be used manually (e.g. for baseline probe evaluation without
        running a training loop).
        """

        if self._built:
            return

        paths = getattr(dm, "paths", None)
        if paths is None or "eos_preproc_dir" not in paths:
            raise RuntimeError("datamodule.paths['eos_preproc_dir'] is required to build the encoder.")

        eos_preproc_dir = paths["eos_preproc_dir"]
        feature_map_path = os.path.join(eos_preproc_dir, "feature_map.json")
        with open(feature_map_path, "r") as f:
            feature_map = json.load(f)

        num_classes = getattr(dm, "num_classes", None)
        if num_classes is None:
            raise RuntimeError("datamodule.num_classes is None.")

        self.encoder = TinyTransformer(
            feature_map=feature_map,
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            num_layers=self.hparams.num_layers,
            d_ff=self.hparams.d_ff,
            dropout=self.hparams.dropout,
            num_classes=num_classes,
        )

        self.projection_head = ProjectionHead(
            input_dim=self.hparams.d_model,
            hidden_dim=self.hparams.hidden_projection_dim,
            output_dim=self.hparams.projection_dim,
        )

        if self.hparams.use_classification_head:
            self.classification_head = nn.Linear(self.hparams.d_model, num_classes)
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)

        self._built = True

    def setup(self, stage: str) -> None:
        self._setup_calls += 1
        self._setup_calls_to_log = float(self._setup_calls)

        # Prevent accidental re-initialization
        if self._built:
            return

        if not (self.trainer and getattr(self.trainer, "datamodule", None)):
            raise RuntimeError("Datamodule not available in setup().")

        dm = self.trainer.datamodule
        self._build_from_datamodule(dm, stage=stage)

    def configure_optimizers(self) -> Dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found.")

        optimizer = self.hparams.optimizer(params=params)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Encoder is not initialized.")
        return self.encoder.get_embeddings(x)
