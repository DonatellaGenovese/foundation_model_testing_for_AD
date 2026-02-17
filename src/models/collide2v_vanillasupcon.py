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
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification.accuracy import Accuracy

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
            #nn.BatchNorm1d(hidden_dim),
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
    
    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Scaling factor for similarities (typical: 0.07-0.1)
                        Lower = harder discrimination, higher = softer
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, strict_multi_class: bool = True) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: L2-normalized projections [batch_size, embedding_dim]
            labels: Class labels [batch_size]
        
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            raise ValueError("SupConLoss requires batch_size >= 2.")
        
        # Ensure labels is proper shape [B, 1]
        labels = labels.contiguous().view(-1, 1)
        
        # Create positive pairs mask: mask[i,j] = 1 if labels[i] == labels[j]
        # This is a [B, B] matrix where entry (i,j) is 1 if same class, 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

        # SupCon needs both positives and negatives in-batch to learn class separation.
        # If a batch contains a single class only, contrastive learning degenerates.
        if strict_multi_class and torch.unique(labels).numel() < 2:
            raise ValueError(
                "SupConLoss received a single-class batch. "
                "Need at least 2 classes per batch for meaningful contrastive learning."
            )
        
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
        
        # Compute log probability for each positive pair
        # Numerator: similarity to positive
        # Denominator: sum of similarities to all samples (except self)
        log_prob = logits / self.temperature - torch.log(
            (exp_logits * logits_mask).sum(1, keepdim=True) + 1e-9  # Add epsilon for stability
        )
        
        # Compute mean log-likelihood over positive pairs
        # For each sample, sum over all its positive pairs and normalize by count
        mask_sum = mask.sum(1)  # Number of positive pairs per sample
        
        # Handle edge case: if a sample has no positives in batch
        # (shouldn't happen with proper batching, but add safety)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        # Sum log probabilities over positives and normalize
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss is negative mean log-likelihood (we want to maximize likelihood)
        loss = -mean_log_prob_pos.mean()
        
        return loss


# ...existing code...

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
        self.contrastive_criterion = SupConLoss(temperature=temperature)
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

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, labels = batch
        embeddings, projections, logits = self.forward(x)

        unique_classes = torch.unique(labels).numel()
        is_single_class_batch = unique_classes < 2

        if is_single_class_batch and self.hparams.allow_single_class_batches:
            contrastive_loss = projections.new_zeros(())
        else:
            contrastive_loss = self.contrastive_criterion(
                projections,
                labels,
                strict_multi_class=self.training,
            )

        result = {
            "contrastive_loss": contrastive_loss,
            "embeddings": embeddings,
            "projections": projections,
            "single_class_batch": torch.tensor(
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
        outputs = self.model_step(batch)

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
        self.log("debug/single_class_batch", outputs["single_class_batch"], on_step=True, on_epoch=True)

        if batch_idx == 0:
            self._log_debug_batch_stats(labels, outputs)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, labels = batch
        outputs = self.model_step(batch)

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
        self.log("val/single_class_batch", outputs["single_class_batch"], on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, labels = batch
        outputs = self.model_step(batch)

        loss = outputs["contrastive_loss"]
        self.log("test/con_loss", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            preds = outputs["preds"]
            test_acc = (preds == labels).float().mean()
            self.log("test/cls_loss", outputs["classification_loss"], on_step=False, on_epoch=True)
            self.log("test/acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/single_class_batch", outputs["single_class_batch"], on_step=False, on_epoch=True)
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

    def setup(self, stage: str) -> None:
        self._setup_calls += 1
        self._setup_calls_to_log = float(self._setup_calls)

        # Prevent accidental re-initialization
        if self._built:
            return

        if not (self.trainer and getattr(self.trainer, "datamodule", None)):
            raise RuntimeError("Datamodule not available in setup().")

        dm = self.trainer.datamodule
        eos_preproc_dir = getattr(dm, "paths", None)["eos_preproc_dir"]
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
