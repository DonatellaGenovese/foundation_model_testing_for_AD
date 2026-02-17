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
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
        
        # Ensure labels is proper shape [B, 1]
        labels = labels.contiguous().view(-1, 1)
        
        # Create positive pairs mask: mask[i,j] = 1 if labels[i] == labels[j]
        # This is a [B, B] matrix where entry (i,j) is 1 if same class, 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]
        
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


class COLLIDE2VVanillaSupConLitModule(LightningModule):
    """
    Vanilla Supervised Contrastive Learning for COLLIDE2V.
    
    This module learns a discriminative embedding space using supervised contrastive
    learning WITHOUT data augmentation (hence "vanilla"). Class labels define
    positive/negative pairs directly.
    
    Training Flow:
        1. Input batch → Encoder → Embeddings [B, 512]
        2. Embeddings → Projection Head → Projections [B, 128]
        3. L2-normalize projections
        4. Compute SupCon loss using class labels
        5. (Optional) Compute classification loss for monitoring
    
    After Training:
        - Remove projection head
        - Use encoder.get_embeddings() for inference
        - Extract cluster centers for each SM class
    
    Usage:
        # Training
        python src/train.py experiment=vanillasupcon_6class_pretrain
        
        # After training, extract embeddings
        encoder = model.encoder
        embeddings = encoder.get_embeddings(batch)
    """
    
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
        use_classification_head: bool = True,  # Keep for monitoring training
        classification_weight: float = 0.1,     # Weight for classification loss
        optimizer: Any = None,
        scheduler: Optional[Any] = None,
        compile: bool = False,
    ) -> None:
        """
        Args:
            d_model: Transformer model dimension
            n_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            projection_dim: Output dimension of projection head (128 recommended)
            hidden_projection_dim: Hidden dimension in projection head
            temperature: Temperature for contrastive loss (0.1 recommended)
            use_classification_head: If True, adds classification head for monitoring
            classification_weight: Weight for classification loss (much lower than contrastive)
            optimizer: Optimizer (instantiated by hydra)
            scheduler: LR scheduler (instantiated by hydra)
            compile: Whether to use torch.compile
        """
        super().__init__()
        
        # Save hyperparameters (except optimizer/scheduler which are callables)
        self.save_hyperparameters(logger=False)
        
        # Model components - initialized in setup() after datamodule is available
        self.encoder = None                # TinyTransformer backbone
        self.projection_head = None        # MLP for contrastive learning
        self.classification_head = None    # Optional linear classifier
        
        # Loss functions
        self.contrastive_criterion = SupConLoss(temperature=temperature)
        self.classification_criterion = nn.CrossEntropyLoss() if use_classification_head else None
        
        # Metrics for logging
        self.train_contrastive_loss = MeanMetric()
        self.val_contrastive_loss = MeanMetric()
        
        # Optional classification metrics (for monitoring)
        if use_classification_head:
            self.train_classification_loss = MeanMetric()
            self.val_classification_loss = MeanMetric()
            self.train_acc = Accuracy(task="multiclass", num_classes=6)  # Updated in setup()
            self.val_acc = Accuracy(task="multiclass", num_classes=6)
            self.val_acc_best = MaxMetric()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder, projection head, and optional classifier.
        
        Args:
            x: Input batch - FLAT TENSOR [batch_size, feature_dim]
        
        Returns:
            embeddings: Encoder output [B, d_model] - used for final inference
            projections: L2-normalized projections [B, projection_dim] - used for contrastive loss
            logits: Classification logits [B, num_classes] or None - only for monitoring
        """
        # Step 1: Get encoder embeddings
        embeddings = self.encoder.get_embeddings(x)  # [B, d_model]
        
        # Step 2: Project to contrastive space
        projections = self.projection_head(embeddings)  # [B, projection_dim]
        
        # Step 3: L2 normalize projections (required for cosine similarity)
        projections = F.normalize(projections, dim=-1, p=2)
        
        # Step 4: Optional classification (for monitoring only)
        logits = None
        if self.hparams.use_classification_head and self.classification_head is not None:
            logits = self.classification_head(embeddings)  # [B, num_classes]
        
        return embeddings, projections, logits
    
    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single forward pass and compute losses.
        
        Args:
            batch: Tuple of (inputs, labels)
        
        Returns:
            Dictionary containing losses and outputs for logging
        """
        x, labels = batch
        
        # Forward pass
        embeddings, projections, logits = self.forward(x)
        
        # Compute contrastive loss (primary objective)
        contrastive_loss = self.contrastive_criterion(projections, labels)
        
        # Prepare return dictionary
        result = {
            "contrastive_loss": contrastive_loss,
            "embeddings": embeddings,
            "projections": projections,
        }
        
        # Optional: compute classification loss for monitoring
        if self.hparams.use_classification_head and logits is not None:
            classification_loss = self.classification_criterion(logits, labels)
            result["classification_loss"] = classification_loss
            result["logits"] = logits
            result["preds"] = logits.argmax(dim=-1)
        
        return result
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step - called for each batch during training.
        
        Loss composition:
            total_loss = contrastive_loss + α * classification_loss
            where α = 0.1 (classification is just for monitoring)
        """
        outputs = self.model_step(batch)
        x, labels = batch
        
        # Primary loss is contrastive
        loss = outputs["contrastive_loss"]
        
        # Add weighted classification loss if enabled (much lower weight)
        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            
            # Log classification metrics
            self.train_classification_loss(outputs["classification_loss"])
            self.train_acc(outputs["preds"], labels)
            self.log("train/cls_loss", self.train_classification_loss, 
                    on_step=False, on_epoch=True)
            self.log("train/acc", self.train_acc, 
                    on_step=False, on_epoch=True, prog_bar=True)
        
        # Log contrastive loss (primary metric)
        self.train_contrastive_loss(outputs["contrastive_loss"])
        self.log("train/con_loss", self.train_contrastive_loss, 
                on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step - called for each batch during validation.
        """
        outputs = self.model_step(batch)
        x, labels = batch
        
        # Compute total loss (same composition as training)
        loss = outputs["contrastive_loss"]
        
        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            
            # Log classification metrics
            self.val_classification_loss(outputs["classification_loss"])
            self.val_acc(outputs["preds"], labels)
            self.log("val/cls_loss", self.val_classification_loss, 
                    on_step=False, on_epoch=True)
            self.log("val/acc", self.val_acc, 
                    on_step=False, on_epoch=True, prog_bar=True)
        
        # Log contrastive loss
        self.val_contrastive_loss(outputs["contrastive_loss"])
        self.log("val/con_loss", self.val_contrastive_loss, 
                on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step - called for each batch during testing.
        
        Evaluates the model on the test set with the same metrics as validation.
        """
        x, labels = batch
        outputs = self.model_step(batch)
        
        # Compute total loss
        loss = outputs["contrastive_loss"]
        
        # Log contrastive loss
        self.log("test/con_loss", outputs["contrastive_loss"], 
                on_step=False, on_epoch=True, prog_bar=True)
        
        # Optional: classification metrics
        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            
            # Log classification metrics
            preds = outputs["preds"]
            test_acc = (preds == labels).float().mean()
            
            self.log("test/cls_loss", outputs["classification_loss"], 
                    on_step=False, on_epoch=True)
            self.log("test/acc", test_acc, 
                    on_step=False, on_epoch=True, prog_bar=True)
        
        # Log total loss
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        
        return {"loss": loss}
    
    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation epoch.
        Track best validation accuracy (if classification head is used).
        """
        if self.hparams.use_classification_head:
            acc = self.val_acc.compute()
            self.val_acc_best(acc)
            self.log("val/acc_best", self.val_acc_best.compute(), 
                    sync_dist=True, prog_bar=True)
    
    def setup(self, stage: str) -> None:
        """
        Setup method called by Lightning before training/validation/testing.
        
        This is where we instantiate the model components because we need
        information from the datamodule (feature_map, num_classes).
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Only setup if we have a datamodule attached
        if self.trainer and getattr(self.trainer, "datamodule", None):
            dm = self.trainer.datamodule
            
            # Load feature map from preprocessed data
            # This tells us which features exist and their dimensions
            eos_preproc_dir = getattr(dm, "paths", None)['eos_preproc_dir']
            feature_map_path = os.path.join(eos_preproc_dir, "feature_map.json")
            with open(feature_map_path, "r") as f:
                feature_map = json.load(f)
            
            # Get number of classes from datamodule
            num_classes = getattr(dm, "num_classes", None)
            
            # Create encoder (TinyTransformer backbone)
            # This is the same architecture as your classification model
            self.encoder = TinyTransformer(
                feature_map=feature_map,
                d_model=self.hparams.d_model,
                n_heads=self.hparams.n_heads,
                num_layers=self.hparams.num_layers,
                d_ff=self.hparams.d_ff,
                dropout=self.hparams.dropout,
                num_classes=num_classes,
            )
            
            # Create projection head for contrastive learning
            # NOTE: This head is only used during training and discarded afterwards
            self.projection_head = ProjectionHead(
                input_dim=self.hparams.d_model,
                hidden_dim=self.hparams.hidden_projection_dim,
                output_dim=self.hparams.projection_dim,
            )
            
            # Create optional classification head for monitoring
            if self.hparams.use_classification_head:
                self.classification_head = nn.Linear(self.hparams.d_model, num_classes)
                
                # Update metrics with correct num_classes
                self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
                self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Optionally compile model for faster training (PyTorch 2.0+)
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and optionally scheduler configuration
        """
        # Instantiate optimizer with model parameters
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        # If scheduler is provided, configure it
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/con_loss",  # Monitor contrastive loss for scheduling
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for inference/clustering.
        
        This method should be used AFTER training to extract the learned
        representation for downstream tasks (anomaly detection, clustering).
        
        Args:
            x: Input batch
        
        Returns:
            embeddings: [batch_size, d_model]
        
        Usage:
            model = COLLIDE2VVanillaSupConLitModule.load_from_checkpoint(ckpt_path)
            embeddings = model.get_embeddings(batch)
        """
        return self.encoder.get_embeddings(x)