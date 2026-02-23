"""
Augmented Supervised Contrastive Learning (Aug-SupCon) for COLLIDE2V.

This module implements metric learning with data augmentation to create a robust
discriminative embedding space where events from the same physics process cluster
together despite input perturbations.

Key Components:
1. Augmentation: Random masking to create multiple views of each sample
2. Encoder: TinyTransformer backbone (shared for all views)
3. Projection Head: 2-layer MLP for contrastive learning (discarded after training)
4. SimCLR Loss: Contrastive loss that treats augmented views as positives
5. Classification Head: Optional for monitoring (can be removed)

Differences from Vanilla SupCon:
- Applies random masking augmentation to create two views per sample
- Uses SimCLR-style loss that considers both augmented views and same-class samples as positives
- More robust to input variations and missing information
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


class RandomMaskingAugmentation(nn.Module):
    """
    Random masking augmentation for physics event data.
    
    Creates perturbed views of input by randomly masking (setting to 0) parts of the input.
    This simulates:
    - Detector inefficiencies
    - Missing particles
    - Measurement uncertainties
    
    Two masking strategies:
    1. Feature-level: Randomly mask individual features with probability p
    2. Particle-level: Randomly mask entire particles (all features of an object) with probability p
    
    Args:
        feature_map: Dictionary mapping group names to their slice indices and structure
        mask_probability: Probability of masking (default: 0.2)
        mask_full_particle: If True, mask entire particles; if False, mask individual features
    """
    
    def __init__(
        self,
        feature_map: Dict[str, Any],
        mask_probability: float = 0.2,
        mask_full_particle: bool = False,
    ):
        super().__init__()
        self.feature_map = feature_map
        self.mask_probability = mask_probability
        self.mask_full_particle = mask_full_particle
        
        # Parse feature map to understand structure
        self.group_configs = []
        for group_name, cfg in feature_map.items():
            start = cfg["start"]
            end = cfg["end"]
            topk = cfg["topk"]
            num_cols = len(cfg["columns"])
            use_count = cfg.get("count", False)
            
            # Calculate total size and structure
            if topk is None:
                # Scalar features (e.g., MET)
                num_particles = 1
                particle_size = num_cols
            else:
                # Object collections (e.g., jets, leptons)
                num_particles = topk
                particle_size = num_cols
            
            self.group_configs.append({
                "name": group_name,
                "start": start,
                "end": end,
                "num_particles": num_particles,
                "particle_size": particle_size,
                "use_count": use_count,
                "is_scalar": topk is None,
            })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random masking augmentation to input batch.
        
        Args:
            x: Input tensor [batch_size, feature_dim]
        
        Returns:
            Augmented tensor [batch_size, feature_dim] with random values masked to 0
        """
        batch_size, feature_dim = x.shape
        device = x.device
        
        # Create a copy to avoid in-place modification
        augmented = x.clone()
        
        if self.mask_full_particle:
            # Strategy 1: Mask entire particles/objects
            # For each group, randomly decide which particles to mask
            for group_cfg in self.group_configs:
                # Skip scalar features (MET, etc.) - don't mask these
                if group_cfg["is_scalar"]:
                    continue
                
                start = group_cfg["start"]
                num_particles = group_cfg["num_particles"]
                particle_size = group_cfg["particle_size"]
                
                # For each sample in batch, create a mask for particles
                # Shape: [batch_size, num_particles]
                particle_mask = torch.rand(batch_size, num_particles, device=device) < self.mask_probability
                
                # Expand mask to cover all features of masked particles
                # Shape: [batch_size, num_particles, particle_size]
                expanded_mask = particle_mask.unsqueeze(-1).expand(-1, -1, particle_size)
                
                # Reshape to match the slice in the feature vector
                # Shape: [batch_size, num_particles * particle_size]
                flat_mask = expanded_mask.reshape(batch_size, -1)
                
                # Calculate end index (excluding count feature if present)
                end_idx = start + num_particles * particle_size
                
                # Apply mask: set masked positions to 0
                augmented[:, start:end_idx][flat_mask] = 0.0
        
        else:
            # Strategy 2: Mask individual features randomly
            # Create a random mask for all features
            # Shape: [batch_size, feature_dim]
            mask = torch.rand(batch_size, feature_dim, device=device) < self.mask_probability
            
            # Apply mask: set masked positions to 0
            augmented[mask] = 0.0
        
        return augmented


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Maps encoder output (d_model) to lower-dimensional contrastive space.
    This head is ONLY used during training and discarded for inference.
    
    Architecture:
        Input (512) → Linear → ReLU → Linear → Output (128)
    
    Note: BatchNorm can be added but is optional for contrastive learning.
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
            print(f"  Loss (before mean): min={loss.min().item():.4f}, max={loss.max().item():.4f}" if loss.dim() > 0 else f"  Loss (scalar): {loss.item():.4f}")
            print(f"  Temperature scaling: {self.temperature / self.base_temperature:.4f}")
            print(f"  Final loss (scalar): {loss.item():.4f}")
        
        return loss


class COLLIDE2VAugmentedSupConLitModule(LightningModule):
    """
    Augmented Supervised Contrastive Learning Module for COLLIDE2V.
    
    This module trains an encoder to produce discriminative embeddings using:
    1. Data augmentation (random masking) to create multiple views
    2. Contrastive learning to pull same-class samples together
    3. Optional classification head for monitoring
    
    Training Strategy:
    - For each batch, create 2 augmented views of each sample
    - Pass both views through the encoder (shared weights)
    - Project embeddings to contrastive space
    - Compute SimCLR loss: treat same-class samples as positives
    - Optionally train a classification head for monitoring
    
    After Training:
    - Encoder can be used for downstream tasks (classification, retrieval, etc.)
    - Projection head is discarded (only used for contrastive learning)
    - Classification head (if used) can be fine-tuned or replaced
    
    Args:
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout probability
        projection_dim: Dimension of contrastive projection space (typically 128)
        hidden_projection_dim: Hidden dimension in projection head (typically 256)
        temperature: Temperature for contrastive loss (0.05-0.1)
        mask_probability: Probability of masking in augmentation (0.1-0.5)
        mask_full_particle: If True, mask entire particles; if False, mask features
        num_augmentations: Number of augmented views per sample (typically 2)
        use_classification_head: Whether to train a classification head for monitoring
        classification_weight: Weight for classification loss (if used)
        optimizer: Optimizer constructor (from hydra config)
        scheduler: LR scheduler constructor (optional)
        compile: Whether to compile encoder with torch.compile (faster on GPU)
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
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        mask_probability: float = 0.2,
        mask_full_particle: bool = False,
        num_augmentations: int = 2,
        use_classification_head: bool = True,
        classification_weight: float = 0.1,
        allow_single_class_batches: bool = True,
        optimizer: Any = None,
        scheduler: Optional[Any] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        
        # Save hyperparameters (will be logged and saved in checkpoint)
        self.save_hyperparameters(logger=False)
        
        # Model components (built in setup() after datamodule is available)
        self.encoder = None
        self.projection_head = None
        self.classification_head = None
        self.augmentation = None
        
        # Debug state
        self._built = False
        self._setup_calls = 0
        
        # Loss functions
        self.contrastive_criterion = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        self.classification_criterion = nn.CrossEntropyLoss() if use_classification_head else None
        
        # Training metrics
        self.train_contrastive_loss = MeanMetric()
        self.val_contrastive_loss = MeanMetric()
        
        if use_classification_head:
            self.train_classification_loss = MeanMetric()
            self.val_classification_loss = MeanMetric()
            self.train_acc = Accuracy(task="multiclass", num_classes=6)  # Updated in setup()
            self.val_acc = Accuracy(task="multiclass", num_classes=6)
            self.val_acc_best = MaxMetric()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder and projection head.
        
        Args:
            x: Input features [batch_size, feature_dim]
        
        Returns:
            embeddings: Encoder output [batch_size, d_model]
            projections: L2-normalized projections [batch_size, projection_dim]
            logits: Classification logits [batch_size, num_classes] (if classification head is used)
        """
        if self.encoder is None or self.projection_head is None:
            raise RuntimeError("Model components not initialized. setup() did not run correctly.")
        
        # Extract embeddings from encoder
        embeddings = self.encoder.get_embeddings(x)
        
        # Safety check: ensure embeddings require gradients during training
        if self.training and not embeddings.requires_grad:
            raise RuntimeError("Embeddings do not require grad. Check encoder/get_embeddings path.")
        
        # Project to contrastive space
        projections = self.projection_head(embeddings)
        
        # L2 normalize projections (required for cosine similarity in loss)
        projections = F.normalize(projections, dim=-1, p=2)
        
        # Optional classification
        logits = None
        if self.hparams.use_classification_head and self.classification_head is not None:
            logits = self.classification_head(embeddings)
        
        return embeddings, projections, logits
    
    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single forward pass with augmentation and loss computation.
        
        Strategy:
        1. Create N augmented views of each sample
        2. Pass all views through encoder (concatenated for efficiency)
        3. Compute contrastive loss over all views
        4. Optionally compute classification loss
        
        Args:
            batch: Tuple of (features, labels)
                  features: [batch_size, feature_dim]
                  labels: [batch_size]
            batch_idx: Index of current batch (for debug logging)
        
        Returns:
            Dictionary containing:
                - contrastive_loss: SupCon contrastive loss
                - embeddings: Encoder embeddings (first view only)
                - projections: Normalized projections (first view only)
                - classification_loss: Optional classification loss
                - logits: Optional classification logits (first view only)
                - preds: Optional predictions (first view only)
                - num_classes_in_batch: Number of unique classes in batch
                - is_single_class: 1.0 if single-class batch, 0.0 otherwise
        """
        x, labels = batch
        batch_size = x.shape[0]
        
        # Create multiple augmented views
        augmented_views = []
        for _ in range(self.hparams.num_augmentations):
            if self.training and self.augmentation is not None:
                # Apply augmentation during training
                augmented_x = self.augmentation(x)
            else:
                # No augmentation during validation/test
                augmented_x = x
            augmented_views.append(augmented_x)
        
        # Stack views: [batch_size * num_augmentations, feature_dim]
        x_augmented = torch.cat(augmented_views, dim=0)
        
        # Repeat labels for all views: [y1, y2, ..., yN, y1, y2, ..., yN]
        labels_repeated = labels.repeat(self.hparams.num_augmentations)
        
        # Forward pass through encoder and projection head
        embeddings, projections, logits = self.forward(x_augmented)
        
        # Check for single-class batches
        unique_classes = torch.unique(labels).numel()
        is_single_class_batch = unique_classes < 2
        
        # DEBUG: Print on first batch
        debug_loss = (batch_idx == 0 and self.training)
        
        # Compute contrastive loss over all augmented views
        if is_single_class_batch and self.hparams.allow_single_class_batches:
            contrastive_loss = projections.new_zeros((), requires_grad=True)
            if debug_loss:
                print(f"\n[AugSupCon] Single-class batch detected. Returning zero loss.")
        else:
            contrastive_loss = self.contrastive_criterion(
                projections,
                labels_repeated,
                strict_multi_class=self.training,
                debug=debug_loss,
            )
        
        # Prepare return dictionary
        result = {
            "contrastive_loss": contrastive_loss,
            # Return only first view's embeddings/projections for logging
            "embeddings": embeddings[:batch_size],
            "projections": projections[:batch_size],
            "num_classes_in_batch": torch.tensor(unique_classes, device=labels.device),
            "is_single_class": torch.tensor(
                1.0 if is_single_class_batch else 0.0,
                device=labels.device,
            ),
        }
        
        # Optional classification loss (only on first view)
        if self.hparams.use_classification_head and logits is not None:
            logits_first_view = logits[:batch_size]
            classification_loss = self.classification_criterion(logits_first_view, labels)
            result["classification_loss"] = classification_loss
            result["logits"] = logits_first_view
            result["preds"] = logits_first_view.argmax(dim=-1)
        
        return result
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step: compute loss and update metrics.
        
        Args:
            batch: Tuple of (features, labels)
            batch_idx: Index of current batch
        
        Returns:
            Total loss (contrastive + optional classification)
        """
        x, labels = batch
        outputs = self.model_step(batch, batch_idx)
        
        # Total loss = contrastive + (optional) weighted classification
        loss = outputs["contrastive_loss"]
        
        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            
            # Update classification metrics
            self.train_classification_loss(outputs["classification_loss"])
            self.train_acc(outputs["preds"], labels)
            
            # Log classification metrics
            self.log("train/cls_loss", self.train_classification_loss, on_step=False, on_epoch=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update and log contrastive metrics
        self.train_contrastive_loss(outputs["contrastive_loss"])
        self.log("train/con_loss", self.train_contrastive_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        
        # Log embedding statistics and batch composition for monitoring
        if batch_idx == 0:
            self.log("debug/emb_mean", outputs["embeddings"].mean(), on_step=False, on_epoch=True)
            self.log("debug/emb_std", outputs["embeddings"].std(), on_step=False, on_epoch=True)
            self.log("debug/proj_std", outputs["projections"].std(), on_step=False, on_epoch=True)
        
        # Log batch composition
        self.log("debug/num_classes_in_batch", outputs["num_classes_in_batch"], on_step=False, on_epoch=True)
        self.log("debug/is_single_class", outputs["is_single_class"], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        """
        Validation step: compute metrics without augmentation.
        
        Note: During validation, we don't apply augmentation to get
        a fair assessment of the learned representations.
        
        Args:
            batch: Tuple of (features, labels)
            batch_idx: Index of current batch
        """
        x, labels = batch
        outputs = self.model_step(batch)
        
        # Total loss
        loss = outputs["contrastive_loss"]
        
        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            
            # Update classification metrics
            self.val_classification_loss(outputs["classification_loss"])
            self.val_acc(outputs["preds"], labels)
            
            # Log classification metrics
            self.log("val/cls_loss", self.val_classification_loss, on_step=False, on_epoch=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update and log contrastive metrics
        self.val_contrastive_loss(outputs["contrastive_loss"])
        self.log("val/con_loss", self.val_contrastive_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
    
    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Test step: evaluate final performance.
        
        Args:
            batch: Tuple of (features, labels)
            batch_idx: Index of current batch
        
        Returns:
            Dictionary with loss
        """
        x, labels = batch
        outputs = self.model_step(batch)
        
        # Total loss
        loss = outputs["contrastive_loss"]
        self.log("test/con_loss", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True)
        
        if self.hparams.use_classification_head and "classification_loss" in outputs:
            loss = loss + self.hparams.classification_weight * outputs["classification_loss"]
            preds = outputs["preds"]
            test_acc = (preds == labels).float().mean()
            
            self.log("test/cls_loss", outputs["classification_loss"], on_step=False, on_epoch=True)
            self.log("test/acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        
        return {"loss": loss}
    
    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation epoch.
        Updates best validation accuracy.
        """
        if self.hparams.use_classification_head:
            acc = self.val_acc.compute()
            self.val_acc_best(acc)
            self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
    
    def setup(self, stage: str) -> None:
        """
        Setup model components after datamodule is initialized.
        
        This is called by Lightning after datamodule.setup() and before training.
        We build the model here because we need information from the datamodule
        (feature_map, num_classes).
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        self._setup_calls += 1
        
        # Prevent accidental re-initialization
        if self._built:
            return
        
        # Get datamodule from trainer
        if not (self.trainer and getattr(self.trainer, "datamodule", None)):
            raise RuntimeError("Datamodule not available in setup(). Cannot initialize model.")
        
        dm = self.trainer.datamodule
        
        # Load feature map from preprocessed data directory
        eos_preproc_dir = getattr(dm, "paths", None)["eos_preproc_dir"]
        feature_map_path = os.path.join(eos_preproc_dir, "feature_map.json")
        
        if not os.path.exists(feature_map_path):
            raise FileNotFoundError(f"feature_map.json not found at {feature_map_path}")
        
        with open(feature_map_path, "r") as f:
            feature_map = json.load(f)
        
        # Get number of classes from datamodule
        num_classes = getattr(dm, "num_classes", None)
        if num_classes is None:
            raise RuntimeError("datamodule.num_classes is None.")
        
        # ============================================================
        # BUILD MODEL COMPONENTS
        # ============================================================
        
        # 1. Augmentation module
        self.augmentation = RandomMaskingAugmentation(
            feature_map=feature_map,
            mask_probability=self.hparams.mask_probability,
            mask_full_particle=self.hparams.mask_full_particle,
        )
        
        # 2. Encoder (TinyTransformer)
        self.encoder = TinyTransformer(
            feature_map=feature_map,
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            num_layers=self.hparams.num_layers,
            d_ff=self.hparams.d_ff,
            dropout=self.hparams.dropout,
            num_classes=num_classes,
        )
        
        # 3. Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=self.hparams.d_model,
            hidden_dim=self.hparams.hidden_projection_dim,
            output_dim=self.hparams.projection_dim,
        )
        
        # 4. Optional classification head for monitoring
        if self.hparams.use_classification_head:
            self.classification_head = nn.Linear(self.hparams.d_model, num_classes)
            
            # Update metrics with correct num_classes
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # 5. Optional: Compile encoder for faster training (PyTorch 2.0+)
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
        
        self._built = True
        
        print(f"✅ Model initialized:")
        print(f"   • Encoder: TinyTransformer ({self.hparams.d_model}d, {self.hparams.num_layers}L)")
        print(f"   • Projection: {self.hparams.d_model} → {self.hparams.projection_dim}")
        print(f"   • Augmentation: mask_p={self.hparams.mask_probability}, full_particle={self.hparams.mask_full_particle}")
        print(f"   • Temperature: {self.hparams.temperature}")
        print(f"   • Num Classes: {num_classes}")
        print(f"   • Classification Head: {self.hparams.use_classification_head}")
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and optional scheduler
        """
        # Collect all trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]
        
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found.")
        
        # Instantiate optimizer from hydra config
        optimizer = self.hparams.optimizer(params=params)
        
        # Optional: Add learning rate scheduler
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
        """
        Extract embeddings from encoder (for inference/downstream tasks).
        
        Args:
            x: Input features [batch_size, feature_dim]
        
        Returns:
            embeddings: [batch_size, d_model]
        """
        if self.encoder is None:
            raise RuntimeError("Encoder is not initialized.")
        return self.encoder.get_embeddings(x)
