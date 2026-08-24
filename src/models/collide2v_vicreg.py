"""
VICReg Unsupervised and VCReg Supervised modules for COLLIDE2V.

Two distinct modules:

1. COLLIDE2VVICRegLitModule  — Unsupervised VICReg (Bardes et al., ICLR 2022)
   - Two augmented views per event via random masking
   - Loss: invariance (MSE) + variance + covariance
   - Labels NOT used in the loss
   - Projection head discarded after training

2. COLLIDE2VVCRegLitModule  — Supervised VCReg (Zhu et al., arXiv 2306.13292)
   - Single view, labels via cross-entropy
   - Loss: CE + variance + covariance on final embedding only
   - No projection head, no intermediate layers

References:
    Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
        for Self-Supervised Learning", ICLR 2022. arXiv:2105.04906
    Zhu et al., "Variance-Covariance Regularization Improves Representation
        Learning", arXiv:2306.13292 (2023/2024)
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification.accuracy import Accuracy

try:
    from sklearn.metrics import silhouette_score
except ImportError:
    silhouette_score = None

from .components.transformer import TinyTransformer


# ---------------------------------------------------------------------------
# Shared: Augmentation
# ---------------------------------------------------------------------------

class RandomMaskingAugmentation(nn.Module):
    """
    Random masking augmentation for physics event data.
    Identical to the one used in Aug-SupCon.
    """

    def __init__(
        self,
        feature_map: Dict[str, Any],
        mask_probability: float = 0.2,
        mask_full_particle: bool = False,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.mask_probability = mask_probability
        self.mask_full_particle = mask_full_particle
        self.noise_std = noise_std

        self.group_configs = []
        for group_name, cfg in feature_map.items():
            topk = cfg["topk"]
            self.group_configs.append({
                "name": group_name,
                "start": cfg["start"],
                "num_particles": 1 if topk is None else topk,
                "particle_size": len(cfg["columns"]),
                "is_scalar": topk is None,
            })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, feature_dim = x.shape
        device = x.device
        augmented = x.clone()

        if self.mask_full_particle:
            for g in self.group_configs:
                if g["is_scalar"]:
                    continue
                particle_mask = (
                    torch.rand(batch_size, g["num_particles"], device=device)
                    < self.mask_probability
                )
                expanded = particle_mask.unsqueeze(-1).expand(-1, -1, g["particle_size"])
                flat = expanded.reshape(batch_size, -1)
                end_idx = g["start"] + g["num_particles"] * g["particle_size"]
                if self.noise_std > 0.0:
                    augmented[:, g["start"]:end_idx][flat] = torch.randn(flat.sum(), device=device) * self.noise_std
                else:
                    augmented[:, g["start"]:end_idx][flat] = 0.0
        else:
            mask = (
                torch.rand(batch_size, feature_dim, device=device) < self.mask_probability
            )
            if self.noise_std > 0.0:
                augmented[mask] = torch.randn(mask.sum(), device=device) * self.noise_std
            else:
                augmented[mask] = 0.0

        return augmented


# ---------------------------------------------------------------------------
# Shared: Projection Head (VICReg only)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head for VICReg.
    Discarded after training — only the encoder is kept.
    d_model → hidden_dim (LayerNorm + ReLU) → projection_dim
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss 1: VICReg — three terms (unsupervised)
# ---------------------------------------------------------------------------

class VICRegLoss(nn.Module):
    """
    VICReg loss: invariance + variance + covariance.

        L = λ * inv(z, z')
          + μ * [var(z) + var(z')] / 2
          + ν * [cov(z) + cov(z')] / 2

    Projections are NOT L2-normalised: invariance uses MSE, not cosine sim.

    Args:
        inv_weight (λ): paper default 25
        var_weight (μ): paper default 25
        cov_weight (ν): paper default  1

    Reference: Bardes et al., ICLR 2022. arXiv:2105.04906
    """

    def __init__(
        self,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
    ):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(
        self, z: torch.Tensor, z_prime: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z:       [N, D] — projections from view 1
            z_prime: [N, D] — projections from view 2
        Returns:
            dict with keys: loss, inv_loss, var_loss, cov_loss
        """
        N, D = z.shape

        inv_loss = F.mse_loss(z, z_prime)
        var_loss = (self._var(z) + self._var(z_prime)) / 2.0
        cov_loss = (self._cov(z, N, D) + self._cov(z_prime, N, D)) / 2.0

        loss = (
            self.inv_weight * inv_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )

        return {
            "loss":     loss,
            "inv_loss": inv_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
        }

    def _var(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge loss: penalise per-dim std below 1."""
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return F.relu(1.0 - std).mean()

    def _cov(self, z: torch.Tensor, N: int, D: int) -> torch.Tensor:
        """Off-diagonal covariance penalty, normalised by D."""
        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / (N - 1)
        # pow(2) not pow_(2) — no in-place on view (would corrupt gradients)
        return self._off_diagonal(cov).pow(2).sum().div(D)

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------------------------------------------------------------------------
# Loss 2: VCReg — two terms (supervised, no invariance)
# ---------------------------------------------------------------------------

class VCRegLoss(nn.Module):
    """
    VCReg loss: variance + covariance only. No invariance term, no two views.

        L_reg = α * var(h) + β * cov(h)

    Uses 1/D(D-1) normalisation for covariance (correct over off-diagonal count).

    Args:
        var_weight (α): paper default 25
        cov_weight (β): paper default  1

    Reference: Zhu et al., arXiv:2306.13292 (2023/2024)
    """

    def __init__(self, var_weight: float = 25.0, cov_weight: float = 1.0):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: [N, D] — final encoder embedding
        Returns:
            dict with keys: loss, var_loss, cov_loss
        """
        N, D = h.shape

        var_loss = self._var(h)
        cov_loss = self._cov(h, N, D)
        loss = self.var_weight * var_loss + self.cov_weight * cov_loss

        return {
            "loss":     loss,
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
        }

    def _var(self, h: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(h.var(dim=0) + 1e-4)
        return F.relu(1.0 - std).mean()

    def _cov(self, h: torch.Tensor, N: int, D: int) -> torch.Tensor:
        h_c = h - h.mean(dim=0)
        cov = (h_c.T @ h_c) / (N - 1)
        norm = D * (D - 1) if D > 1 else 1
        return self._off_diagonal(cov).pow(2).sum().div(norm)

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------------------------------------------------------------------------
# Shared silhouette helper
# ---------------------------------------------------------------------------

def _compute_and_log_silhouette(
    module: LightningModule,
    emb_cache: List[torch.Tensor],
    label_cache: List[torch.Tensor],
    max_samples: int,
    warned_flag_name: str = "_silhouette_warned",
    tag: str = "",
) -> None:
    if not emb_cache or silhouette_score is None:
        return
    embeddings = torch.cat(emb_cache, dim=0).numpy()
    labels     = torch.cat(label_cache, dim=0).numpy()
    if embeddings.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(embeddings.shape[0], size=max_samples, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]
    unique = np.unique(labels)
    if 2 <= unique.size < embeddings.shape[0]:
        try:
            sil = float(silhouette_score(embeddings, labels, metric="cosine"))
            module.log("val/silhouette", sil, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            if not getattr(module, warned_flag_name, False):
                print(f"[{tag}] Silhouette error: {e}")
                setattr(module, warned_flag_name, True)


# ===========================================================================
# Module 1: VICReg Unsupervised
# ===========================================================================

class COLLIDE2VVICRegLitModule(LightningModule):
    """
    Unsupervised VICReg for COLLIDE2V.

    Training:
        1. Two random-masked views per event
        2. Both through encoder + projection head (shared weights)
        3. VICReg loss: invariance + variance + covariance
        4. Labels NOT used in the VICReg loss
        5. Optional classification head for monitoring only

    After training: discard projection head, use get_embeddings().

    Reference: Bardes et al., ICLR 2022. arXiv:2105.04906
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        # Projection head
        projection_dim: int = 128,
        hidden_projection_dim: int = 256,
        # VICReg loss weights (paper defaults)
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        # Augmentation
        mask_probability: float = 0.2,
        mask_full_particle: bool = False,
        mask_noise_std: float = 0.0,
        augmentation_type: str = "random",   # "random" | "physics"
        eta_sigma: float = 1.5,
        pt_sigma: float = 0.1,
        # Monitoring
        use_classification_head: bool = True,
        classification_weight: float = 0.1,
        silhouette_max_samples: int = 20000,
        # Training
        optimizer: Any = None,
        scheduler: Optional[Any] = None,
        warmup_epochs: int = 0,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder          = None
        self.projection_head  = None
        self.classification_head = None
        self.augmentation     = None
        self._built           = False
        self._silhouette_warned = False

        self._val_emb_cache: List[torch.Tensor]   = []
        self._val_label_cache: List[torch.Tensor] = []
        self._warmup_base_lrs: Optional[List[float]] = None

        self.vicreg_criterion = VICRegLoss(inv_weight, var_weight, cov_weight)
        self.classification_criterion = (
            nn.CrossEntropyLoss() if use_classification_head else None
        )

        # Metrics
        self.train_vicreg_loss = MeanMetric()
        self.train_inv_loss    = MeanMetric()
        self.train_var_loss    = MeanMetric()
        self.train_cov_loss    = MeanMetric()

        self.val_vicreg_loss   = MeanMetric()
        self.val_inv_loss      = MeanMetric()
        self.val_var_loss      = MeanMetric()
        self.val_cov_loss      = MeanMetric()

        if use_classification_head:
            self.train_classification_loss = MeanMetric()
            self.val_classification_loss   = MeanMetric()
            self.val_acc_best              = MaxMetric()
            # train_acc / val_acc built in _build_model_components

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            embeddings:  [B, d_model]
            projections: [B, projection_dim]  — NOT L2-normalised (VICReg uses MSE)
            logits:      [B, num_classes] or None
        """
        embeddings  = self.encoder.get_embeddings(x)
        projections = self.projection_head(embeddings)
        logits = None
        if self.hparams.use_classification_head and self.classification_head is not None:
            logits = self.classification_head(embeddings)
        return embeddings, projections, logits

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int = 0,
        apply_augmentation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        x, labels = batch

        if apply_augmentation and self.augmentation is not None:
            x1 = self.augmentation(x)
            x2 = self.augmentation(x)
            emb1, proj1, logits1 = self.forward(x1)
            _,    proj2, _       = self.forward(x2)
        else:
            # Single forward pass — proj2 is the same tensor, inv_loss == 0
            emb1, proj1, logits1 = self.forward(x)
            proj2 = proj1

        vicreg_out = self.vicreg_criterion(proj1, proj2)

        result = {
            "vicreg_loss": vicreg_out["loss"],
            "inv_loss":    vicreg_out["inv_loss"],
            "var_loss":    vicreg_out["var_loss"],
            "cov_loss":    vicreg_out["cov_loss"],
            "embeddings":  emb1.detach(),
        }

        if self.hparams.use_classification_head and logits1 is not None:
            result["classification_loss"] = self.classification_criterion(logits1, labels)
            result["preds"] = logits1.argmax(dim=-1)

        return result

    # ------------------------------------------------------------------
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, labels = batch
        out = self.model_step(batch, batch_idx, apply_augmentation=True)

        loss = out["vicreg_loss"]
        if self.hparams.use_classification_head and "classification_loss" in out:
            loss = loss + self.hparams.classification_weight * out["classification_loss"]
            self.train_classification_loss(out["classification_loss"])
            self.train_acc(out["preds"], labels)
            self.log("train/cls_loss", self.train_classification_loss, on_step=False, on_epoch=True)
            self.log("train/acc",      self.train_acc,                 on_step=False, on_epoch=True, prog_bar=True)

        self.train_vicreg_loss(out["vicreg_loss"])
        self.train_inv_loss(out["inv_loss"])
        self.train_var_loss(out["var_loss"])
        self.train_cov_loss(out["cov_loss"])

        self.log("train/vicreg_loss", self.train_vicreg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/inv_loss",    self.train_inv_loss,    on_step=False, on_epoch=True)
        self.log("train/var_loss",    self.train_var_loss,    on_step=False, on_epoch=True)
        self.log("train/cov_loss",    self.train_cov_loss,    on_step=False, on_epoch=True)
        self.log("train/loss",        loss,                   on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, labels = batch
        out = self.model_step(batch, batch_idx, apply_augmentation=False)

        loss = out["vicreg_loss"]
        if self.hparams.use_classification_head and "classification_loss" in out:
            loss = loss + self.hparams.classification_weight * out["classification_loss"]
            self.val_classification_loss(out["classification_loss"])
            self.val_acc(out["preds"], labels)
            self.log("val/cls_loss", self.val_classification_loss, on_step=False, on_epoch=True)
            self.log("val/acc",      self.val_acc,                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_vicreg_loss(out["vicreg_loss"])
        self.val_inv_loss(out["inv_loss"])
        self.val_var_loss(out["var_loss"])
        self.val_cov_loss(out["cov_loss"])

        self.log("val/vicreg_loss", self.val_vicreg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/inv_loss",    self.val_inv_loss,    on_step=False, on_epoch=True)
        self.log("val/var_loss",    self.val_var_loss,    on_step=False, on_epoch=True)
        self.log("val/cov_loss",    self.val_cov_loss,    on_step=False, on_epoch=True)
        self.log("val/loss",        loss,                 on_step=False, on_epoch=True)

        self._val_emb_cache.append(out["embeddings"].cpu())
        self._val_label_cache.append(labels.cpu())

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, labels = batch
        out = self.model_step(batch, batch_idx, apply_augmentation=False)
        self.log("test/vicreg_loss", out["vicreg_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss",        out["vicreg_loss"], on_step=False, on_epoch=True)
        return {"loss": out["vicreg_loss"]}

    def on_validation_epoch_end(self) -> None:
        _compute_and_log_silhouette(
            self, self._val_emb_cache, self._val_label_cache,
            int(self.hparams.silhouette_max_samples), tag="VICReg",
        )
        self._val_emb_cache.clear()
        self._val_label_cache.clear()

        if self.hparams.use_classification_head and hasattr(self, "val_acc"):
            acc = self.val_acc.compute()
            self.val_acc_best(acc)
            self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    # ------------------------------------------------------------------
    def _build_model_components(
        self,
        feature_map: Dict[str, Any],
        num_classes: int,
        stage: Optional[str] = None,
    ) -> None:
        if self.hparams.augmentation_type == "physics":
            from src.data.augmentations import PhysicsAugmentation
            _trainer = getattr(self, "_trainer", None)
            _dm = getattr(_trainer, "datamodule", None) if _trainer is not None else None
            if _dm is not None:
                preproc_dir = _dm.paths["eos_preproc_dir"]
                self.augmentation = PhysicsAugmentation(
                    feature_map_path=os.path.join(preproc_dir, "feature_map.json"),
                    norm_stats_path=os.path.join(preproc_dir, "norm_stats.json"),
                    eta_sigma=self.hparams.eta_sigma,
                    pt_sigma=self.hparams.pt_sigma,
                )
            else:
                self.augmentation = nn.Identity()
        else:
            self.augmentation = RandomMaskingAugmentation(
                feature_map=feature_map,
                mask_probability=self.hparams.mask_probability,
                mask_full_particle=self.hparams.mask_full_particle,
                noise_std=self.hparams.mask_noise_std,
            )
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
            self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)

        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)

        self._built = True
        print(f"✅ [VICReg Unsupervised] Model initialised:")
        print(f"   • Encoder: TinyTransformer ({self.hparams.d_model}d, {self.hparams.num_layers}L)")
        print(f"   • Projection: {self.hparams.d_model} → {self.hparams.projection_dim}")
        print(f"   • Augmentation: mask_p={self.hparams.mask_probability}, full_particle={self.hparams.mask_full_particle}")
        print(f"   • VICReg weights: inv={self.hparams.inv_weight}, var={self.hparams.var_weight}, cov={self.hparams.cov_weight}")
        print(f"   • Classification head: {self.hparams.use_classification_head} (monitoring only)")

    def setup(self, stage: str) -> None:
        if self._built:
            return
        if not (self.trainer and getattr(self.trainer, "datamodule", None)):
            raise RuntimeError("Datamodule not available in setup().")
        dm = self.trainer.datamodule
        eos_preproc_dir = getattr(dm, "paths", {})["eos_preproc_dir"]
        with open(os.path.join(eos_preproc_dir, "feature_map.json")) as f:
            feature_map = json.load(f)
        num_classes = getattr(dm, "num_classes", None)
        if num_classes is None:
            raise RuntimeError("datamodule.num_classes is None.")
        self._build_model_components(feature_map, num_classes, stage)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self._built:
            return
        dm_hparams = checkpoint.get("datamodule_hyper_parameters", {})
        eos_preproc_dir = dm_hparams.get("paths", {}).get("eos_preproc_dir")
        with open(os.path.join(eos_preproc_dir, "feature_map.json")) as f:
            feature_map = json.load(f)
        num_classes = len(dm_hparams.get("to_classify", []))
        if num_classes == 0:
            raise RuntimeError("to_classify is empty in checkpoint.")
        self._build_model_components(feature_map, num_classes)

    def configure_optimizers(self) -> Dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=params)
        self._warmup_base_lrs = [g["lr"] for g in optimizer.param_groups]
        if self.hparams.scheduler is not None:
            if isinstance(self.hparams.scheduler, functools.partial):
                scheduler_class = self.hparams.scheduler.func
                scheduler_kwargs = {
                    k: v for k, v in self.hparams.scheduler.keywords.items()
                    if v is not None
                }
                scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            cfg = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                cfg["monitor"] = "val/vicreg_loss"
            return {"optimizer": optimizer, "lr_scheduler": cfg}
        return {"optimizer": optimizer}

    def on_train_epoch_start(self) -> None:
        """Linear LR warmup over the first `warmup_epochs` epochs.

        Runs before the epoch's optimizer steps and overrides whatever the
        scheduler set, so it only has effect while `current_epoch <
        warmup_epochs`; afterwards the configured scheduler (e.g.
        ReduceLROnPlateau) takes over uncontested.
        """
        warmup_epochs = getattr(self.hparams, "warmup_epochs", 0)
        if not warmup_epochs or self._warmup_base_lrs is None:
            return
        if self.current_epoch >= warmup_epochs:
            return
        scale = (self.current_epoch + 1) / warmup_epochs
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
            group["lr"] = base_lr * scale

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Encoder not initialised.")
        return self.encoder.get_embeddings(x)


# ===========================================================================
# Module 2: VCReg Supervised
# ===========================================================================

class COLLIDE2VVCRegLitModule(LightningModule):
    """
    Supervised VCReg for COLLIDE2V.

    Training:
        1. Single forward pass — no augmentation, no two views
        2. CE loss for class supervision
        3. Variance + covariance regularisation on the final embedding

    Total loss:
        L = L_CE + α * var(h) + β * cov(h)

    No projection head. No intermediate layers.
    CE handles inter-class separation.
    VCReg handles intra-class diversity and dimension decorrelation.

    Reference: Zhu et al., arXiv:2306.13292 (2023/2024)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        silhouette_max_samples: int = 20000,
        optimizer: Any = None,
        scheduler: Optional[Any] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder             = None
        self.classification_head = None
        self._built              = False
        self._silhouette_warned  = False

        self._val_emb_cache: List[torch.Tensor]   = []
        self._val_label_cache: List[torch.Tensor] = []

        self.vcreg_criterion          = VCRegLoss(var_weight, cov_weight)
        self.classification_criterion = nn.CrossEntropyLoss()

        self.train_ce_loss    = MeanMetric()
        self.train_vcreg_loss = MeanMetric()
        self.train_var_loss   = MeanMetric()
        self.train_cov_loss   = MeanMetric()

        self.val_ce_loss      = MeanMetric()
        self.val_vcreg_loss   = MeanMetric()
        self.val_var_loss     = MeanMetric()
        self.val_cov_loss     = MeanMetric()

        self.val_acc_best     = MaxMetric()
        # train_acc / val_acc / test_acc built in _build_model_components

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embeddings: [B, d_model]
            logits:     [B, num_classes]
        """
        embeddings = self.encoder.get_embeddings(x)
        logits     = self.classification_head(embeddings)
        return embeddings, logits

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        x, labels = batch

        embeddings, logits = self.forward(x)

        ce_loss   = self.classification_criterion(logits, labels)
        vcreg_out = self.vcreg_criterion(embeddings)

        loss = ce_loss + vcreg_out["loss"]

        return {
            "loss":       loss,
            "ce_loss":    ce_loss,
            "vcreg_loss": vcreg_out["loss"],
            "var_loss":   vcreg_out["var_loss"],
            "cov_loss":   vcreg_out["cov_loss"],
            "preds":      logits.argmax(dim=-1),
            "embeddings": embeddings.detach(),
        }

    # ------------------------------------------------------------------
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, labels = batch
        out = self.model_step(batch, batch_idx)

        self.train_ce_loss(out["ce_loss"])
        self.train_vcreg_loss(out["vcreg_loss"])
        self.train_var_loss(out["var_loss"])
        self.train_cov_loss(out["cov_loss"])
        self.train_acc(out["preds"], labels)

        self.log("train/ce_loss",    self.train_ce_loss,    on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/vcreg_loss", self.train_vcreg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/var_loss",   self.train_var_loss,   on_step=False, on_epoch=True)
        self.log("train/cov_loss",   self.train_cov_loss,   on_step=False, on_epoch=True)
        self.log("train/acc",        self.train_acc,        on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss",       out["loss"],           on_step=False, on_epoch=True)
        return out["loss"]

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, labels = batch
        out = self.model_step(batch, batch_idx)

        self.val_ce_loss(out["ce_loss"])
        self.val_vcreg_loss(out["vcreg_loss"])
        self.val_var_loss(out["var_loss"])
        self.val_cov_loss(out["cov_loss"])
        self.val_acc(out["preds"], labels)

        self.log("val/ce_loss",    self.val_ce_loss,    on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/vcreg_loss", self.val_vcreg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/var_loss",   self.val_var_loss,   on_step=False, on_epoch=True)
        self.log("val/cov_loss",   self.val_cov_loss,   on_step=False, on_epoch=True)
        self.log("val/acc",        self.val_acc,        on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss",       out["loss"],         on_step=False, on_epoch=True)

        self._val_emb_cache.append(out["embeddings"].cpu())
        self._val_label_cache.append(labels.cpu())

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, labels = batch
        out = self.model_step(batch, batch_idx)

        self.test_acc(out["preds"], labels)

        self.log("test/ce_loss",    out["ce_loss"],    on_step=False, on_epoch=True)
        self.log("test/vcreg_loss", out["vcreg_loss"], on_step=False, on_epoch=True)
        self.log("test/acc",        self.test_acc,     on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss",       out["loss"],       on_step=False, on_epoch=True)
        return {"loss": out["loss"]}

    def on_validation_epoch_end(self) -> None:
        _compute_and_log_silhouette(
            self, self._val_emb_cache, self._val_label_cache,
            int(self.hparams.silhouette_max_samples), tag="VCReg",
        )
        self._val_emb_cache.clear()
        self._val_label_cache.clear()

        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    # ------------------------------------------------------------------
    def _build_model_components(
        self,
        feature_map: Dict[str, Any],
        num_classes: int,
        stage: Optional[str] = None,
    ) -> None:
        self.encoder = TinyTransformer(
            feature_map=feature_map,
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            num_layers=self.hparams.num_layers,
            d_ff=self.hparams.d_ff,
            dropout=self.hparams.dropout,
            num_classes=num_classes,
        )
        self.classification_head = nn.Linear(self.hparams.d_model, num_classes)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes)

        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)

        self._built = True
        print(f"✅ [VCReg Supervised] Model initialised:")
        print(f"   • Encoder: TinyTransformer ({self.hparams.d_model}d, {self.hparams.num_layers}L)")
        print(f"   • VCReg weights: var={self.hparams.var_weight}, cov={self.hparams.cov_weight}")
        print(f"   • Num classes: {num_classes}")

    def setup(self, stage: str) -> None:
        if self._built:
            return
        if not (self.trainer and getattr(self.trainer, "datamodule", None)):
            raise RuntimeError("Datamodule not available in setup().")
        dm = self.trainer.datamodule
        eos_preproc_dir = getattr(dm, "paths", {})["eos_preproc_dir"]
        with open(os.path.join(eos_preproc_dir, "feature_map.json")) as f:
            feature_map = json.load(f)
        num_classes = getattr(dm, "num_classes", None)
        if num_classes is None:
            raise RuntimeError("datamodule.num_classes is None.")
        self._build_model_components(feature_map, num_classes, stage)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self._built:
            return
        dm_hparams = checkpoint.get("datamodule_hyper_parameters", {})
        eos_preproc_dir = dm_hparams.get("paths", {}).get("eos_preproc_dir")
        with open(os.path.join(eos_preproc_dir, "feature_map.json")) as f:
            feature_map = json.load(f)
        num_classes = len(dm_hparams.get("to_classify", []))
        if num_classes == 0:
            raise RuntimeError("to_classify is empty in checkpoint.")
        self._build_model_components(feature_map, num_classes)

    def configure_optimizers(self) -> Dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            if isinstance(self.hparams.scheduler, functools.partial):
                scheduler_class = self.hparams.scheduler.func
                scheduler_kwargs = {
                    k: v for k, v in self.hparams.scheduler.keywords.items()
                    if v is not None
                }
                scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            cfg = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                cfg["monitor"] = "val/ce_loss"
            return {"optimizer": optimizer, "lr_scheduler": cfg}
        return {"optimizer": optimizer}

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Encoder not initialised.")
        return self.encoder.get_embeddings(x)
