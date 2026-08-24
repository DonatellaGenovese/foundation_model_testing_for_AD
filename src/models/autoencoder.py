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

from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from collections import defaultdict

log = logging.getLogger(__name__)

TARGET_FPRS: list = [0.01, 0.05, 0.10]

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@torch.no_grad()
def compute_errors_numpy(ae_model, X: np.ndarray,
                          batch_size: int = 1024,
                          device: str = "cpu") -> np.ndarray:
    """Per-sample MSE reconstruction errors for a numpy array of inputs."""
    ae_model = ae_model.to(device).eval()
    parts = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
        recon, _ = ae_model(batch)
        parts.append(((recon - batch) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def derive_thresholds(qcd_errors: np.ndarray, target_fprs: list = None) -> dict:
    """FPR → threshold map derived from QCD reconstruction errors."""
    fprs = target_fprs if target_fprs is not None else TARGET_FPRS
    return {fpr: float(np.quantile(qcd_errors, 1.0 - fpr)) for fpr in fprs}


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

        bottleneck_dim = max(4, input_dim // compression)

        hidden_dims = []
        current_dim = input_dim

        for _ in range(depth):
            next_dim = max(bottleneck_dim, current_dim // 2)
            hidden_dims.append(next_dim)
            current_dim = next_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))

        self.encoder = nn.Sequential(*encoder_layers)

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

    Validation monitoring modes (controlled via ModelCheckpoint monitor):
        - "ae/val_drift_fpr01/05/10": unsupervised drift (Strategy 1)
        - "ae/val_auroc": AUROC on val signals (Strategies 2 & 3)
        - "ae/separation_ratio": MSE ratio (can also be used)

    Args:
        val_signal_classes: which anomaly classes to include in val monitoring.
            None  → inherit from anomaly_classes_labels (all signals)
            []    → no signals in val (pure unsupervised, Strategy 1)
            [17]  → only ggHtautau in val (Strategy 3)
            [15,16,17] → all signals in val (Strategy 2)
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
        val_signal_classes: Optional[List[int]] = None,
        score_classes: Optional[List[int]] = None,
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
        self.val_loss_score = MeanMetric()

        # Reconstruction error accumulation (shared by val and test)
        self.val_errors_normal = []
        self.val_errors_anomaly = []
        self.val_labels = []

        # Per-class error tracking (separate for val and test)
        self._val_errors_by_class: Dict[int, List[float]] = defaultdict(list)
        self._test_errors_by_class: Dict[int, List[float]] = defaultdict(list)

        # Comprehensive test metrics (populated in on_test_epoch_end)
        self._test_metrics: Dict[str, Any] = {}

        # Separate error accumulator for score_classes (used for threshold/AUROC
        # when score_classes differs from normal_classes_labels, e.g. allsm + qcd-only scoring)
        self._val_errors_score: List[float] = []

        # Drift metric: thresholds set on val, measured on test
        self._target_fpr_values: List[float] = [0.01, 0.05, 0.10]
        self._val_thresholds: Dict[float, float] = {}
        self._bc_rate: int = 0

        # Val-split drift (HPO metric, no test leakage)
        self._val_drift_metric: float = float("nan")
        self._val_drift_per_fpr: Dict[float, float] = {}

        # Resolve which signals are monitored during validation
        if val_signal_classes is None:
            self._eff_val_signal_classes: List[int] = list(anomaly_classes_labels or [])
        else:
            self._eff_val_signal_classes = list(val_signal_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def compute_reconstruction_error(self, x: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
        reconstruction, _ = self.forward(x)
        if reduction == 'none':
            return torch.mean((x - reconstruction) ** 2, dim=1)
        return torch.mean((x - reconstruction) ** 2)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features, labels = batch
        normal_mask = torch.isin(labels, torch.tensor(self.hparams.normal_classes_labels, device=labels.device))

        if not normal_mask.any():
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        features_normal = features[normal_mask]
        reconstruction, _ = self.forward(features_normal)
        loss = self.criterion(reconstruction, features_normal)

        self.train_loss(loss)
        self.log("ae/train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        features, labels = batch
        errors = self.compute_reconstruction_error(features, reduction='none')

        normal_mask = torch.isin(labels, torch.tensor(self.hparams.normal_classes_labels, device=labels.device))
        if normal_mask.any():
            ne = errors[normal_mask]
            self.val_loss_normal(ne.mean())
            self.val_errors_normal.extend(ne.cpu().numpy())

        score_classes = self.hparams.get("score_classes", None)
        if score_classes is not None:
            score_mask = torch.isin(labels, torch.tensor(score_classes, device=labels.device))
            if score_mask.any():
                se = errors[score_mask]
                self.val_loss_score(se.mean())
                self._val_errors_score.extend(se.cpu().numpy())

        # Track per-class errors for the signals monitored in val
        for cls in self._eff_val_signal_classes:
            mask = (labels == cls)
            if mask.any():
                self._val_errors_by_class[cls].extend(errors[mask].cpu().numpy())

        # Aggregate anomaly errors for separation_ratio (legacy)
        if self._eff_val_signal_classes:
            sig_t = torch.tensor(self._eff_val_signal_classes, device=labels.device)
            anomaly_mask = torch.isin(labels, sig_t)
            if anomaly_mask.any():
                ae = errors[anomaly_mask]
                self.val_loss_anomaly(ae.mean())
                self.val_errors_anomaly.extend(ae.cpu().numpy())

        tracked_mask = normal_mask.clone()
        if self._eff_val_signal_classes:
            sig_t = torch.tensor(self._eff_val_signal_classes, device=labels.device)
            tracked_mask = tracked_mask | torch.isin(labels, sig_t)
        self.val_labels.extend(labels[tracked_mask].detach().cpu().numpy())

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """At test time, track ALL anomaly_classes (not just val_signal_classes)."""
        features, labels = batch
        errors = self.compute_reconstruction_error(features, reduction='none')

        normal_mask = torch.isin(labels, torch.tensor(self.hparams.normal_classes_labels, device=labels.device))
        if normal_mask.any():
            ne = errors[normal_mask]
            self.val_loss_normal(ne.mean())
            self.val_errors_normal.extend(ne.cpu().numpy())

        score_classes = self.hparams.get("score_classes", None)
        if score_classes is not None:
            score_mask = torch.isin(labels, torch.tensor(score_classes, device=labels.device))
            if score_mask.any():
                se = errors[score_mask]
                self._val_errors_score.extend(se.cpu().numpy())

        all_anomaly = self.hparams.anomaly_classes_labels or []
        for cls in all_anomaly:
            mask = (labels == cls)
            if mask.any():
                self._test_errors_by_class[cls].extend(errors[mask].cpu().numpy())

        if all_anomaly:
            anomaly_mask = torch.isin(labels, torch.tensor(all_anomaly, device=labels.device))
            if anomaly_mask.any():
                ae = errors[anomaly_mask]
                self.val_loss_anomaly(ae.mean())
                self.val_errors_anomaly.extend(ae.cpu().numpy())

        tracked = normal_mask.clone()
        if all_anomaly:
            tracked = tracked | torch.isin(labels, torch.tensor(all_anomaly, device=labels.device))
        self.val_labels.extend(labels[tracked].detach().cpu().numpy())

    def on_validation_epoch_end(self):
        normal_loss = self.val_loss_normal.compute()
        anomaly_loss = self.val_loss_anomaly.compute()

        self.log("ae/val_loss_normal", normal_loss, prog_bar=True)
        self.log("ae/val_loss_anomaly", anomaly_loss, prog_bar=True)

        if normal_loss > 0:
            separation_ratio = anomaly_loss / normal_loss
            self.log("ae/separation_ratio", separation_ratio, prog_bar=True)

        # When score_classes is set, log its MSE and use it for thresholds/drift
        score_classes = self.hparams.get("score_classes", None)
        if score_classes is not None and self._val_errors_score:
            score_loss = self.val_loss_score.compute()
            self.log("ae/val_loss_score", score_loss, prog_bar=True)

        # Thresholds for drift — use score_classes errors if available, else all normal
        _threshold_errors = self._val_errors_score if (score_classes and self._val_errors_score) else self.val_errors_normal
        if _threshold_errors:
            errors = np.array(_threshold_errors)
            self._bc_rate = len(errors)
            self._val_thresholds = {
                fpr: float(np.quantile(errors, 1.0 - fpr))
                for fpr in self._target_fpr_values
            }

        # Val-split drift (HPO metric, no test leakage)
        is_testing = self.trainer is not None and self.trainer.testing
        if not is_testing and _threshold_errors and len(_threshold_errors) >= 20:
            errors = np.array(_threshold_errors)
            half = len(errors) // 2
            threshold_errors = errors[:half]
            holdout_errors   = errors[half:]
            N   = len(holdout_errors)
            eps = 0.5 / N
            drift_values = []
            for fpr in self._target_fpr_values:
                thr   = float(np.quantile(threshold_errors, 1.0 - fpr))
                p_hat = float(np.sum(holdout_errors > thr)) / N
                d = abs(math.log((p_hat + eps) / (fpr + eps)))
                drift_values.append(d)
                self._val_drift_per_fpr[fpr] = d
                self.log(f"ae/val_drift_fpr{int(fpr*100):02d}", d, prog_bar=False)
            self._val_drift_metric = float(np.mean(drift_values))
            self.log("ae/val_drift_metric", self._val_drift_metric, prog_bar=True)

        # Val AUROC (only if signals are present in val monitoring)
        _normal_for_auroc = self._val_errors_score if (score_classes and self._val_errors_score) else self.val_errors_normal
        if not is_testing and self._val_errors_by_class and _normal_for_auroc and _HAS_SKLEARN:
            qcd_arr = np.array(_normal_for_auroc)
            sig_errors = []
            for cls in sorted(self._val_errors_by_class.keys()):
                sig_errors.extend(self._val_errors_by_class[cls])
            if sig_errors:
                y_true  = np.array([0] * len(qcd_arr) + [1] * len(sig_errors))
                y_score = np.concatenate([qcd_arr, np.array(sig_errors)])
                try:
                    val_auroc = float(_sklearn_roc_auc_score(y_true, y_score))
                    self.log("ae/val_auroc", val_auroc, prog_bar=True)
                except Exception:
                    pass

            # Per-signal val AUROC
            for cls, cls_errs in self._val_errors_by_class.items():
                if not cls_errs:
                    continue
                cls_arr = np.array(cls_errs)
                y_true  = np.array([0] * len(qcd_arr) + [1] * len(cls_arr))
                y_score = np.concatenate([qcd_arr, cls_arr])
                try:
                    per_auroc = float(_sklearn_roc_auc_score(y_true, y_score))
                    self.log(f"ae/val_auroc_cls{cls}", per_auroc)
                except Exception:
                    pass

        # Reset
        self._val_errors_by_class.clear()
        self.val_errors_normal.clear()
        self.val_errors_anomaly.clear()
        self.val_labels.clear()
        self._val_errors_score.clear()
        self.val_loss_normal.reset()
        self.val_loss_anomaly.reset()
        self.val_loss_score.reset()

    def on_test_epoch_end(self):
        score_classes = self.hparams.get("score_classes", None)
        qcd_errors = np.array(self._val_errors_score if (score_classes and self._val_errors_score) else self.val_errors_normal)
        anomaly_classes = self.hparams.anomaly_classes_labels or []

        # ── Per-class MSE summary ─────────────────────────────────────────────
        errors_by_class = defaultdict(list)
        for e, lbl in zip(
            self.val_errors_normal + self.val_errors_anomaly, self.val_labels
        ):
            errors_by_class[lbl].append(e)

        log.info("\n" + "="*80)
        log.info("RECONSTRUCTION MSE PER CLASS (Test Set)")
        log.info("="*80)
        log.info(f"{'Class':<20} {'Count':<10} {'Mean MSE':<15} {'Std MSE':<15}")
        log.info("-"*80)
        for cls in sorted(errors_by_class.keys()):
            errs = errors_by_class[cls]
            marker = (
                "(NORMAL)" if cls in self.hparams.normal_classes_labels
                else "(ANOMALY)" if anomaly_classes and cls in anomaly_classes
                else ""
            )
            log.info(
                f"Class {cls:<15} {len(errs):<10} "
                f"{np.mean(errs):<15.6f} {np.std(errs):<15.6f} {marker}"
            )
        log.info("="*80)

        # ── Drift metrics (thresholds from val, measured on test QCD) ────────
        drift_metrics = self._compute_drift_metrics(qcd_errors)
        if drift_metrics:
            log.info("DRIFT METRICS (threshold set on val, measured on test)")
            log.info("-"*80)
            for fpr, drift in sorted(drift_metrics.items()):
                fpr_tag = f"fpr{int(fpr*100):02d}"
                log.info(f"  FPR {fpr*100:.0f}%: drift = {drift:.4f}")
                self.log(f"ae/drift_{fpr_tag}", drift, prog_bar=False)
            avg_drift = float(np.mean(list(drift_metrics.values())))
            log.info(f"  Average drift: {avg_drift:.4f}")
            log.info("="*80)
            self.log("ae/drift_metric", avg_drift, prog_bar=True)

        # ── Per-signal comprehensive metrics ─────────────────────────────────
        test_metrics: Dict[str, Any] = {}

        # QCD: MSE stats + sample count
        normal_cls = self.hparams.normal_classes_labels[0] if self.hparams.normal_classes_labels else 0
        if len(qcd_errors) > 0:
            qcd_mean = float(np.mean(qcd_errors))
            test_metrics[f"mse_mean_cls{normal_cls}"] = qcd_mean
            test_metrics[f"mse_std_cls{normal_cls}"]  = float(np.std(qcd_errors))
            test_metrics[f"n_cls{normal_cls}"]        = len(qcd_errors)
        else:
            qcd_mean = float("nan")

        # Thresholds and measured FPR on test QCD (one entry per FPR working point)
        for fpr_target, threshold in self._val_thresholds.items():
            fpr_tag = f"fpr{int(fpr_target*100):02d}"
            test_metrics[f"threshold_{fpr_tag}"] = float(threshold)
            if len(qcd_errors) > 0:
                p_hat = float(np.mean(qcd_errors > threshold))
                test_metrics[f"fpr_measured_{fpr_tag}"] = p_hat

        # Per-signal: MSE stats + sample count (no sklearn needed)
        for cls in sorted(self._test_errors_by_class.keys()):
            signal_errors = np.array(self._test_errors_by_class[cls])
            if len(signal_errors) == 0:
                continue
            cls_tag = f"cls{cls}"
            test_metrics[f"mse_mean_{cls_tag}"] = float(np.mean(signal_errors))
            test_metrics[f"mse_std_{cls_tag}"]  = float(np.std(signal_errors))
            test_metrics[f"n_{cls_tag}"]        = len(signal_errors)

            # TPR and FNR at each FPR working point
            for fpr_target, threshold in self._val_thresholds.items():
                fpr_tag = f"fpr{int(fpr_target*100):02d}"
                tpr = float(np.mean(signal_errors > threshold))
                fnr = 1.0 - tpr
                self.log(f"ae/test_tpr_{fpr_tag}_{cls_tag}", tpr)
                self.log(f"ae/test_fnr_{fpr_tag}_{cls_tag}", fnr)
                test_metrics[f"tpr_{fpr_tag}_{cls_tag}"] = tpr
                test_metrics[f"fnr_{fpr_tag}_{cls_tag}"] = fnr

            # Separation ratio
            sep = float(np.mean(signal_errors) / qcd_mean) if qcd_mean > 0 else float("nan")
            self.log(f"ae/test_sep_ratio_{cls_tag}", sep)
            test_metrics[f"sep_ratio_{cls_tag}"] = sep

        # AUROC (requires sklearn)
        if _HAS_SKLEARN and len(qcd_errors) > 0:
            all_signal_errors: List[float] = []

            for cls in sorted(self._test_errors_by_class.keys()):
                signal_errors = np.array(self._test_errors_by_class[cls])
                if len(signal_errors) == 0:
                    continue
                cls_tag = f"cls{cls}"
                all_signal_errors.extend(signal_errors.tolist())

                y_true  = np.array([0] * len(qcd_errors) + [1] * len(signal_errors))
                y_score = np.concatenate([qcd_errors, signal_errors])
                try:
                    auroc = float(_sklearn_roc_auc_score(y_true, y_score))
                    self.log(f"ae/test_auroc_{cls_tag}", auroc)
                    test_metrics[f"auroc_{cls_tag}"] = auroc
                except Exception:
                    pass

                log.info(
                    f"[{cls_tag}] AUROC={test_metrics.get(f'auroc_{cls_tag}', float('nan')):.4f} "
                    f"sep_ratio={test_metrics.get(f'sep_ratio_{cls_tag}', float('nan')):.4f} "
                    f"mse={test_metrics.get(f'mse_mean_{cls_tag}', float('nan')):.5f}"
                )

            # Combined AUROC (all signals vs QCD)
            if all_signal_errors:
                y_true  = np.array([0] * len(qcd_errors) + [1] * len(all_signal_errors))
                y_score = np.concatenate([qcd_errors, np.array(all_signal_errors)])
                try:
                    auroc_all = float(_sklearn_roc_auc_score(y_true, y_score))
                    self.log("ae/test_auroc_all", auroc_all, prog_bar=True)
                    test_metrics["auroc_all"] = auroc_all
                    log.info(f"[all signals] AUROC={auroc_all:.4f}")
                except Exception:
                    pass

        # Drift
        test_metrics.update({
            f"drift_fpr{int(fpr*100):02d}": drift
            for fpr, drift in drift_metrics.items()
        })
        if drift_metrics:
            test_metrics["drift_metric"] = float(np.mean(list(drift_metrics.values())))

        self._test_metrics = test_metrics
        self._test_errors_by_class.clear()

        # Persist errors for plotting before on_validation_epoch_end clears them
        self._last_test_errors_normal = list(self.val_errors_normal)
        self._last_test_errors_anomaly = list(self.val_errors_anomaly)

        # Call on_validation_epoch_end for standard logging + cleanup
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint: dict):
        """Persist val thresholds and val drift so both survive best-ckpt reload."""
        checkpoint["val_thresholds"]    = self._val_thresholds
        checkpoint["bc_rate"]           = self._bc_rate
        checkpoint["val_drift_metric"]  = self._val_drift_metric
        checkpoint["val_drift_per_fpr"] = self._val_drift_per_fpr

    def on_load_checkpoint(self, checkpoint: dict):
        """Restore val thresholds and val drift saved at the best epoch."""
        self._val_thresholds    = checkpoint.get("val_thresholds", {})
        self._bc_rate           = checkpoint.get("bc_rate", 0)
        self._val_drift_metric  = checkpoint.get("val_drift_metric", float("nan"))
        self._val_drift_per_fpr = checkpoint.get("val_drift_per_fpr", {})

    def _compute_drift_metrics(self, test_errors_normal: np.ndarray) -> Dict[float, float]:
        """
        drift = | log( (p_hat + eps) / (fpr + eps) ) |
        threshold set on val, p_hat measured on test QCD.
        """
        if not self._val_thresholds or self._bc_rate == 0:
            return {}
        N = len(test_errors_normal)
        if N == 0:
            return {}
        eps = 0.5 / float(N)
        drift_metrics = {}
        for fpr, threshold in self._val_thresholds.items():
            fp    = int(np.sum(test_errors_normal > threshold))
            p_hat = fp / float(N)
            drift_metrics[fpr] = abs(math.log((p_hat + eps) / (float(fpr) + eps)))
        return drift_metrics

    def plot_reconstruction_errors(self, output_dir: Path, class_names: Optional[list] = None):
        # Use saved test errors if the live lists were already cleared
        normal_errors  = self.val_errors_normal  or getattr(self, "_last_test_errors_normal",  [])
        anomaly_errors = self.val_errors_anomaly or getattr(self, "_last_test_errors_anomaly", [])

        if len(normal_errors) == 0 or len(anomaly_errors) == 0:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 7))
        bins = np.linspace(
            min(min(normal_errors), min(anomaly_errors)),
            max(max(normal_errors), max(anomaly_errors)),
            60
        )
        ax.hist(normal_errors, bins=bins, alpha=0.6,
                label=f'Normal (QCD) - n={len(normal_errors)}',
                color='blue', edgecolor='darkblue')
        ax.hist(anomaly_errors, bins=bins, alpha=0.6,
                label=f'Anomalies (Higgs) - n={len(anomaly_errors)}',
                color='red', edgecolor='darkred')

        normal_mean  = np.mean(normal_errors)
        anomaly_mean = np.mean(anomaly_errors)
        ratio = anomaly_mean / normal_mean if normal_mean > 0 else float('inf')

        ax.set_xlabel('Reconstruction Error (MSE)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax.set_title(f'Anomaly Detection — Reconstruction Errors (Sep. Ratio={ratio:.2f}x)',
                     fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'anomaly_scores_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()

        with open(output_dir / 'anomaly_detection_metrics.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ANOMALY DETECTION METRICS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"  Normal samples: {len(normal_errors)}\n")
            f.write(f"  Anomaly samples: {len(anomaly_errors)}\n\n")
            f.write(f"  QCD mean MSE: {normal_mean:.6f} ± {np.std(normal_errors):.6f}\n")
            f.write(f"  Higgs mean MSE: {anomaly_mean:.6f} ± {np.std(anomaly_errors):.6f}\n")
            f.write(f"  Separation ratio: {ratio:.3f}x\n")

        print(f"\nPlots saved to: {output_dir}")
