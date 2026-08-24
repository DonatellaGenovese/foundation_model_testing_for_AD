"""Autoencoder MSE scoring and FPR-based threshold (paper: AE flags, GMM interprets)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _ensure_project_root() -> Path:
    import sys

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_ae(ae_ckpt_path: str | Path):
    """Load AutoencoderLitModule from a Lightning checkpoint."""
    _ensure_project_root()
    import torch
    from src.models.autoencoder import AutoencoderLitModule

    ckpt = torch.load(str(ae_ckpt_path), map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    model = AutoencoderLitModule(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def compute_ae_mse(
    ae_ckpt_path: str | Path,
    z: np.ndarray,
) -> np.ndarray:
    """Per-event MSE reconstruction error. Shape (N,).

    Scores the full array in one forward pass (same as mock_test/xai_gmm_latent.py).
    """
    _ensure_project_root()
    import torch

    model = load_ae(ae_ckpt_path)
    z_t = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        reconstruction, _ = model.model(z_t)
        mse = ((z_t - reconstruction) ** 2).mean(dim=1).numpy()
    return mse.astype(np.float64)


def compute_ae_residual(
    ae_ckpt_path: str | Path,
    z: np.ndarray,
) -> np.ndarray:
    """Per-event, per-dimension squared reconstruction error (z-ẑ)^2. Shape (N, d).

    compute_ae_mse(ckpt, z) is equivalent to compute_ae_residual(ckpt, z).mean(axis=1);
    use this instead when the per-dimension breakdown is needed (e.g. 06_ae_mechanism.py)
    to avoid a second forward pass.
    """
    _ensure_project_root()
    import torch

    model = load_ae(ae_ckpt_path)
    z_t = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        reconstruction, _ = model.model(z_t)
        residual = (z_t - reconstruction) ** 2
    return residual.numpy().astype(np.float64)


def load_val_thresholds(ae_ckpt_path: str | Path) -> dict:
    """Val-calibrated FPR→threshold map saved by src/models/autoencoder.py
    (AutoencoderLitModule.on_save_checkpoint -> checkpoint["val_thresholds"]).

    Keys are the FPRs in autoencoder.TARGET_FPRS (0.01, 0.05, 0.10). Empty dict
    if the checkpoint predates this field.
    """
    _ensure_project_root()
    import torch

    ckpt = torch.load(str(ae_ckpt_path), map_location="cpu", weights_only=False)
    return {float(k): float(v) for k, v in ckpt.get("val_thresholds", {}).items()}


def load_val_threshold(ae_ckpt_path: str | Path, fpr: float, atol: float = 1e-6) -> Optional[float]:
    """Single val-calibrated threshold at `fpr`, or None if not present in the checkpoint."""
    thresholds = load_val_thresholds(ae_ckpt_path)
    for k, v in thresholds.items():
        if abs(k - fpr) < atol:
            return v
    return None


def threshold_at_fpr(
    scores_bg: np.ndarray,
    fpr: float = 0.10,
) -> float:
    """
    Threshold retaining `fpr` of background (higher score = more anomalous).
    For FPR=10%, returns the 90th percentile of background scores.
    """
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0,1), got {fpr}")
    return float(np.quantile(scores_bg, 1.0 - fpr))


def flag_anomalies(
    mse: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Boolean mask: True where MSE exceeds threshold."""
    return mse > threshold


def resolve_ae_threshold(
    mse: np.ndarray,
    labels: np.ndarray,
    ae_threshold: Optional[float] = None,
    ae_ckpt_path: Optional[str | Path] = None,
    bg_label: int = 0,
    fpr: float = 0.10,
) -> Tuple[float, str]:
    """
    Return (threshold, source), in order of precedence:

    1. ae_threshold, if given explicitly (CLI override).
    2. The val-calibrated threshold saved in the AE checkpoint at ae_ckpt_path
       (autoencoder.py:derive_thresholds, computed on val QCD during training —
       the same threshold behind the README's AUROC/TPR@10% strategy tables).
    3. Self-calibration at `fpr` on bg_label within `mse`/`labels` (fallback for
       older checkpoints without a saved val_thresholds map). This calibrates
       and evaluates on the same array, so treat it as a display cut, not a
       validated operating point.
    """
    if ae_threshold is not None:
        return float(ae_threshold), "cli"
    if ae_ckpt_path is not None:
        val_thr = load_val_threshold(ae_ckpt_path, fpr)
        if val_thr is not None:
            return val_thr, f"val_thresholds[fpr={fpr:.2f}]"
    bg = mse[labels == bg_label]
    if len(bg) == 0:
        raise ValueError(f"No background events with label={bg_label} to calibrate AE threshold")
    thr = threshold_at_fpr(bg, fpr=fpr)
    return thr, f"selfcal_fpr{fpr:.2f}_label{bg_label}"
