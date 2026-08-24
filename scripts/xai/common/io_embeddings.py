"""Load train/val/test embedding npz files and SM/BSM masks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .constants import BSM_INDICES, SM_INDICES


def load_split(embedding_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load `{split}_embeddings.npz` with keys embeddings, labels."""
    path = Path(embedding_dir) / f"{split}_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return data["embeddings"].astype(np.float32), data["labels"].astype(int)


def load_train_val_test(embedding_dir: Path):
    """Return dict split -> (X, y). Missing val is OK (returns None)."""
    out = {}
    for split in ("train", "val", "test"):
        path = Path(embedding_dir) / f"{split}_embeddings.npz"
        if path.exists():
            out[split] = load_split(embedding_dir, split)
        else:
            out[split] = None
    if out["train"] is None or out["test"] is None:
        raise FileNotFoundError(
            f"Need train_embeddings.npz and test_embeddings.npz under {embedding_dir}"
        )
    return out


def filter_classes(
    X: np.ndarray,
    y: np.ndarray,
    class_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, list(class_indices))
    return X[mask], y[mask]


def sm_mask(y: np.ndarray) -> np.ndarray:
    return np.isin(y, SM_INDICES)


def bsm_mask(y: np.ndarray) -> np.ndarray:
    return np.isin(y, BSM_INDICES)


def filter_low_norm(
    X: np.ndarray,
    y: Optional[np.ndarray],
    percentile: float,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """
    Remove low-L2-norm embeddings (degenerate / mostly-padded events).

    If threshold is None, calibrate on X at the given percentile.
    Returns (X_kept, y_kept_or_None, threshold_used).
    """
    norms = np.linalg.norm(X, axis=1)
    if threshold is None:
        if percentile <= 0:
            return X, y, 0.0
        threshold = float(np.percentile(norms, percentile))
    keep = norms >= threshold
    X_out = X[keep]
    y_out = y[keep] if y is not None else None
    return X_out, y_out, float(threshold)
