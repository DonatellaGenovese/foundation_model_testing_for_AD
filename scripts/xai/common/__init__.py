"""Shared helpers for the paper XAI pipeline (Sec. 4.3)."""

from .constants import (
    BSM_INDICES,
    CLASS_FOLDERS,
    CLASS_NAMES,
    HH4B_LABEL,
    PHYSICS_BINS,
    PHYSICS_LABELS,
    PHYSICS_VARS,
    SIG_LABELS,
    SM_COLORS,
    SM_INDICES,
)
from . import ae_score, io_embeddings, physics

__all__ = [
    "ae_score",
    "io_embeddings",
    "physics",
    "SM_INDICES",
    "BSM_INDICES",
    "HH4B_LABEL",
    "SIG_LABELS",
    "CLASS_NAMES",
    "CLASS_FOLDERS",
    "SM_COLORS",
    "PHYSICS_VARS",
    "PHYSICS_LABELS",
    "PHYSICS_BINS",
]
