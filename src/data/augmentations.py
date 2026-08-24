"""
Online physics augmentations for normalized COLLIDE2V feature vectors.

Operates on single flat tensors in normalized (preprocessed) space.
Requires the expanded feature_map.json and norm_stats.json produced
by the preprocessing pipeline.

Augmentations:
  phi_rotation  — global azimuthal rotation Δφ ~ Uniform(0, 2π)
                  exact in (sin, cos) space since phi normalization is "none"
  eta_boost     — longitudinal Lorentz boost Δη ~ N(0, eta_sigma)
                  maps to Δη/IQR shift on the robust-normalized η values
  pt_smearing   — per-particle multiplicative pT noise in log1p space
                  maps to N(0, pt_sigma)/IQR additive shift on normalized values
"""

import json
import math
import re
import torch
from typing import List, Tuple


class PhysicsAugmentation:
    """
    Physics-motivated online augmentation for a single normalized event vector.

    Args:
        feature_map_path: path to expanded feature_map.json (eos_preproc_dir)
        norm_stats_path:  path to norm_stats.json            (eos_preproc_dir)
        phi_rotation:     enable azimuthal rotation
        eta_boost:        enable longitudinal boost (eta shift)
        pt_smearing:      enable per-particle pT smearing
        eta_sigma:        std of Δη ~ N(0, eta_sigma) in physical units
        pt_sigma:         noise scale in log1p(pT) space (~fractional pT smearing)
    """

    def __init__(
        self,
        feature_map_path: str,
        norm_stats_path: str,
        phi_rotation: bool = True,
        eta_boost: bool = True,
        pt_smearing: bool = True,
        eta_sigma: float = 1.5,
        pt_sigma: float = 0.1,
    ):
        self.phi_rotation = phi_rotation
        self.eta_boost    = eta_boost
        self.pt_smearing  = pt_smearing
        self.eta_sigma    = eta_sigma
        self.pt_sigma     = pt_sigma

        with open(feature_map_path) as f:
            feature_map = json.load(f)
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        block_stats = norm_stats.get("block_group_stats", {})

        self.phi_pairs:   List[Tuple[int, int]] = []
        self.eta_entries: List[Tuple[int, float]] = []
        self.pt_entries:  List[Tuple[int, float]] = []

        for section_name, section in feature_map.items():
            start    = section["start"]
            columns  = section["columns"]
            topk     = section.get("topk", None)
            topk_int = topk if topk is not None else 1
            n_cols   = len(columns)

            phi_sin_j = next((j for j, c in enumerate(columns) if c.endswith("_Phi_sin")), None)
            phi_cos_j = next((j for j, c in enumerate(columns) if c.endswith("_Phi_cos")), None)
            eta_j     = next((j for j, c in enumerate(columns) if re.search(r"_Eta$",  c)), None)
            pt_j      = next((j for j, c in enumerate(columns) if re.search(r"_PT$",   c)), None)
            met_j     = next((j for j, c in enumerate(columns) if re.search(r"_MET$",  c)), None)

            eta_iqr = float(block_stats.get(f"{section_name}.eta", {}).get("iqr", 1.0))
            pt_iqr  = float(block_stats.get(f"{section_name}.pt",  {}).get("iqr", 1.0))
            met_iqr = float(block_stats.get(f"{section_name}.met", {}).get("iqr", 1.0))

            for k in range(topk_int):
                base = start + k * n_cols

                if phi_sin_j is not None and phi_cos_j is not None:
                    self.phi_pairs.append((base + phi_sin_j, base + phi_cos_j))

                if eta_j is not None:
                    self.eta_entries.append((base + eta_j, eta_iqr))

                if pt_j is not None:
                    self.pt_entries.append((base + pt_j, pt_iqr))

                if met_j is not None:
                    self.pt_entries.append((base + met_j, met_iqr))

        # Precompute index/IQR tensors (moved to device lazily in __call__)
        self._sin_idx_t = torch.tensor([s for s, _ in self.phi_pairs])   if self.phi_pairs   else None
        self._cos_idx_t = torch.tensor([c for _, c in self.phi_pairs])   if self.phi_pairs   else None
        self._eta_idx_t = torch.tensor([i for i, _ in self.eta_entries]) if self.eta_entries else None
        self._eta_iqr_t = torch.tensor([q for _, q in self.eta_entries], dtype=torch.float32) if self.eta_entries else None
        self._pt_idx_t  = torch.tensor([i for i, _ in self.pt_entries])  if self.pt_entries  else None
        self._pt_iqr_t  = torch.tensor([q for _, q in self.pt_entries],  dtype=torch.float32) if self.pt_entries  else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts both a single event [D] and a batch [B, D]."""
        squeezed = x.dim() == 1
        if squeezed:
            x = x.unsqueeze(0)
        x = x.clone()
        if self.phi_rotation:
            x = self._rotate_phi_batch(x)
        if self.eta_boost:
            x = self._boost_eta_batch(x)
        if self.pt_smearing:
            x = self._smear_pt_batch(x)
        if squeezed:
            x = x.squeeze(0)
        return x

    def _rotate_phi_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Global azimuthal rotation by Δφ ~ Uniform(0, 2π), one per event."""
        if self._sin_idx_t is None:
            return x
        dev     = x.device
        sin_idx = self._sin_idx_t.to(dev)
        cos_idx = self._cos_idx_t.to(dev)
        dphi    = torch.empty(x.shape[0], 1, device=dev).uniform_(0.0, 2.0 * math.pi)
        cos_d   = torch.cos(dphi)
        sin_d   = torch.sin(dphi)
        s = x[:, sin_idx]
        c = x[:, cos_idx]
        x[:, sin_idx] = s * cos_d + c * sin_d
        x[:, cos_idx] = c * cos_d - s * sin_d
        return x

    def _boost_eta_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Longitudinal boost: shift all η by Δη ~ N(0, eta_sigma), one per event.
        Padded entries (stored as 0.0) are left unchanged.
        """
        if self._eta_idx_t is None:
            return x
        dev     = x.device
        eta_idx = self._eta_idx_t.to(dev)
        iqrs    = self._eta_iqr_t.to(dev)
        deta    = torch.randn(x.shape[0], 1, device=dev) * self.eta_sigma
        vals    = x[:, eta_idx]
        nonzero = (vals != 0.0).float()
        x[:, eta_idx] = vals + nonzero * (deta / iqrs.unsqueeze(0))
        return x

    def _smear_pt_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Per-particle pT smearing: independent N(0, pt_sigma) noise per (event, particle).
        Padded entries (stored as 0.0) are left unchanged.
        """
        if self._pt_idx_t is None:
            return x
        dev    = x.device
        pt_idx = self._pt_idx_t.to(dev)
        iqrs   = self._pt_iqr_t.to(dev)
        noise  = torch.randn(x.shape[0], len(self.pt_entries), device=dev) * self.pt_sigma
        vals   = x[:, pt_idx]
        nonzero = (vals != 0.0).float()
        x[:, pt_idx] = vals + nonzero * (noise / iqrs.unsqueeze(0))
        return x
