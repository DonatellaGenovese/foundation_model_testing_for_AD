"""High-level physics observables from raw vectorized Full-Reco events.

Migrated from scripts/xai_mocktest/gmm_step2_2_physical_profiles.py.
Vectorized layout (247 cols):
  Jets     : 7 × 12 = 84 (0–83)  + count at 84
  Electrons: 8 × 8  = 64 (85–148)
  Muons    : 7 × 8  = 56 (149–204)
  Photons  : 5 × 8  = 40 (205–244)
  MET      : 2           (245–246)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .constants import CLASS_FOLDERS, PHYSICS_VARS

J_NCOLS = 7
J_TOPK = 12
J_START = 0
J_COUNT = J_START + J_NCOLS * J_TOPK
E_NCOLS = 8
E_TOPK = 8
E_START = J_COUNT + 1
M_NCOLS = 7
M_TOPK = 8
M_START = E_START + E_NCOLS * E_TOPK
P_NCOLS = 5
P_TOPK = 8
P_START = M_START + M_NCOLS * M_TOPK
MET_START = P_START + P_NCOLS * P_TOPK  # 245

# COLLIDE-2V stores JetPuppiAK4_BTag as an int8 bitmask, not a boolean: the
# Delphes CMS_PhaseII_200PU_v04 card defines six b-tagging working points and
# writes one bit each, so the field takes all 64 values in [0, 63]. Bits pair up
# into three severity levels, measured here against the generator-matched
# `Flavor` field on HH->4b:
#
#   bit 0, 3  loose    eff(b) 0.82   light-jet mistag 0.119
#   bit 1, 4  medium   eff(b) 0.65   light-jet mistag 0.018
#   bit 2, 5  tight    eff(b) 0.47   light-jet mistag 0.0026
#
# The bits are independent draws, not nested cuts on a continuous discriminant
# (jets exist with only the tightest bit set), so testing `BTag > 0.5` is an OR
# over six trials: it reaches eff 0.93 but at a 0.24 mistag rate — a worse
# efficiency-to-purity ratio (3.9) than any single working point. Medium is the
# operating point used here.
#
# BTagPhys, the other tagging field, is unusable: eff(b) 0.100 against a 0.101
# mistag rate, i.e. it tags b and light jets at the same frequency.
BTAG_WP = 1


def compute_physics(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute paper Table observables from raw vectorized data (N, 247)."""
    jet_PT = X[:, J_START + np.arange(J_TOPK) * J_NCOLS + 0]
    jet_Eta = X[:, J_START + np.arange(J_TOPK) * J_NCOLS + 1]
    jet_Phi = X[:, J_START + np.arange(J_TOPK) * J_NCOLS + 2]
    jet_Mass = X[:, J_START + np.arange(J_TOPK) * J_NCOLS + 3]
    jet_BTag = X[:, J_START + np.arange(J_TOPK) * J_NCOLS + 4]

    elec_PT = X[:, E_START + np.arange(E_TOPK) * E_NCOLS + 0]
    elec_Phi = X[:, E_START + np.arange(E_TOPK) * E_NCOLS + 2]
    muon_PT = X[:, M_START + np.arange(M_TOPK) * M_NCOLS + 0]
    muon_Phi = X[:, M_START + np.arange(M_TOPK) * M_NCOLS + 2]

    MET_val = X[:, MET_START]
    MET_phi = X[:, MET_START + 1]

    jet_mask = jet_PT > 0
    elec_mask = elec_PT > 0
    muon_mask = muon_PT > 0

    HT = (jet_PT * jet_mask).sum(axis=1)
    n_jets = jet_mask.astype(float).sum(axis=1)
    n_bjets = (((jet_BTag.astype(np.int64) >> BTAG_WP) & 1) * jet_mask).astype(float).sum(axis=1)
    n_leptons = elec_mask.astype(float).sum(axis=1) + muon_mask.astype(float).sum(axis=1)

    has2j = n_jets >= 2
    PT1, Eta1, Phi1, M1 = jet_PT[:, 0], jet_Eta[:, 0], jet_Phi[:, 0], jet_Mass[:, 0]
    PT2, Eta2, Phi2, M2 = jet_PT[:, 1], jet_Eta[:, 1], jet_Phi[:, 1], jet_Mass[:, 1]

    E1 = np.sqrt(np.maximum(PT1**2 * np.cosh(Eta1) ** 2 + M1**2, 0))
    E2 = np.sqrt(np.maximum(PT2**2 * np.cosh(Eta2) ** 2 + M2**2, 0))
    px1 = PT1 * np.cos(Phi1)
    py1 = PT1 * np.sin(Phi1)
    pz1 = PT1 * np.sinh(Eta1)
    px2 = PT2 * np.cos(Phi2)
    py2 = PT2 * np.sin(Phi2)
    pz2 = PT2 * np.sinh(Eta2)
    Mjj2 = (E1 + E2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    Mjj = np.where(has2j, np.sqrt(np.maximum(Mjj2, 0)), np.nan)
    deta_jj = np.where(has2j, np.abs(Eta1 - Eta2), np.nan)

    all_lept_PT = np.concatenate([elec_PT, muon_PT], axis=1)
    all_lept_Phi = np.concatenate([elec_Phi, muon_Phi], axis=1)
    has_lept = n_leptons > 0
    lead_idx = np.argmax(all_lept_PT, axis=1)
    lead_PT = all_lept_PT[np.arange(len(X)), lead_idx]
    lead_phi = all_lept_Phi[np.arange(len(X)), lead_idx]
    dphi = np.abs(lead_phi - MET_phi)
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    MT2 = 2 * lead_PT * MET_val * (1 - np.cos(dphi))
    MT = np.where(has_lept, np.sqrt(np.maximum(MT2, 0)), np.nan)

    return {
        "HT": HT,
        "MET": MET_val,
        "n_jets": n_jets,
        "n_bjets": n_bjets,
        "n_leptons": n_leptons,
        "Mjj": Mjj,
        "deta_jj": deta_jj,
        "MT": MT,
    }


def load_class_vectorized(
    vec_dir: Path,
    class_idx: int,
    max_events: int,
) -> Optional[np.ndarray]:
    """Load up to max_events raw vectorized rows for one class folder."""
    folder = CLASS_FOLDERS[class_idx]
    cls_dir = Path(vec_dir) / folder
    if not cls_dir.exists():
        print(f"  [WARN] {cls_dir} not found")
        return None
    files = sorted(f for f in cls_dir.iterdir() if f.name.endswith("_x.npy"))
    chunks: List[np.ndarray] = []
    total = 0
    for f in files:
        X = np.load(f, mmap_mode="r")
        chunks.append(np.array(X))
        total += len(X)
        if total >= max_events:
            break
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)[:max_events]


def load_class_preprocessed(
    preproc_split_dir: Path,
    class_idx: int,
    max_events: int,
) -> Optional[np.ndarray]:
    """Load preprocessed _x.npy shards (encoder input) for one class."""
    folder = CLASS_FOLDERS[class_idx]
    cls_dir = Path(preproc_split_dir) / folder
    if not cls_dir.exists():
        print(f"  [WARN] preprocessed {cls_dir} not found")
        return None
    files = sorted(f for f in cls_dir.iterdir() if f.name.endswith("_x.npy"))
    chunks: List[np.ndarray] = []
    total = 0
    for f in files:
        X = np.load(f, mmap_mode="r")
        chunks.append(np.array(X))
        total += len(X)
        if total >= max_events:
            break
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)[:max_events]


def build_matched_arrays(
    vectorized_dir: Path,
    preproc_split_dir: Path,
    class_indices: Sequence[int],
    encode_fn,
    max_per_class: int = 20_000,
    pca=None,
) -> tuple:
    """
    Build aligned (embeddings, labels, phys_dict) for the requested classes.

    Physics come from raw vectorized files; embeddings from preprocessed + encode_fn.
    encode_fn(X_pp) -> (N, d) ndarray.
    """
    all_phys = {v: [] for v in PHYSICS_VARS}
    all_embs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for c in class_indices:
        print(f"\nClass {c}…")
        X_raw = load_class_vectorized(vectorized_dir, c, max_per_class)
        if X_raw is None:
            continue
        phys = compute_physics(X_raw)
        X_pp = load_class_preprocessed(preproc_split_dir, c, max_per_class)
        if X_pp is None:
            continue
        n = min(len(X_raw), len(X_pp))
        X_pp = X_pp[:n]
        for v in PHYSICS_VARS:
            all_phys[v].append(phys[v][:n])

        print(f"  Encoding {n} events…", flush=True)
        embs = encode_fn(X_pp)
        if pca is not None:
            embs = pca.transform(embs)
        all_embs.append(embs)
        all_labels.append(np.full(n, c, dtype=int))

    if not all_embs:
        raise RuntimeError("No classes loaded — check vectorized/preprocessed paths")

    labels = np.concatenate(all_labels)
    embeddings = np.concatenate(all_embs)
    phys_out = {v: np.concatenate(all_phys[v]) for v in PHYSICS_VARS}
    return embeddings, labels, phys_out


def save_matched_npz(
    path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    phys: Dict[str, np.ndarray],
) -> None:
    payload = {"embeddings": embeddings.astype(np.float32), "labels": labels.astype(np.int32)}
    for v, arr in phys.items():
        payload[f"phys_{v}"] = arr.astype(np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"Saved matched data: {path}  ({len(labels):,} events)")


def load_matched_npz(path: Path) -> tuple:
    """Return (embeddings, labels, phys_dict)."""
    d = np.load(path)
    embeddings = d["embeddings"].astype(np.float32)
    labels = d["labels"].astype(int)
    phys = {v: d[f"phys_{v}"].astype(np.float64) for v in PHYSICS_VARS if f"phys_{v}" in d.files}
    missing = [v for v in PHYSICS_VARS if v not in phys]
    if missing:
        raise KeyError(f"matched npz missing physics keys: {missing}")
    return embeddings, labels, phys
