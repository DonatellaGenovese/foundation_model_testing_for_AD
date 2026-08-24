"""
GMM-based anomaly explanation pipeline.

Uses two separate experiment configs:
  - SM (Phase 1)  → loads 12 nosparse SM classes from v2_12class_nosparse_highlevel
                    (same norm_stats the encoder was trained on)
  - BSM (Phase 2) → loads 3 BSM classes (indices 12,13,14) from v2_nosparse_higgs_smstats_highlevel
                    (BSM data normalised with SM stats for consistency)

Usage:
    python src/gmm_explain.py \
        --ckpt-path /eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/aug_supcon_12class_nosparse_dmodel128_cern/seed_0/checkpoints/epoch_048.ckpt \
        --sm-experiment  aug_supcon_12class_nosparse_dmodel128_cern \
        --bsm-experiment anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern \
        --output-dir  /eos/user/d/dgenoves/anomaly_pipeline/gmm_explain

    # Override sweep parameters:
        --k-values 6,10,12,15,20 --cov-types diag --seed 42
"""

import argparse
import collections
import functools
import json
import os
from importlib import import_module
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

CONFIG_DIR = str(Path(__file__).parent.parent / "configs")

# Feature types that have log1p applied before robust normalization
_LOG1P_TYPES = {"pt", "mass", "met"}


# ---------------------------------------------------------------------------
# Un-normalization
# ---------------------------------------------------------------------------

def _unnormalize(
    x: np.ndarray,
    feat_type: str,
    group: str,
    norm_stats: dict,
) -> np.ndarray:
    """
    Invert robust normalization (and log1p if applicable).

    Transform pipeline: raw → [log1p] → robust_scale → stored
    Inverse:            stored → robust_unscale → [expm1] → raw
    """
    mode = norm_stats["_meta"]["normalization_mode_per_feature_group"].get(feat_type, "none")
    stats = norm_stats["block_group_stats"].get(f"{group}.{feat_type}", {})

    if mode == "none" or not stats:
        return x.copy()

    if mode == "robust":
        x_transformed = x * stats["iqr"] + stats["median"]
        if feat_type in _LOG1P_TYPES:
            return np.expm1(np.clip(x_transformed, -50, 50))
        return x_transformed

    # minmax and other modes — not needed for physical features
    return x.copy()


# ---------------------------------------------------------------------------
# Feature index parsing
# ---------------------------------------------------------------------------

def _parse_jet_indices(feature_map: dict) -> dict:
    """
    Parse flat indices for jet PT, Eta, Phi_sin, Phi_cos, Mass from the
    *preprocessed* feature_map.json (which includes trig/onehot expansions).
    """
    jets = feature_map["jets"]
    start = jets["start"]
    cols = jets["columns"]
    n_cols = len(cols)
    topk = jets["topk"]

    col_pos = {col: i for i, col in enumerate(cols)}

    idx = {}
    for i in range(topk):
        base = start + i * n_cols
        idx[f"jet{i}_PT"]      = base + col_pos["FullReco_JetPuppiAK4_PT"]
        idx[f"jet{i}_Eta"]     = base + col_pos["FullReco_JetPuppiAK4_Eta"]
        idx[f"jet{i}_Phi_sin"] = base + col_pos["FullReco_JetPuppiAK4_Phi_sin"]
        idx[f"jet{i}_Phi_cos"] = base + col_pos["FullReco_JetPuppiAK4_Phi_cos"]
        idx[f"jet{i}_Mass"]    = base + col_pos["FullReco_JetPuppiAK4_Mass"]

    return idx


# ---------------------------------------------------------------------------
# Physical feature computation
# ---------------------------------------------------------------------------

def compute_physical_features(
    x: np.ndarray,
    feature_map: dict,
    norm_stats: dict,
) -> np.ndarray:
    """
    Compute M_jj and HT from a batch of *preprocessed* (normalized) features.

    M_jj: invariant mass of the two leading jets (4-vector sum).
    HT:   scalar sum of all valid jet pTs.

    Returns array of shape [N, 2] = [M_jj, HT].
    """
    idx = _parse_jet_indices(feature_map)
    n, topk = x.shape[0], feature_map["jets"]["topk"]

    pt      = np.zeros((n, topk))
    eta     = np.zeros((n, topk))
    sin_phi = np.zeros((n, topk))
    cos_phi = np.zeros((n, topk))
    mass    = np.zeros((n, topk))

    for i in range(topk):
        pt[:, i]      = _unnormalize(x[:, idx[f"jet{i}_PT"]],   "pt",   "jets", norm_stats)
        eta[:, i]     = _unnormalize(x[:, idx[f"jet{i}_Eta"]],  "eta",  "jets", norm_stats)
        mass[:, i]    = _unnormalize(x[:, idx[f"jet{i}_Mass"]], "mass", "jets", norm_stats)
        # phi stored as (sin, cos) — use directly, no atan2 round-trip needed
        sin_phi[:, i] = x[:, idx[f"jet{i}_Phi_sin"]]
        cos_phi[:, i] = x[:, idx[f"jet{i}_Phi_cos"]]

    # Padded jets have pT → 0 after expm1(log1p(0)) = 0
    valid = pt > 0  # [N, topk]

    # HT: sum of valid jet pTs
    ht = (pt * valid).sum(axis=1)

    # M_jj: invariant mass of the two leading jets using 4-vectors
    has_two = valid[:, 0] & valid[:, 1]
    pt0, eta0, sp0, cp0, m0 = pt[:, 0], eta[:, 0], sin_phi[:, 0], cos_phi[:, 0], mass[:, 0]
    pt1, eta1, sp1, cp1, m1 = pt[:, 1], eta[:, 1], sin_phi[:, 1], cos_phi[:, 1], mass[:, 1]

    e0 = np.sqrt(np.maximum((pt0 * np.cosh(eta0)) ** 2 + m0 ** 2, 0.0))
    e1 = np.sqrt(np.maximum((pt1 * np.cosh(eta1)) ** 2 + m1 ** 2, 0.0))
    px0, py0, pz0 = pt0 * cp0, pt0 * sp0, pt0 * np.sinh(eta0)
    px1, py1, pz1 = pt1 * cp1, pt1 * sp1, pt1 * np.sinh(eta1)

    mjj_sq = np.maximum(
        (e0 + e1) ** 2 - (px0 + px1) ** 2 - (py0 + py1) ** 2 - (pz0 + pz1) ** 2,
        0.0,
    )
    mjj = np.where(has_two, np.sqrt(mjj_sq), 0.0)

    return np.stack([mjj, ht], axis=1)  # [N, 2]


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def _load_encoder(ckpt_path: str, model_class_path: str, device: str):
    """Dynamically load a LightningModule encoder from checkpoint."""
    module_path, class_name = model_class_path.rsplit(".", 1)
    ModelClass = getattr(import_module(module_path), class_name)

    import omegaconf
    torch.serialization.add_safe_globals([
        functools.partial,
        torch.optim.AdamW,
        torch.optim.Adam,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        torch.optim.lr_scheduler.OneCycleLR,
        torch.optim.lr_scheduler.CosineAnnealingLR,
        omegaconf.ListConfig,
        omegaconf.DictConfig,
        collections.defaultdict,
    ])

    model = ModelClass.load_from_checkpoint(
        ckpt_path, map_location=device, weights_only=False
    )
    model.eval()
    model.to(device)
    return model


def _get_embeddings(model, x: torch.Tensor) -> torch.Tensor:
    """Call get_embeddings regardless of whether model wraps an encoder."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "get_embeddings"):
        return model.encoder.get_embeddings(x)
    if hasattr(model, "get_embeddings"):
        return model.get_embeddings(x)
    raise AttributeError(f"{type(model).__name__} has no get_embeddings method")


# ---------------------------------------------------------------------------
# Data extraction (single forward pass)
# ---------------------------------------------------------------------------

def extract_split(
    model,
    dataloader,
    feature_map: dict,
    norm_stats: dict,
    device: str,
    split_name: str,
    max_events: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single forward pass over one dataloader split.

    Returns:
        embeddings  [N, d_model]
        phys        [N, 2]  (M_jj, HT)
        labels      [N]
    """
    all_z, all_phys, all_labels = [], [], []
    n_seen = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {split_name}"):
            if max_events is not None and n_seen >= max_events:
                break

            x = batch[0].to(device)
            y = batch[1]

            z    = _get_embeddings(model, x)
            phys = compute_physical_features(x.cpu().numpy(), feature_map, norm_stats)

            all_z.append(z.cpu().numpy())
            all_phys.append(phys)
            all_labels.append(y.numpy() if hasattr(y, "numpy") else np.asarray(y))
            n_seen += len(x)

    return (
        np.concatenate(all_z,      axis=0),
        np.concatenate(all_phys,   axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# GMM fitting
# ---------------------------------------------------------------------------

def select_gmm(
    z_train: np.ndarray,
    z_val: np.ndarray,
    k_values: List[int],
    cov_types: List[str],
    seed: int = 42,
) -> Tuple[GaussianMixture, dict]:
    """
    Grid search over K and covariance type using val SM log-likelihood.

    Fits each candidate on z_train, scores on z_val (mean log-likelihood
    per sample — higher is better). Returns the best GMM (already fitted
    on z_train) and a dict with the full selection results.

    Note: 'full' covariance may be unreliable for d_model > 128 due to
    the high number of parameters. A warning is logged in that case.
    """
    d = z_train.shape[1]
    if "full" in cov_types and d > 128:
        log.warning(
            f"d_model={d} > 128 with full covariance: "
            "regularisation (reg_covar=1e-3) applied, results may be unreliable."
        )

    scores: dict = {}
    all_gmms: dict = {}
    best_score = -np.inf
    best_gmm   = None
    best_key   = None

    for cov in cov_types:
        for k in k_values:
            reg = 1e-3 if cov == "full" else 1e-6
            try:
                g = GaussianMixture(
                    n_components=k, covariance_type=cov,
                    random_state=seed, max_iter=300, n_init=3,
                    reg_covar=reg,
                )
                g.fit(z_train)
                train_ll = float(g.score(z_train))
                val_ll   = float(g.score(z_val))
                bic      = float(g.bic(z_train))
                aic      = float(g.aic(z_train))
                gap      = train_ll - val_ll
                key      = f"K{k}_{cov}"
                scores[key] = {
                    "train_loglik": train_ll,
                    "val_loglik":   val_ll,
                    "bic":          bic,
                    "aic":          aic,
                    "gap":          gap,
                }
                all_gmms[key] = g
                log.info(
                    f"  K={k:3d}  cov={cov:4s}  "
                    f"train={train_ll:.4f}  val={val_ll:.4f}  "
                    f"gap={gap:.4f}  BIC={bic:.1f}  AIC={aic:.1f}"
                )
                if val_ll > best_score:
                    best_score = val_ll
                    best_gmm   = g
                    best_key   = key
            except Exception as e:
                log.warning(f"  K={k}  cov={cov}  FAILED: {e}")
                scores[f"K{k}_{cov}"] = {"val_loglik": float("-inf")}

    log.info(f"  Best: {best_key}  val_loglik={best_score:.4f}")
    return best_gmm, all_gmms, {
        "best_key":        best_key,
        "best_k":          best_gmm.n_components,
        "best_cov_type":   best_gmm.covariance_type,
        "best_val_loglik": best_score,
        "all_scores":      scores,
    }


# ---------------------------------------------------------------------------
# Physical profiles per GMM component
# ---------------------------------------------------------------------------

PHYS_NAMES = ["M_jj", "HT"]


def build_profiles(
    gmm: GaussianMixture,
    z_sm: np.ndarray,
    phys_sm: np.ndarray,
    labels_sm: np.ndarray,
    sm_class_names: List[str],
) -> dict:
    """
    For each GMM component k, compute responsibility-weighted mean and std
    of physical features, and identify the top-3 dominant SM classes with
    their proportions.
    """
    resp = gmm.predict_proba(z_sm)  # [N, K]
    profiles = {}

    for ki in range(gmm.n_components):
        w = resp[:, ki]
        w_sum = w.sum() + 1e-12

        mean = (w[:, None] * phys_sm).sum(axis=0) / w_sum
        var  = (w[:, None] * (phys_sm - mean) ** 2).sum(axis=0) / w_sum
        std  = np.sqrt(np.maximum(var, 1e-12))

        # Per-class weighted responsibility (labels are 0..14)
        class_w = np.array([w[labels_sm == ci].sum() for ci in range(len(sm_class_names))])
        total_w = class_w.sum() + 1e-12

        class_proportions = {sm_class_names[ci]: float(class_w[ci] / total_w)
                             for ci in range(len(sm_class_names))}

        # Top-3 dominant classes by proportion
        top3_idx = np.argsort(class_w)[::-1][:3]
        top3 = [
            {"class": sm_class_names[ci], "proportion": float(class_w[ci] / total_w)}
            for ci in top3_idx
            if class_w[ci] > 0
        ]

        profiles[ki] = {
            "mean": {n: float(mean[i]) for i, n in enumerate(PHYS_NAMES)},
            "std":  {n: float(std[i])  for i, n in enumerate(PHYS_NAMES)},
            "dominant_class":        top3[0]["class"],
            "dominant_class_weight": top3[0]["proportion"],
            "top3_classes":          top3,
            "class_proportions":     class_proportions,
        }

    return profiles


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

def compute_anomaly_scores(
    gmm: GaussianMixture,
    z: np.ndarray,
    resp: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns two anomaly scores for each event:
      nll      = -log p(z)       higher = more anomalous
      max_resp = 1 - max_k r_k   higher = more anomalous (less "owned" by any component)
    """
    nll = -gmm.score_samples(z)
    if resp is None:
        resp = gmm.predict_proba(z)
    max_resp = 1.0 - resp.max(axis=1)
    return nll, max_resp


def compute_auroc_per_signal(
    nll_bg: np.ndarray,
    max_resp_bg: np.ndarray,
    z_bsm: np.ndarray,
    labels_bsm: np.ndarray,
    bsm_orig_indices: List[int],
    bsm_class_names: List[str],
    gmm: GaussianMixture,
) -> dict:
    """
    Compute AUROC (NLL and maxResp scores) for each BSM class vs QCD background.
    Returns dict keyed by class name with auroc_nll and auroc_max_resp.
    """
    nll_bsm_all, max_resp_bsm_all = compute_anomaly_scores(gmm, z_bsm)
    aurocs: dict = {}
    for cls_id, cls_name in zip(bsm_orig_indices, bsm_class_names):
        mask = labels_bsm == cls_id
        if not mask.any():
            continue
        nll_sig      = nll_bsm_all[mask]
        max_resp_sig = max_resp_bsm_all[mask]
        n_bg         = len(nll_bg)
        n_sig        = int(mask.sum())
        y_true       = np.concatenate([np.zeros(n_bg), np.ones(n_sig)])
        auroc_nll      = float(roc_auc_score(y_true, np.concatenate([nll_bg,      nll_sig])))
        auroc_max_resp = float(roc_auc_score(y_true, np.concatenate([max_resp_bg, max_resp_sig])))
        aurocs[cls_name] = {
            "auroc_nll":      auroc_nll,
            "auroc_max_resp": auroc_max_resp,
            "n_signal":       n_sig,
            "n_background":   n_bg,
        }
        log.info(
            f"    {cls_name:20s}: AUROC(NLL)={auroc_nll:.4f}  AUROC(maxResp)={auroc_max_resp:.4f}"
            f"  (sig={n_sig}, bg={n_bg})"
        )
    return aurocs


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_residuals(
    gmm: GaussianMixture,
    z: np.ndarray,
    phys: np.ndarray,
    profiles: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each event:
        x_expected = Σ_k r_k * mean_k        (soft assignment)
        std        = std of dominant component
        residual   = (x - x_expected) / std   in σ units

    Returns: residuals [N, F], x_expected [N, F], responsibilities [N, K]
    """
    resp = gmm.predict_proba(z)  # [N, K]
    k = gmm.n_components

    means = np.array([[profiles[ki]["mean"][n] for n in PHYS_NAMES] for ki in range(k)])
    stds  = np.array([[profiles[ki]["std"][n]  for n in PHYS_NAMES] for ki in range(k)])

    x_expected = resp @ means                  # [N, F]
    std_soft   = resp @ stds                   # [N, F] — responsibility-weighted std
    residuals  = (phys - x_expected) / np.maximum(std_soft, 1.0)  # floor 1 GeV

    return residuals, x_expected, resp


# ---------------------------------------------------------------------------
# Hydra config composition
# ---------------------------------------------------------------------------

def _compose_cfg(config_name: str, experiment: str, output_dir: str) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"experiment={experiment}",
                f"paths.output_dir={output_dir}",
            ],
        )
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GMM anomaly explanation")
    parser.add_argument("--ckpt-path",       required=True,  help="Path to Phase 1 encoder checkpoint")
    parser.add_argument("--sm-experiment",   required=True,  help="Phase 1 experiment config (15-class SM, e.g. aug_supcon_15class_best_cern)")
    parser.add_argument("--bsm-experiment",  required=True,  help="Phase 2 experiment config (BSM classes, e.g. anomaly_qcd_vs_higgs_embedding_augsupcon_best_cern)")
    parser.add_argument("--output-dir",      required=True,  help="Directory for all outputs")
    parser.add_argument("--k-values",        default="6,10,12,15,20",  help="Comma-separated K values to sweep")
    parser.add_argument("--cov-types",       default="diag",         help="Comma-separated covariance types to sweep")
    parser.add_argument("--max-gmm-samples", type=int, default=500_000, help="Max train SM samples for GMM fitting")
    parser.add_argument("--max-events",      type=int, default=None, help="Cap total events per split (for quick local testing)")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SM uses train.yaml base config; BSM uses anomaly_detection.yaml
    cfg_sm  = _compose_cfg("train.yaml",             args.sm_experiment,  str(output_dir))
    cfg_bsm = _compose_cfg("anomaly_detection.yaml", args.bsm_experiment, str(output_dir))
    os.chdir(cfg_sm.paths.root_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # SM norm_stats + feature_map (from v2_12class_nosparse_highlevel)
    preproc_dir_sm = Path(cfg_sm.data.paths.eos_preproc_dir)
    if not (preproc_dir_sm / "norm_stats.json").exists():
        preproc_dir_sm = Path(cfg_sm.data.paths.tmp_preproc_dir)
    norm_stats_sm = json.load(open(preproc_dir_sm / "norm_stats.json"))
    feature_map   = json.load(open(preproc_dir_sm / "feature_map.json"))
    log.info(f"SM  preprocessing: {preproc_dir_sm}")

    # BSM norm_stats (from v2_nosparse_higgs_allsm_highlevel — separate normalization)
    preproc_dir_bsm = Path(cfg_bsm.data.paths.eos_preproc_dir)
    if not (preproc_dir_bsm / "norm_stats.json").exists():
        preproc_dir_bsm = Path(cfg_bsm.data.paths.tmp_preproc_dir)
    norm_stats_bsm = json.load(open(preproc_dir_bsm / "norm_stats.json"))
    log.info(f"BSM preprocessing: {preproc_dir_bsm}")

    # Load encoder
    # cfg_sm is a train config (no model_class field) — use allSM config or fallback
    model_class_path = cfg_bsm.get(
        "model_class",
        "src.models.collide2v_augmented_supcon.COLLIDE2VAugmentedSupConLitModule",
    )
    log.info(f"Loading encoder ({model_class_path}) from {args.ckpt_path}")
    encoder = _load_encoder(args.ckpt_path, model_class_path, device)

    import hydra as _hydra
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

    sm_class_names = list(cfg_sm.data.to_classify)  # 12 SM class names (labels 0..11)

    # SM datamodule — all 15 classes from v2_15class_highlevel_full
    dm_sm = _hydra.utils.instantiate(cfg_sm.data)
    dm_sm.setup()

    # BSM class info from cfg_bsm (no datamodule instantiated)
    all_bsm_classes  = list(cfg_bsm.data.to_classify)
    bsm_orig_indices = list(cfg_bsm.anomaly_classes)   # [12, 13, 14] in allSM 15-class scheme
    bsm_class_names  = [all_bsm_classes[i] for i in bsm_orig_indices]

    k_values  = [int(k) for k in args.k_values.split(",")]
    cov_types = [c.strip() for c in args.cov_types.split(",")]

    # --- Extract SM splits (train / val / test) ---
    sm_data: dict = {}
    for split_name, loader_fn in [
        ("train", dm_sm.train_dataloader),
        ("val",   dm_sm.val_dataloader),
        ("test",  dm_sm.test_dataloader),
    ]:
        log.info(f"Extracting SM {split_name}...")
        z, phys, labels = extract_split(
            encoder, loader_fn(), feature_map, norm_stats_sm, device, f"SM_{split_name}",
            max_events=args.max_events,
        )
        sm_data[split_name] = (z, phys, labels)
        log.info(f"  SM {split_name}: {z.shape[0]} events")

    z_train_sm, phys_train_sm, labels_train_sm = sm_data["train"]
    z_val_sm    = sm_data["val"][0]    # only embeddings needed (for GMM model selection)
    z_test_sm,  phys_test_sm,  labels_test_sm  = sm_data["test"]

    # --- Build BSM test dataloader directly from .npy files ---
    # Labels 12,13,14 match the allSM 15-class scheme (v2_nosparse_higgs_allsm_highlevel)
    BSM_CLASS_FOLDERS = {12: "VBFHbb", 13: "HH_4b", 14: "ggHtautau"}
    xs_bsm, ys_bsm = [], []
    for cls_id, folder in BSM_CLASS_FOLDERS.items():
        cls_dir = preproc_dir_bsm / "test" / folder
        if not cls_dir.exists():
            log.warning(f"BSM dir not found: {cls_dir}")
            continue
        x_files = sorted(cls_dir.glob("*_x.npy"))
        X = np.concatenate([np.load(f) for f in x_files], axis=0)
        log.info(f"  BSM test {folder}: {len(X)} events, dim={X.shape[1]}")
        xs_bsm.append(torch.from_numpy(X).float())
        ys_bsm.append(torch.full((len(X),), cls_id, dtype=torch.long))

    bsm_test_loader = TorchDataLoader(
        TensorDataset(torch.cat(xs_bsm), torch.cat(ys_bsm)),
        batch_size=1024, shuffle=False, num_workers=0,
    )

    # --- Extract BSM test split ---
    log.info("Extracting BSM test split...")
    z_test_bsm, phys_test_bsm, labels_test_bsm = extract_split(
        encoder, bsm_test_loader, feature_map, norm_stats_bsm, device, "BSM_test",
        max_events=args.max_events,
    )
    log.info(f"  BSM test: {z_test_bsm.shape[0]} events  "
             f"classes={dict(zip(*np.unique(labels_test_bsm, return_counts=True)))}")

    # --- Subsample train SM for GMM fitting if needed, keeping proportions across splits ---
    rng = np.random.default_rng(args.seed)
    if len(z_train_sm) > args.max_gmm_samples:
        ratio = args.max_gmm_samples / len(z_train_sm)
        idx_train = np.sort(rng.choice(len(z_train_sm), size=args.max_gmm_samples, replace=False))
        z_fit          = z_train_sm[idx_train]
        phys_fit       = phys_train_sm[idx_train]
        labels_fit     = labels_train_sm[idx_train]

        n_val  = max(1, int(len(z_val_sm)  * ratio))
        n_test = max(1, int(len(z_test_sm) * ratio))
        idx_val  = np.sort(rng.choice(len(z_val_sm),  size=n_val,  replace=False))
        idx_test = np.sort(rng.choice(len(z_test_sm), size=n_test, replace=False))
        z_val_sm_sub   = z_val_sm[idx_val]
        z_test_sm_sub, phys_test_sm_sub, labels_test_sm_sub = z_test_sm[idx_test], phys_test_sm[idx_test], labels_test_sm[idx_test]

        log.info(
            f"Subsampled SM (ratio={ratio:.4f}): "
            f"train {len(z_train_sm)}→{len(z_fit)}  "
            f"val {len(z_val_sm)}→{n_val}  "
            f"test {len(z_test_sm)}→{n_test}"
        )
    else:
        z_fit          = z_train_sm
        phys_fit       = phys_train_sm
        labels_fit     = labels_train_sm
        z_val_sm_sub   = z_val_sm
        z_test_sm_sub, phys_test_sm_sub, labels_test_sm_sub = z_test_sm, phys_test_sm, labels_test_sm
        log.info(f"Using all {len(z_fit)} train SM events for GMM fitting (no subsampling)")

    # --- Model selection: fit on train SM, score on val SM ---
    log.info(f"Model selection — K ∈ {k_values}  cov ∈ {cov_types}  (scored on val SM log-likelihood)...")
    gmm, all_gmms, selection = select_gmm(z_fit, z_val_sm_sub, k_values, cov_types, seed=args.seed)
    log.info(f"Selected: K={selection['best_k']}  cov={selection['best_cov_type']}")

    # --- Build physical profiles for ALL K values ---
    log.info("Building physical profiles per GMM component (all K)...")
    all_profiles: dict = {}
    for key, g in all_gmms.items():
        log.info(f"  Profiles for {key}...")
        all_profiles[key] = build_profiles(g, z_fit, phys_fit, labels_fit, sm_class_names)

    profiles = all_profiles[selection["best_key"]]  # best K for downstream use

    for ki, p in profiles.items():
        top3_str = ", ".join(f"{t['class']}({t['proportion']:.0%})" for t in p["top3_classes"])
        log.info(
            f"  Component {ki:2d}: {top3_str}  "
            f"M_jj={p['mean']['M_jj']:7.1f}±{p['std']['M_jj']:5.1f} GeV  "
            f"HT={p['mean']['HT']:7.1f}±{p['std']['HT']:5.1f} GeV"
        )

    # --- BSM residuals + analysis for ALL K values ---
    log.info("Computing BSM residuals and per-class analysis for all K...")
    all_bsm_results: dict = {}
    all_bsm_phys:   dict = {}

    for key, g in all_gmms.items():
        log.info(f"  BSM analysis for {key}...")
        prof_k = all_profiles[key]
        res_bsm_k, x_exp_bsm_k, resp_bsm_k = compute_residuals(g, z_test_bsm, phys_test_bsm, prof_k)

        bsm_phys_k: dict = {}
        for cls_id, cls_name in zip(bsm_orig_indices, bsm_class_names):
            mask      = labels_test_bsm == cls_id
            phys_cls  = phys_test_bsm[mask]
            resp_cls  = resp_bsm_k[mask]
            dom_comp  = resp_cls.argmax(axis=1)
            comp_counts = {int(c): int(n) for c, n in zip(*np.unique(dom_comp, return_counts=True))}
            bsm_phys_k[cls_name] = {
                "n_events":  int(mask.sum()),
                "M_jj_mean": float(phys_cls[:, 0].mean()),
                "M_jj_std":  float(phys_cls[:, 0].std()),
                "M_jj_p50":  float(np.median(phys_cls[:, 0])),
                "HT_mean":   float(phys_cls[:, 1].mean()),
                "HT_std":    float(phys_cls[:, 1].std()),
                "HT_p50":    float(np.median(phys_cls[:, 1])),
                "dominant_components": dict(sorted(comp_counts.items(), key=lambda x: -x[1])[:5]),
            }

        all_bsm_results[key] = (res_bsm_k, x_exp_bsm_k, resp_bsm_k)
        all_bsm_phys[key]   = bsm_phys_k

        for cls_name, info in bsm_phys_k.items():
            dom_str = ", ".join(f"comp{c}({n})" for c, n in list(info["dominant_components"].items())[:3])
            log.info(f"    {cls_name}: M_jj={info['M_jj_mean']:.1f}±{info['M_jj_std']:.1f}  HT={info['HT_mean']:.1f}±{info['HT_std']:.1f}  dom=[{dom_str}]")

    # convenience aliases for best K
    res_bsm, x_exp_bsm, resp_bsm = all_bsm_results[selection["best_key"]]
    bsm_phys_analysis             = all_bsm_phys[selection["best_key"]]

    # --- SM residuals (calibration, best K only) ---
    log.info("Computing SM test residuals (best K)...")
    res_sm, x_exp_sm, resp_sm = compute_residuals(gmm, z_test_sm_sub, phys_test_sm_sub, profiles)

    # QCD-only SM test slice used as background for AUROC (label 0 = QCD)
    qcd_mask = labels_test_sm_sub == 0

    # --- Save results ---
    log.info("=" * 60)
    log.info(f"Saving results to {output_dir}")

    np.savez_compressed(
        output_dir / "gmm.npz",
        means=gmm.means_,
        covariances=gmm.covariances_,
        weights=gmm.weights_,
    )

    with open(output_dir / "profiles.json", "w") as f:
        json.dump({str(k): v for k, v in profiles.items()}, f, indent=2)

    # Save profiles, BSM analysis, GMM params, SM results, and AUROC for every K
    all_aurocs: dict = {}
    for key, prof in all_profiles.items():
        with open(output_dir / f"profiles_{key}.json", "w") as f:
            json.dump({str(ki): v for ki, v in prof.items()}, f, indent=2)
        with open(output_dir / f"bsm_phys_analysis_{key}.json", "w") as f:
            json.dump(all_bsm_phys[key], f, indent=2)
        res_k, x_exp_k, resp_k = all_bsm_results[key]
        g_k = all_gmms[key]

        # Anomaly scores
        nll_bsm_k, max_resp_bsm_k = compute_anomaly_scores(g_k, z_test_bsm, resp=resp_k)
        nll_qcd_k, max_resp_qcd_k = compute_anomaly_scores(g_k, z_test_sm_sub[qcd_mask])

        np.savez_compressed(
            output_dir / f"bsm_results_{key}.npz",
            labels=labels_test_bsm,
            embeddings=z_test_bsm,
            phys_features=phys_test_bsm,
            phys_expected=x_exp_k,
            residuals=res_k,
            responsibilities=resp_k,
            nll=nll_bsm_k,
            max_resp=max_resp_bsm_k,
        )
        np.savez_compressed(
            output_dir / f"gmm_{key}.npz",
            means=g_k.means_,
            covariances=g_k.covariances_,
            weights=g_k.weights_,
        )

        # SM test responsibilities, residuals, and anomaly scores for this K
        res_sm_k, x_exp_sm_k, resp_sm_k = compute_residuals(
            g_k, z_test_sm_sub, phys_test_sm_sub, prof
        )
        nll_sm_k, max_resp_sm_k = compute_anomaly_scores(g_k, z_test_sm_sub, resp=resp_sm_k)
        np.savez_compressed(
            output_dir / f"sm_results_{key}.npz",
            labels=labels_test_sm_sub,
            phys_features=phys_test_sm_sub,
            phys_expected=x_exp_sm_k,
            residuals=res_sm_k,
            responsibilities=resp_sm_k,
            nll=nll_sm_k,
            max_resp=max_resp_sm_k,
        )

        # AUROC: each BSM class vs QCD background
        log.info(f"  AUROC scores for {key}:")
        aurocs_k = compute_auroc_per_signal(
            nll_bg=nll_qcd_k,
            max_resp_bg=max_resp_qcd_k,
            z_bsm=z_test_bsm,
            labels_bsm=labels_test_bsm,
            bsm_orig_indices=bsm_orig_indices,
            bsm_class_names=bsm_class_names,
            gmm=g_k,
        )
        all_aurocs[key] = aurocs_k
        with open(output_dir / f"auroc_{key}.json", "w") as f:
            json.dump(aurocs_k, f, indent=2)

        log.info(
            f"  Saved profiles_{key}.json  bsm_phys_analysis_{key}.json  "
            f"bsm_results_{key}.npz  gmm_{key}.npz  sm_results_{key}.npz  auroc_{key}.json"
        )

    # Keep best-K files with canonical names for backwards compatibility
    with open(output_dir / "bsm_phys_analysis.json", "w") as f:
        json.dump(bsm_phys_analysis, f, indent=2)
    np.savez_compressed(
        output_dir / "bsm_results.npz",
        labels=labels_test_bsm,
        embeddings=z_test_bsm,
        phys_features=phys_test_bsm,
        phys_expected=x_exp_bsm,
        residuals=res_bsm,
        responsibilities=resp_bsm,
    )

    # SM test results (calibration — best K) + embeddings subsampled for UMAP
    umap_n = min(50_000, len(z_test_sm_sub))
    rng_umap = np.random.default_rng(args.seed + 1)
    idx_umap_sm = np.sort(rng_umap.choice(len(z_test_sm_sub), size=umap_n, replace=False))
    np.savez_compressed(
        output_dir / "sm_results.npz",
        labels=labels_test_sm_sub,
        phys_features=phys_test_sm_sub,
        phys_expected=x_exp_sm,
        residuals=res_sm,
        responsibilities=resp_sm,
        embeddings=z_test_sm_sub,
        embeddings_umap=z_test_sm_sub[idx_umap_sm],
        labels_umap=labels_test_sm_sub[idx_umap_sm],
    )

    d_model = int(z_train_sm.shape[1])
    summary = {
        "ckpt_path":      args.ckpt_path,
        "sm_experiment":  args.sm_experiment,
        "bsm_experiment": args.bsm_experiment,
        "d_model":        d_model,
        "n_sm_fit":       int(len(z_fit)),
        "n_sm_train":     int(len(z_train_sm)),
        "n_sm_val":       int(len(z_val_sm_sub)),
        "n_sm_test":      int(len(z_test_sm_sub)),
        "n_bsm_test":     int(len(z_test_bsm)),
        "bsm_classes":    bsm_class_names,
        "model_selection": selection,
        "aurocs_by_k": all_aurocs,
        "mean_abs_residuals_bsm": {
            n: float(np.abs(res_bsm[:, i]).mean()) for i, n in enumerate(PHYS_NAMES)
        },
        "mean_abs_residuals_sm_test": {
            n: float(np.abs(res_sm[:, i]).mean()) for i, n in enumerate(PHYS_NAMES)
        },
        "profiles": {
            str(ki): {
                "top3_classes": p["top3_classes"],
                "mean_M_jj":   p["mean"]["M_jj"],
                "std_M_jj":    p["std"]["M_jj"],
                "mean_HT":     p["mean"]["HT"],
                "std_HT":      p["std"]["HT"],
            }
            for ki, p in profiles.items()
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("GMM EXPLANATION COMPLETE")
    log.info(f"  d_model:              {d_model}")
    log.info(f"  Selected: K={selection['best_k']}  cov={selection['best_cov_type']}  val_loglik={selection['best_val_loglik']:.4f}")
    log.info(f"  SM fit / train / val: {len(z_fit)} / {len(z_train_sm)} / {len(z_val_sm)}")
    log.info(f"  BSM test events:      {len(z_test_bsm)}")
    log.info(f"  Mean |residual| BSM       M_jj: {summary['mean_abs_residuals_bsm']['M_jj']:.3f}σ")
    log.info(f"  Mean |residual| BSM       HT:   {summary['mean_abs_residuals_bsm']['HT']:.3f}σ")
    log.info(f"  Mean |residual| SM (test) M_jj: {summary['mean_abs_residuals_sm_test']['M_jj']:.3f}σ  (should be ~1)")
    log.info(f"  Mean |residual| SM (test) HT:   {summary['mean_abs_residuals_sm_test']['HT']:.3f}σ  (should be ~1)")
    best_key = selection["best_key"]
    if best_key in all_aurocs:
        log.info(f"  AUROC summary (best K={selection['best_k']}):")
        for cls_name, v in all_aurocs[best_key].items():
            log.info(f"    {cls_name:20s}: NLL={v['auroc_nll']:.4f}  maxResp={v['auroc_max_resp']:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
