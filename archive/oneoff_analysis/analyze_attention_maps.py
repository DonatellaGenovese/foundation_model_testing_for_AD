"""
Attention map analysis for explainability: QCD vs Higgs signals.

Loads a pretrained AugmentedSupCon encoder and optionally an Autoencoder, runs
inference on QCD and the 3 Higgs types (VBFHbb, HH_4b, ggHtautau), and produces:
  - Mean attention maps per class and transformer layer
  - Mean attention maps split by classification outcome (well/missed)
  - A few individual event attention maps for visual inspection
  - AE reconstruction error distributions with threshold lines

Classification criterion
------------------------
The AE checkpoint stores `val_thresholds`, a dict mapping FPR → MSE threshold
computed on the validation QCD set:  threshold = np.quantile(val_mse_QCD, 1 - fpr)
Three FPR working points are available: 1%, 5%, 10%.  Default: 5%.
  - Well classified: error > threshold for Higgs (TP),  error ≤ threshold for QCD (TN)
  - Missed:          error ≤ threshold for Higgs (FN),  error > threshold for QCD (FP)

Attention extraction
--------------------
PyTorch's TransformerEncoderLayer passes need_weights=False by default, so
attention weights are not returned.  We register a forward_pre_hook on each
MultiheadAttention module that overrides the kwarg to True, then capture the
returned weights ([batch, n_heads, seq, seq]) in a forward_hook.

Token layout (38 tokens for the standard CERN config)
------------------------------------------------------
  0          : event  (count features — nJets)
  1  – 12    : jet_0  … jet_11
  13 – 20    : e_0    … e_7    (electrons)
  21 – 28    : mu_0   … mu_7   (muons)
  29 – 36    : γ_0    … γ_7    (photons)
  37         : MET

Usage
-----
  python scripts/analyze_attention_maps.py \\
      --supcon-ckpt /eos/.../aug_supcon_15class_secondtake/checkpoints/epoch_XXX.ckpt \\
      --ae-ckpt     /eos/.../anomaly_pipeline/best_trial/ae/checkpoints/ae-XXX.ckpt \\
      --data-dir    /eos/user/d/dgenoves/foundation_model_testing_data/v2_18class_highlevel_full/preprocessed/test \\
      --output-dir  /eos/user/d/dgenoves/attention_analysis \\
      --fpr 0.05 \\
      --n-samples 2000
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ── make project importable when run as a plain script ───────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from models.collide2v_augmented_supcon import COLLIDE2VAugmentedSupConLitModule  # noqa: E402
from models.autoencoder import AutoencoderLitModule  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

QCD_CLASS = 0
HIGGS_CLASSES: Dict[int, str] = {15: "VBFHbb", 16: "HH_4b", 17: "ggHtautau"}
CLASS_NAMES: Dict[int, str] = {0: "QCD", **HIGGS_CLASSES}
CLASS_COLORS: Dict[int, str] = {0: "steelblue", 15: "tomato", 16: "mediumorchid", 17: "goldenrod"}

# folder names as they appear in the preprocessed data directory
CLASS_FOLDERS: Dict[int, str] = {
    0:  "QCD_HT50toInf",
    15: "VBFHbb",
    16: "HH_4b",
    17: "ggHtautau",
}

GROUP_SHORT: Dict[str, str] = {
    "jets": "jet",
    "electrons": "e",
    "muons": "mu",
    "photons": "γ",
    "puppi_met": "MET",
}


# ─────────────────────────────────────────────────────────────────────────────
# Token helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_token_labels(group_configs: list) -> List[str]:
    """Human-readable label for every transformer token."""
    labels = ["event"]  # event count token is always prepended
    for cfg in group_configs:
        short = GROUP_SHORT.get(cfg["name"], cfg["name"])
        n = cfg["num_tokens"]
        labels += [short] if n == 1 else [f"{short}_{i}" for i in range(n)]
    return labels


def build_token_group_spans(group_configs: list) -> List[Tuple[str, int, int]]:
    """Return [(group_short_name, start_idx, end_idx), ...] for axis annotation."""
    spans = [("event", 0, 1)]
    idx = 1
    for cfg in group_configs:
        short = GROUP_SHORT.get(cfg["name"], cfg["name"])
        n = cfg["num_tokens"]
        spans.append((short, idx, idx + n))
        idx += n
    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

class AttentionExtractor:
    """
    Patches a nn.TransformerEncoder to capture per-layer attention weights.

    Uses a forward_pre_hook (with_kwargs=True, requires PyTorch ≥ 2.0) on each
    MultiheadAttention to force need_weights=True and average_attn_weights=False,
    then captures the per-head weights in a forward_hook.

    Stored shape per layer: [batch, n_heads, seq_len, seq_len]
    """

    def __init__(self, transformer_encoder: nn.TransformerEncoder):
        self._store: Dict[int, torch.Tensor] = {}
        self._hooks: list = []

        for layer_idx, layer in enumerate(transformer_encoder.layers):
            mha: nn.MultiheadAttention = layer.self_attn

            def _pre(module, args, kwargs, _idx=layer_idx):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return args, kwargs

            def _post(module, inp, out, _idx=layer_idx):
                # out = (attn_output, attn_weights)  since need_weights=True
                if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                    self._store[_idx] = out[1].detach().cpu()

            self._hooks.append(
                mha.register_forward_pre_hook(_pre, with_kwargs=True)
            )
            self._hooks.append(mha.register_forward_hook(_post))

    def get(self) -> Dict[int, torch.Tensor]:
        return dict(self._store)

    def clear(self):
        self._store.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_class_data(
    data_dir: Path,
    folder_name: str,
    n_samples: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load up to n_samples from a preprocessed class directory (sharded .npy)."""
    class_dir = data_dir / folder_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    x_files = sorted(class_dir.glob("*_x.npy"))
    if not x_files:
        raise FileNotFoundError(f"No *_x.npy files in {class_dir}")

    xs, ys = [], []
    for xf in x_files:
        yf = xf.with_name(xf.name.replace("_x.npy", "_y.npy"))
        xs.append(np.load(xf, mmap_mode="r"))
        ys.append(np.load(yf, mmap_mode="r"))

    X_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(y_all))
    idx = rng.choice(len(y_all), size=n, replace=False)

    return (
        torch.from_numpy(X_all[idx].copy()).float(),
        torch.from_numpy(y_all[idx].copy()).long(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings_and_attention(
    encoder,  # TinyTransformer
    extractor: AttentionExtractor,
    X: torch.Tensor,
    batch_size: int = 256,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Returns:
        embeddings : [N, d_model]
        attn_maps  : {layer_idx: [N, n_heads, seq_len, seq_len]}
    """
    encoder = encoder.to(device).eval()
    all_emb: List[torch.Tensor] = []
    all_attn: Dict[int, List[torch.Tensor]] = defaultdict(list)

    for start in range(0, len(X), batch_size):
        batch = X[start : start + batch_size].to(device)
        extractor.clear()
        emb = encoder.get_embeddings(batch)
        all_emb.append(emb.cpu())
        for layer_idx, attn in extractor.get().items():
            all_attn[layer_idx].append(attn)

    embeddings = torch.cat(all_emb, dim=0)
    attn_maps = {k: torch.cat(v, dim=0) for k, v in all_attn.items()}
    return embeddings, attn_maps


@torch.no_grad()
def compute_ae_errors(
    ae_net: nn.Module,
    embeddings: torch.Tensor,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Per-sample MSE reconstruction error."""
    ae_net = ae_net.to(device).eval()
    errors: List[float] = []
    for start in range(0, len(embeddings), batch_size):
        batch = embeddings[start : start + batch_size].to(device)
        recon, _ = ae_net(batch)
        mse = ((recon - batch) ** 2).mean(dim=1)
        errors.extend(mse.cpu().numpy().tolist())
    return np.array(errors)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_group_lines(ax, spans: List[Tuple[str, int, int]], n_tokens: int):
    """Draw faint vertical/horizontal lines between token groups."""
    for _, start, end in spans[1:]:  # skip first boundary at 0
        ax.axvline(start - 0.5, color="white", linewidth=0.5, alpha=0.6)
        ax.axhline(start - 0.5, color="white", linewidth=0.5, alpha=0.6)


def _set_token_ticks(ax, token_labels: List[str], spans: List[Tuple[str, int, int]]):
    """Use group-center ticks with short labels to reduce clutter."""
    centers = [int((s + e) / 2) for _, s, e in spans]
    group_names = [name for name, _, _ in spans]
    ax.set_xticks(centers)
    ax.set_xticklabels(group_names, fontsize=7)
    ax.set_yticks(centers)
    ax.set_yticklabels(group_names, fontsize=7)


def _attn_head_mean(attn: torch.Tensor) -> np.ndarray:
    """[N, heads, S, S] → [N, S, S] by averaging over heads."""
    return attn.mean(dim=1).numpy()


def _make_tight_fig(n_rows: int, n_cols: int, cell: float = 2.8):
    return plt.subplots(
        n_rows, n_cols,
        figsize=(cell * n_cols, cell * n_rows),
        constrained_layout=True,
        squeeze=False,
    )


def _fill_attn_grid(axes, row_labels, layer_data_list, token_labels, spans, n_layers, vmax):
    """Fill a pre-created axes grid. Returns last imshow handle."""
    im = None
    for row, (label, rd) in enumerate(zip(row_labels, layer_data_list)):
        for col in range(n_layers):
            ax = axes[row, col]
            if col not in rd or len(rd[col]) == 0:
                ax.axis("off")
                continue
            mean_a = rd[col].mean(axis=0) if rd[col].ndim == 3 else rd[col]
            im = ax.imshow(mean_a, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
            _draw_group_lines(ax, spans, len(token_labels))
            _set_token_ticks(ax, token_labels, spans)
            col_title = f"Layer {col}" if row == 0 else ""
            row_label  = label if col == 0 else ""
            if col_title:
                ax.set_title(col_title, fontsize=8, pad=3)
            if row_label:
                ax.set_ylabel(row_label, fontsize=8, labelpad=4)
    return im


def plot_mean_attention_per_higgs(
    higgs_id: int,
    attn_qcd:   Dict[int, np.ndarray],
    attn_higgs: Dict[int, np.ndarray],
    token_labels: List[str],
    spans: List[Tuple[str, int, int]],
    n_layers: int,
    output_path: Path,
):
    """One image per Higgs type: 2 rows (QCD / Higgs), n_layers cols."""
    higgs_name = HIGGS_CLASSES[higgs_id]
    row_labels = ["QCD", higgs_name]
    layer_data = [attn_qcd, attn_higgs]

    vmax = max(
        rd[l].mean(axis=0).max()
        for rd in layer_data
        for l in range(n_layers)
        if l in rd and len(rd[l]) > 0
    )

    fig, axes = _make_tight_fig(2, n_layers)
    im = _fill_attn_grid(axes, row_labels, layer_data, token_labels, spans, n_layers, vmax)
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), shrink=0.8, label="Mean attn weight")
    fig.suptitle(f"Mean attention maps — QCD vs {higgs_name}", fontsize=10)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_mean_attention_by_outcome(
    higgs_id: int,
    attn_correct: Dict[int, Dict[int, np.ndarray]],
    attn_missed:  Dict[int, Dict[int, np.ndarray]],
    token_labels: List[str],
    spans: List[Tuple[str, int, int]],
    n_layers: int,
    fpr: float,
    output_path: Path,
):
    """
    One image per Higgs type.
    Rows: QCD-correct, QCD-missed, Higgs-correct, Higgs-missed (skipped if empty).
    """
    higgs_name = HIGGS_CLASSES[higgs_id]

    def _has(d, cls_id):
        return cls_id in d and any(len(v) > 0 for v in d[cls_id].values())

    row_specs = []
    for cls_id, label_base in [(QCD_CLASS, "QCD"), (higgs_id, higgs_name)]:
        if _has(attn_correct, cls_id):
            row_specs.append((f"{label_base}\ncorrect", attn_correct[cls_id]))
        if _has(attn_missed, cls_id):
            row_specs.append((f"{label_base}\nmissed",  attn_missed[cls_id]))

    if not row_specs:
        print(f"  No events for outcome plot ({higgs_name}) — skipping.")
        return

    vmax = max(
        rd[l].mean(axis=0).max()
        for _, rd in row_specs
        for l in range(n_layers)
        if l in rd and len(rd[l]) > 0
    )

    n_rows = len(row_specs)
    fig, axes = _make_tight_fig(n_rows, n_layers)
    im = _fill_attn_grid(
        axes,
        [lbl for lbl, _ in row_specs],
        [rd  for _, rd  in row_specs],
        token_labels, spans, n_layers, vmax,
    )
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), shrink=0.8, label="Mean attn weight")
    fig.suptitle(
        f"Attention by outcome — QCD vs {higgs_name}  (FPR={fpr:.0%})", fontsize=10
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sample_events(
    higgs_id: int,
    attn_qcd:   Dict[int, torch.Tensor],
    attn_higgs: Dict[int, torch.Tensor],
    token_labels: List[str],
    spans: List[Tuple[str, int, int]],
    last_layer: int,
    n_events: int,
    output_path: Path,
):
    """
    One image per Higgs type.
    Rows: QCD / Higgs (head-averaged).  Cols: individual event samples.
    """
    higgs_name = HIGGS_CLASSES[higgs_id]
    rng = np.random.default_rng(0)

    fig, axes = _make_tight_fig(2, n_events, cell=2.6)

    for row, (cname, attn_tensor) in enumerate([("QCD", attn_qcd), (higgs_name, attn_higgs)]):
        attn = attn_tensor[last_layer]          # [N, heads, S, S]
        mean_attn = attn.mean(dim=1).numpy()    # [N, S, S]
        n_avail = len(mean_attn)
        idxs = rng.choice(n_avail, size=min(n_events, n_avail), replace=False)
        vmax = mean_attn[idxs].max()

        for col, ev_idx in enumerate(idxs):
            ax = axes[row, col]
            im = ax.imshow(mean_attn[ev_idx], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
            _draw_group_lines(ax, spans, len(token_labels))
            _set_token_ticks(ax, token_labels, spans)
            if row == 0:
                ax.set_title(f"Event {ev_idx}", fontsize=7, pad=3)
            if col == 0:
                ax.set_ylabel(cname, fontsize=8, labelpad=4)

    fig.suptitle(f"Sample attention maps — QCD vs {higgs_name}  (Layer {last_layer})", fontsize=10)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_mse_distributions(
    errors_by_class: Dict[int, np.ndarray],
    thresholds: Dict[float, float],
    output_path: Path,
):
    """Reconstruction error histograms + threshold lines for all FPR values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls_id, errors in errors_by_class.items():
        cname = CLASS_NAMES.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(cls_id, "gray")
        ax.hist(errors, bins=100, alpha=0.55, label=cname, color=color,
                density=True, histtype="stepfilled")

    thr_colors = {0.01: "red", 0.05: "darkorange", 0.10: "green"}
    for fpr, thr in sorted(thresholds.items()):
        ax.axvline(thr, linestyle="--", color=thr_colors.get(fpr, "black"),
                   linewidth=1.5, label=f"FPR={fpr:.0%}  thr={thr:.5f}")

    ax.set_xlabel("Reconstruction MSE", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("AE reconstruction error by class", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Attention map analysis for QCD vs Higgs explainability"
    )
    p.add_argument("--supcon-ckpt", required=True,
                   help="Path to SupCon encoder checkpoint (.ckpt)")
    p.add_argument("--ae-ckpt", required=True,
                   help="Path to Autoencoder checkpoint (.ckpt).")
    p.add_argument("--data-dir", required=True,
                   help="Preprocessed test-split directory "
                        "(contains QCD_inclusive/, VBFHbb/, HH_4b/, ggHtautau/ subdirs)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for plots and summary JSON")
    p.add_argument("--fpr", type=float, default=0.05,
                   help="FPR working point for classification (0.01 / 0.05 / 0.10). Default: 0.05")
    p.add_argument("--n-samples", type=int, default=2000,
                   help="Max events per class to analyse. Default: 2000")
    p.add_argument("--n-sample-events", type=int, default=4,
                   help="Number of individual events to show in the sample plot. Default: 4")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for encoder inference. Default: 256")
    p.add_argument("--device", default="cpu",
                   help="Device: cpu or cuda. Default: cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # ── 1. Load SupCon encoder ───────────────────────────────────────────────
    model_cls = COLLIDE2VAugmentedSupConLitModule
    print(f"\n[1] Loading AugmentedSupCon checkpoint:\n    {args.supcon_ckpt}")
    supcon = model_cls.load_from_checkpoint(
        args.supcon_ckpt, map_location=args.device, weights_only=False
    )
    supcon.eval()
    encoder = supcon.encoder  # TinyTransformer

    token_labels = build_token_labels(encoder.group_configs)
    spans = build_token_group_spans(encoder.group_configs)
    n_tokens = len(token_labels)
    n_layers = len(encoder.encoder.layers)
    n_heads = encoder.encoder.layers[0].self_attn.num_heads

    print(f"    d_model={encoder.d_model}, layers={n_layers}, heads={n_heads}, tokens={n_tokens}")
    print(f"    Token layout: {token_labels}")

    # ── 2. Patch transformer for attention extraction ────────────────────────
    print("\n[2] Patching transformer layers for attention extraction...")
    extractor = AttentionExtractor(encoder.encoder)

    # ── 3. Load AE and thresholds ────────────────────────────────────────────
    ae_net = None
    thresholds: Dict[float, float] = {}

    if args.ae_ckpt:
        print(f"\n[3] Loading Autoencoder checkpoint:\n    {args.ae_ckpt}")
        ae_litmodule = AutoencoderLitModule.load_from_checkpoint(
            args.ae_ckpt, map_location=args.device, weights_only=False
        )
        ae_litmodule.eval()
        ae_net = ae_litmodule.model

        thresholds = ae_litmodule._val_thresholds or {}
        if thresholds:
            print("    Thresholds from val QCD set:")
            for fpr_val, thr in sorted(thresholds.items()):
                marker = " ◄ selected" if abs(fpr_val - args.fpr) < 1e-6 else ""
                print(f"      FPR={fpr_val:.0%}  MSE threshold = {thr:.6f}{marker}")
        else:
            print("    Warning: val_thresholds not in checkpoint — will derive from loaded QCD errors.")

    # ── 4. Load data ─────────────────────────────────────────────────────────
    print(f"\n[4] Loading data from:\n    {data_dir}")
    X_by_class: Dict[int, torch.Tensor] = {}

    for cls_id, folder in CLASS_FOLDERS.items():
        try:
            X, _ = load_class_data(data_dir, folder, args.n_samples, args.seed)
            X_by_class[cls_id] = X
            print(f"    {folder:20s} (label {cls_id}): {len(X):6d} events")
        except FileNotFoundError as e:
            print(f"    Warning: {e} — skipping.")

    if not X_by_class:
        raise RuntimeError("No data loaded. Check --data-dir and class folder names.")

    # ── 5. Extract embeddings + attention maps ───────────────────────────────
    print("\n[5] Extracting embeddings and attention maps...")
    emb_by_class: Dict[int, torch.Tensor] = {}
    # {cls_id: {layer_idx: Tensor[N, heads, S, S]}}
    attn_raw: Dict[int, Dict[int, torch.Tensor]] = {}

    for cls_id, X in X_by_class.items():
        cname = CLASS_NAMES.get(cls_id, str(cls_id))
        print(f"    {cname}...")
        emb, attn = extract_embeddings_and_attention(
            encoder, extractor, X,
            batch_size=args.batch_size, device=args.device,
        )
        emb_by_class[cls_id] = emb
        attn_raw[cls_id] = attn

    extractor.remove()

    # head-averaged attention: {cls_id: {layer: np.ndarray [N, S, S]}}
    attn_mean: Dict[int, Dict[int, np.ndarray]] = {
        cls_id: {
            layer: _attn_head_mean(attn_raw[cls_id][layer])
            for layer in attn_raw[cls_id]
        }
        for cls_id in attn_raw
    }

    # ── 6. AE reconstruction errors + classification ─────────────────────────
    errors_by_class: Dict[int, np.ndarray] = {}
    # {cls_id: {layer: np.ndarray [N_subset, S, S]}}
    attn_correct: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    attn_missed:  Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)

    if ae_net is not None:
        print("\n[6] Computing AE reconstruction errors...")

        for cls_id, emb in emb_by_class.items():
            cname = CLASS_NAMES.get(cls_id, str(cls_id))
            errors = compute_ae_errors(ae_net, emb, batch_size=512, device=args.device)
            errors_by_class[cls_id] = errors
            print(f"    {cname:10s}  mean MSE = {errors.mean():.6f} ± {errors.std():.6f}")

        # derive threshold from QCD if not in checkpoint
        if not thresholds and QCD_CLASS in errors_by_class:
            print("    Deriving thresholds from loaded QCD errors...")
            for fpr_val in [0.01, 0.05, 0.10]:
                thr = float(np.quantile(errors_by_class[QCD_CLASS], 1.0 - fpr_val))
                thresholds[fpr_val] = thr
                print(f"      FPR={fpr_val:.0%}  threshold = {thr:.6f}")

        threshold = thresholds.get(args.fpr)
        if threshold is None:
            # try nearest available key
            nearest = min(thresholds, key=lambda k: abs(k - args.fpr))
            threshold = thresholds[nearest]
            print(f"    FPR={args.fpr} not in thresholds; using nearest FPR={nearest:.0%}")

        print(f"\n    Classification at FPR={args.fpr:.0%}, threshold={threshold:.6f}")
        print(f"    {'Class':12s}  {'N':>6}  {'TP/TN':>6}  {'FP/FN':>6}  {'Acc':>7}")
        print(f"    {'-'*46}")

        for cls_id in sorted(errors_by_class):
            cname = CLASS_NAMES.get(cls_id, str(cls_id))
            errors = errors_by_class[cls_id]
            is_anomaly_pred = errors > threshold
            is_anomaly_true = cls_id != QCD_CLASS

            correct_mask = is_anomaly_pred == is_anomaly_true
            missed_mask  = ~correct_mask
            n_tot = len(errors)
            n_ok  = correct_mask.sum()
            print(f"    {cname:12s}  {n_tot:>6}  {n_ok:>6}  {missed_mask.sum():>6}  {100*n_ok/n_tot:>6.1f}%")

            for layer_idx in range(n_layers):
                if layer_idx not in attn_mean[cls_id]:
                    continue
                a = attn_mean[cls_id][layer_idx]  # [N, S, S]
                if correct_mask.sum() > 0:
                    attn_correct[cls_id][layer_idx] = a[correct_mask]
                if missed_mask.sum() > 0:
                    attn_missed[cls_id][layer_idx] = a[missed_mask]

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    print("\n[7] Generating plots...")

    last_layer = n_layers - 1

    for higgs_id, higgs_name in HIGGS_CLASSES.items():
        if higgs_id not in attn_mean:
            continue
        tag = higgs_name.replace(" ", "_")

        # 7a. Mean attention by class (QCD vs this Higgs)
        plot_mean_attention_per_higgs(
            higgs_id,
            attn_qcd=attn_mean[QCD_CLASS],
            attn_higgs=attn_mean[higgs_id],
            token_labels=token_labels, spans=spans, n_layers=n_layers,
            output_path=out / f"mean_attn_{tag}.png",
        )

        # 7b. Individual event samples (head-averaged, last layer)
        if last_layer in attn_raw.get(QCD_CLASS, {}) and last_layer in attn_raw.get(higgs_id, {}):
            plot_sample_events(
                higgs_id,
                attn_qcd=attn_raw[QCD_CLASS],
                attn_higgs=attn_raw[higgs_id],
                token_labels=token_labels, spans=spans,
                last_layer=last_layer,
                n_events=args.n_sample_events,
                output_path=out / f"sample_events_{tag}_layer{last_layer}.png",
            )

        # 7c. Mean attention by classification outcome
        plot_mean_attention_by_outcome(
            higgs_id,
            dict(attn_correct), dict(attn_missed),
            token_labels, spans, n_layers,
            fpr=args.fpr,
            output_path=out / f"attn_by_outcome_{tag}_fpr{int(args.fpr * 100):02d}pct.png",
        )

    # 7d. MSE distributions
    plot_mse_distributions(
        errors_by_class, thresholds,
        output_path=out / "mse_distributions.png",
    )

    # ── 8. Save summary ───────────────────────────────────────────────────────
    summary = {
        "supcon_ckpt": str(args.supcon_ckpt),
        "ae_ckpt": str(args.ae_ckpt),
        "n_tokens": n_tokens,
        "token_labels": token_labels,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "fpr": args.fpr,
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "n_events": {CLASS_NAMES.get(k, str(k)): len(v) for k, v in X_by_class.items()},
        "mean_mse": {
            CLASS_NAMES.get(k, str(k)): float(v.mean())
            for k, v in errors_by_class.items()
        },
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[8] Summary written to {summary_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
