"""
Generate UMAP plots of encoder embeddings after the anomaly pipeline.

Loads embeddings from two sources within the pipeline output directory:
  - Phase 1 probe eval  : all 15 background classes
  - Phase 2 extraction  : QCD (normal) + Higgs anomaly classes

Produces:
  1. Combined UMAP — all classes + centroids, background vs anomaly coloured
  2. Per-class subplot grid — each class highlighted on grey background

Usage:
    python scripts/plot_embedding_umap.py \
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/aug_supcon_12class_nosparse_dmodel128_cern \
        --split test \
        --max-samples 5000 \
        --out-dir /eos/user/d/dgenoves/anomaly_pipeline/aug_supcon_12class_nosparse_dmodel128_cern/umap_plots
"""

import argparse
import math
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Class registry (18-class full dataset) ────────────────────────────────────

CLASS_NAMES = [
    "QCD_inclusive",       # 0  — normal (used for AE training)
    "DY_to_ll",            # 1
    "Z_to_vv_jet",         # 2
    "Z_to_qq_uds",         # 3
    "Z_to_bb",             # 4
    "Z_to_cc",             # 5
    "W_to_lv",             # 6
    "W_to_qq",             # 7
    "gamma",               # 8
    "QCD_bb",              # 9
    "tt_all-hadr",         # 10
    "tt_semi-lept",        # 11
    "tt_all-lept",         # 12
    "Minbias_Soft_QCD",    # 13
    "Upsilon_to_Leptons",  # 14
    "VBFHbb",              # 15 — anomaly
    "HH_4b",               # 16 — anomaly
    "ggHtautau",           # 17 — anomaly
]

ANOMALY_INDICES = {15, 16, 17}
BACKGROUND_INDICES = set(range(15))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_embeddings(path: Path, max_samples: int = None, rng=None):
    """Load embeddings and labels from an npz file, optionally subsampling."""
    data = np.load(path)
    emb = data["embeddings"]
    lbl = data["labels"]
    if max_samples and len(emb) > max_samples:
        idx = rng.choice(len(emb), max_samples, replace=False)
        emb, lbl = emb[idx], lbl[idx]
    return emb, lbl


def collect_embeddings(output_dir: Path, split: str, max_per_class: int, rng):
    """
    Collect embeddings from:
      - phase1/probe_evaluation/embeddings_debug.npz  (15 background classes)
      - embeddings/{split}_embeddings.npz              (QCD + Higgs)

    Returns (embeddings, labels) arrays with original class indices as labels.
    """
    all_emb, all_lbl = [], []

    # Phase 1: background classes (all 15)
    p1_path = output_dir / "phase1" / "probe_evaluation" / "visualizations" / "embeddings_debug.npz"
    if p1_path.exists():
        emb, lbl = load_embeddings(p1_path)
        # Subsample per class
        for cls in np.unique(lbl):
            mask = lbl == cls
            e, l = emb[mask], lbl[mask]
            if len(e) > max_per_class:
                idx = rng.choice(len(e), max_per_class, replace=False)
                e, l = e[idx], l[idx]
            all_emb.append(e)
            all_lbl.append(l)
        print(f"Loaded {len(emb)} background embeddings from Phase 1 probe eval")
    else:
        print(f"WARNING: Phase 1 embeddings not found at {p1_path}")

    # Phase 2: QCD + Higgs embeddings
    p2_path = output_dir / "embeddings" / f"{split}_embeddings.npz"
    if p2_path.exists():
        emb, lbl = load_embeddings(p2_path)
        # Only keep anomaly classes (QCD already covered by phase1)
        for cls in np.unique(lbl):
            if int(cls) not in ANOMALY_INDICES:
                continue
            mask = lbl == cls
            e, l = emb[mask], lbl[mask]
            if len(e) > max_per_class:
                idx = rng.choice(len(e), max_per_class, replace=False)
                e, l = e[idx], l[idx]
            all_emb.append(e)
            all_lbl.append(l)
        print(f"Loaded anomaly embeddings from Phase 2 ({split} split)")
    else:
        # Fallback: use phase2 embeddings for everything
        print(f"WARNING: Phase 2 embeddings not found at {p2_path}, trying fallback")
        p2_path_fallback = output_dir / "embeddings" / "test_embeddings.npz"
        if p2_path_fallback.exists():
            emb, lbl = load_embeddings(p2_path_fallback)
            for cls in np.unique(lbl):
                mask = lbl == cls
                e, l = emb[mask], lbl[mask]
                if len(e) > max_per_class:
                    idx = rng.choice(len(e), max_per_class, replace=False)
                    e, l = e[idx], l[idx]
                all_emb.append(e)
                all_lbl.append(l)

    if not all_emb:
        raise FileNotFoundError(
            f"No embeddings found in {output_dir}. "
            "Run the full anomaly pipeline first."
        )

    return np.concatenate(all_emb), np.concatenate(all_lbl)


def compute_umap(embeddings, n_neighbors, min_dist, metric):
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")
    print(f"Computing UMAP on {len(embeddings)} points...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        n_components=2,
        init="random",
    )
    return reducer.fit_transform(embeddings)


def _class_color(idx, n_background, n_anomaly):
    """Return color for a class: tab20 for background, red shades for anomaly."""
    if idx in ANOMALY_INDICES:
        anomaly_list = sorted(ANOMALY_INDICES)
        i = anomaly_list.index(idx)
        return plt.cm.Reds(0.5 + 0.2 * i)
    else:
        bg_list = sorted(BACKGROUND_INDICES)
        i = bg_list.index(idx) if idx in bg_list else 0
        return plt.cm.tab20(i / max(len(bg_list) - 1, 1))


# ── Plot 1: Combined UMAP ─────────────────────────────────────────────────────

def plot_combined(embedding_2d, labels, title, output_path):
    unique = sorted(np.unique(labels).tolist())
    n_bg = sum(1 for u in unique if u not in ANOMALY_INDICES)

    fig, ax = plt.subplots(figsize=(14, 10))

    for cls in unique:
        mask = labels == cls
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        color = _class_color(cls, n_bg, len(ANOMALY_INDICES))
        is_anomaly = cls in ANOMALY_INDICES

        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=[color],
            label=name,
            alpha=0.5 if not is_anomaly else 0.8,
            s=6 if not is_anomaly else 12,
            edgecolors='none',
            zorder=2 if is_anomaly else 1,
        )

        # Centroid
        cx, cy = embedding_2d[mask, 0].mean(), embedding_2d[mask, 1].mean()
        marker = '*' if is_anomaly else 'D'
        size = 250 if is_anomaly else 120
        ax.scatter(cx, cy, c=[color], s=size, marker=marker,
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(
            name.replace("_", " "),
            (cx, cy), fontsize=6.5,
            xytext=(4, 4), textcoords='offset points',
            zorder=6,
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ncol = 2 if len(unique) > 8 else 1
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
              ncol=ncol, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    # Anomaly legend note
    ax.plot([], [], 'k*', ms=10, label='centroid (anomaly)')
    ax.plot([], [], 'kD', ms=6, label='centroid (background)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined UMAP saved to: {output_path}")


# ── Plot 2: Per-class subplots ────────────────────────────────────────────────

def plot_per_class(embedding_2d, labels, title, output_dir):
    """Save one PNG per class — all points in grey, class highlighted."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique = sorted(np.unique(labels).tolist())
    n = len(unique)

    for cls in unique:
        mask = labels == cls
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        color = _class_color(cls, n, len(ANOMALY_INDICES))
        is_anomaly = cls in ANOMALY_INDICES

        fig, ax = plt.subplots(figsize=(8, 7))

        # All other points in light grey
        ax.scatter(
            embedding_2d[~mask, 0], embedding_2d[~mask, 1],
            c='lightgrey', alpha=0.2, s=4, edgecolors='none', label='other classes'
        )
        # Highlighted class
        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=[color], alpha=0.75, s=8, edgecolors='none', label=name
        )
        # Centroid
        cx, cy = embedding_2d[mask, 0].mean(), embedding_2d[mask, 1].mean()
        marker = '*' if is_anomaly else 'D'
        ax.scatter(cx, cy, c=[color], s=300 if is_anomaly else 150,
                   marker=marker, edgecolors='black', linewidths=1.0,
                   zorder=5, label='centroid')

        border_color = 'crimson' if is_anomaly else 'steelblue'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5 if is_anomaly else 1.2)

        tag = " [ANOMALY]" if is_anomaly else ""
        ax.set_title(f"{title}\n{name.replace('_', ' ')}{tag}",
                     fontsize=11, fontweight='bold', color=border_color)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)

        fname = f"umap_{cls:02d}_{name}.png"
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Per-class plots saved to: {output_dir}  ({n} files)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate UMAP plots from anomaly pipeline embeddings"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Root output directory of the anomaly pipeline run"
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which embedding split to visualise (default: test)"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to save plots (default: <output-dir>/umap_plots)"
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    out_dir = Path(args.out_dir) if args.out_dir else output_dir / "umap_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    encoder_name = output_dir.name

    for n_samples in [2000, 5000, 10000]:
        print(f"\n{'='*60}")
        print(f"Generating UMAP with {n_samples} samples/class")
        print(f"{'='*60}")

        sample_dir = out_dir / f"samples_{n_samples}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        embeddings, labels = collect_embeddings(
            output_dir, args.split, n_samples, rng
        )
        print(f"Total points: {len(embeddings)}  |  classes: {sorted(np.unique(labels).tolist())}")

        embedding_2d = compute_umap(
            embeddings, args.n_neighbors, args.min_dist, args.metric
        )

        title = f"Embedding Space — {encoder_name} ({args.split}, {n_samples}/class)"

        plot_combined(embedding_2d, labels, title, sample_dir / "umap_combined.png")
        plot_per_class(embedding_2d, labels, title, sample_dir / "per_class")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
