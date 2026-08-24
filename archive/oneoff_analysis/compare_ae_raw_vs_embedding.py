"""
Compare anomaly detection: AE on raw inputs  vs  AE on SupCon embeddings.

This script:
  1. Trains (or loads) an AE on raw features (anomaly_qcd_vs_higgs_raw_cern config).
  2. Loads the embedding-based AE results from a pretrained encoder.
  3. Prints a side-by-side comparison table and saves ROC + MSE comparison plots.

Usage (train raw AE from scratch + compare):
    python scripts/compare_ae_raw_vs_embedding.py \\
        --embedding-ae-ckpt /eos/.../phase2/checkpoints/ae-.../separation_ratio=X.ckpt \\
        --embedding-test-npz /eos/.../embeddings/test_embeddings.npz \\
        --output-dir /eos/.../raw_vs_embedding_comparison

Usage (load pre-trained raw AE, skip training):
    python scripts/compare_ae_raw_vs_embedding.py \\
        --embedding-ae-ckpt /eos/.../separation_ratio=X.ckpt \\
        --embedding-test-npz /eos/.../test_embeddings.npz \\
        --raw-ae-ckpt /eos/.../raw_ae/checkpoints/ae-best.ckpt \\
        --raw-data-dir /eos/.../foundation_model_testing_data/v2_18class_highlevel_full/preprocessed/test \\
        --output-dir /eos/.../raw_vs_embedding_comparison
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.autoencoder import AutoencoderLitModule, compute_errors_numpy, derive_thresholds
from sklearn.metrics import roc_auc_score, roc_curve

CLASS_NAMES   = {0: "QCD_inclusive", 15: "VBFHbb", 16: "HH_4b", 17: "ggHtautau"}
CLASS_COLORS  = {0: "steelblue", 15: "tomato", 16: "mediumorchid", 17: "goldenrod"}
NORMAL_CLASS  = 0
ANOMALY_CLASSES = [15, 16, 17]
TARGET_FPRS   = [0.01, 0.05, 0.10]


# ─────────────────────────────────────────────────────────────────────────────
# Raw AE training
# ─────────────────────────────────────────────────────────────────────────────

from src.data.raw_npy_datamodule import RawNpyDataModule  # noqa: E402


def train_raw_ae(preprocessed_dir: Path,
                 output_dir: Path,
                 seed: int = 42,
                 max_epochs: int = 50,
                 compression: int = 4,
                 depth: int = 4,
                 dropout: float = 0.03473,
                 lr: float = 1.965e-4,
                 weight_decay: float = 3.023e-5,
                 monitor: str = "ae/val_drift_metric") -> str:
    """
    Train an AE on raw features by loading preprocessed .npy shards directly.
    Returns the path to the best checkpoint.
    """
    torch.set_float32_matmul_precision("high")
    L.seed_everything(seed, workers=True)

    dm = RawNpyDataModule(
        preprocessed_dir=preprocessed_dir,
        normal_classes=[NORMAL_CLASS],
        anomaly_classes=ANOMALY_CLASSES,
        n_train=100_000, n_val=20_000, n_test=20_000,
        batch_size=512, seed=seed,
    )
    input_dim = dm._probe_input_dim()
    print(f"    Raw input dimension: {input_dim}")

    ae = AutoencoderLitModule(
        input_dim=input_dim,
        compression=compression,
        depth=depth,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        normal_classes_labels=[NORMAL_CLASS],
        anomaly_classes_labels=ANOMALY_CLASSES,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="ae-raw-epoch{epoch:02d}",
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=20,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[ckpt_cb, early_stop],
        deterministic=True,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    print(f"\n    Training raw AE (max_epochs={max_epochs}, compression={compression}, depth={depth})...")
    trainer.fit(ae, datamodule=dm)
    trainer.test(ae, datamodule=dm, ckpt_path="best", weights_only=False)

    best_ckpt = ckpt_cb.best_model_path
    best_qcd_loss = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    print(f"    Best raw AE checkpoint     : {best_ckpt}")
    print(f"    Best val QCD loss (monitor): {best_qcd_loss:.6f}")
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Load raw test data from preprocessed shards
# ─────────────────────────────────────────────────────────────────────────────

CLASS_FOLDERS = {0: "QCD_HT50toInf", 15: "VBFHbb", 16: "HH_4b", 17: "ggHtautau"}


def load_raw_test_data(data_dir: Path, n_samples: int = 20_000,
                       seed: int = 42, class_folders: dict = None) -> dict:
    """Load raw feature arrays from preprocessed test split."""
    rng = np.random.default_rng(seed)
    folders = class_folders if class_folders is not None else CLASS_FOLDERS
    data = {}
    for cls, folder in folders.items():
        cls_dir = data_dir / folder
        if not cls_dir.exists():
            print(f"    Warning: {cls_dir} not found — skipping class {cls}.")
            continue
        x_files = sorted(cls_dir.glob("*_x.npy"))
        if not x_files:
            print(f"    Warning: no *_x.npy files in {cls_dir} — skipping.")
            continue
        xs = [np.load(f) for f in x_files]
        X_all = np.concatenate(xs, axis=0)
        n = min(n_samples, len(X_all))
        idx = rng.choice(len(X_all), size=n, replace=False)
        data[cls] = X_all[idx]
        print(f"    {CLASS_NAMES[cls]:<20}: {len(data[cls])} raw events, shape {data[cls].shape}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_for_method(errors_by_class: dict, val_thresholds: dict,
                                method_name: str) -> dict:
    normal_e = errors_by_class.get(NORMAL_CLASS, np.array([]))
    all_anom = np.concatenate([errors_by_class.get(c, np.array([]))
                               for c in ANOMALY_CLASSES if c in errors_by_class])

    sep = float(all_anom.mean() / normal_e.mean()) if len(normal_e) > 0 and len(all_anom) > 0 else float("nan")

    auc_per_class = {}
    for cls in ANOMALY_CLASSES:
        if cls not in errors_by_class or len(normal_e) == 0:
            continue
        y_t = np.concatenate([np.zeros(len(normal_e)), np.ones(len(errors_by_class[cls]))])
        y_s = np.concatenate([normal_e, errors_by_class[cls]])
        auc_per_class[cls] = float(roc_auc_score(y_t, y_s))
    if len(normal_e) > 0 and len(all_anom) > 0:
        y_t = np.concatenate([np.zeros(len(normal_e)), np.ones(len(all_anom))])
        y_s = np.concatenate([normal_e, all_anom])
        auc_per_class["all"] = float(roc_auc_score(y_t, y_s))

    eff_per_fpr = {}
    for fpr_t in TARGET_FPRS:
        thr = val_thresholds.get(fpr_t, np.quantile(normal_e, 1.0 - fpr_t))
        eff_per_fpr[fpr_t] = {}
        for cls in ANOMALY_CLASSES:
            if cls not in errors_by_class:
                continue
            e = errors_by_class[cls]
            eff_per_fpr[fpr_t][cls] = float((e > thr).sum()) / len(e)

    drift = {}
    if len(normal_e) > 0:
        N, eps = len(normal_e), 0.5 / len(normal_e)
        for fpr_t in TARGET_FPRS:
            thr = val_thresholds.get(fpr_t, np.quantile(normal_e, 1.0 - fpr_t))
            p_hat = float((normal_e > thr).sum()) / N
            drift[fpr_t] = abs(math.log((p_hat + eps) / (fpr_t + eps)))
        drift["avg"] = float(np.mean([drift[f] for f in TARGET_FPRS]))

    return {
        "method": method_name,
        "separation_ratio": sep,
        "auc": auc_per_class,
        "efficiency": eff_per_fpr,
        "drift": drift,
        "mse": {cls: {"mean": float(e.mean()), "std": float(e.std())}
                for cls, e in errors_by_class.items()},
    }


def print_comparison(raw_metrics: dict, emb_metrics: dict):
    print("\n" + "=" * 80)
    print("COMPARISON: AE on Raw Inputs  vs  AE on SupCon Embeddings")
    print("=" * 80)

    def _row(label, raw_val, emb_val, fmt=".4f"):
        delta = ""
        if isinstance(raw_val, float) and isinstance(emb_val, float) and not math.isnan(raw_val + emb_val):
            diff = emb_val - raw_val
            delta = f"  (Δ = {diff:+.4f})"
        print(f"  {label:<40} {raw_val:{fmt}}  →  {emb_val:{fmt}}{delta}")

    print(f"\n{'Metric':<40} {'Raw AE':>10}     {'Emb AE':>10}")
    print("-" * 65)
    _row("Separation ratio",       raw_metrics["separation_ratio"], emb_metrics["separation_ratio"])
    _row("AUC-ROC (all Higgs)",    raw_metrics["auc"].get("all", float("nan")), emb_metrics["auc"].get("all", float("nan")))
    for cls in ANOMALY_CLASSES:
        _row(f"  AUC-ROC {CLASS_NAMES[cls]}",
             raw_metrics["auc"].get(cls, float("nan")),
             emb_metrics["auc"].get(cls, float("nan")))

    print(f"\n  --- Signal Efficiency (TPR) ---")
    for fpr_t in TARGET_FPRS:
        for cls in ANOMALY_CLASSES:
            raw_eff = raw_metrics["efficiency"].get(fpr_t, {}).get(cls, float("nan"))
            emb_eff = emb_metrics["efficiency"].get(fpr_t, {}).get(cls, float("nan"))
            _row(f"  FPR={fpr_t:.0%}  {CLASS_NAMES[cls]}", raw_eff, emb_eff)
        print()

    print(f"\n  --- Drift Metrics ---")
    for fpr_t in TARGET_FPRS:
        _row(f"  Drift @ FPR={fpr_t:.0%}",
             raw_metrics["drift"].get(fpr_t, float("nan")),
             emb_metrics["drift"].get(fpr_t, float("nan")))
    _row("  Average drift", raw_metrics["drift"].get("avg", float("nan")), emb_metrics["drift"].get("avg", float("nan")))
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(raw_by_class: dict, emb_by_class: dict,
                    raw_thresholds: dict, emb_thresholds: dict,
                    output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def _roc_panel(ax, errors_by_class, title):
        normal_e = errors_by_class.get(NORMAL_CLASS, np.array([]))
        for cls in ANOMALY_CLASSES:
            if cls not in errors_by_class or len(normal_e) == 0:
                continue
            y_t = np.concatenate([np.zeros(len(normal_e)), np.ones(len(errors_by_class[cls]))])
            y_s = np.concatenate([normal_e, errors_by_class[cls]])
            fpr, tpr, _ = roc_curve(y_t, y_s)
            auc = roc_auc_score(y_t, y_s)
            ax.plot(fpr, tpr, label=f"{CLASS_NAMES[cls]} (AUC={auc:.3f})",
                    color=CLASS_COLORS[cls], linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR (Signal Eff.)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _mse_panel(ax, errors_by_class, thresholds, title):
        all_e = np.concatenate(list(errors_by_class.values()))
        bins  = np.linspace(all_e.min(), np.percentile(all_e, 99.5), 80)
        for cls, e in errors_by_class.items():
            ax.hist(e, bins=bins, alpha=0.55, label=CLASS_NAMES.get(cls, str(cls)),
                    color=CLASS_COLORS.get(cls, "gray"), density=True, histtype="stepfilled")
        for fpr_t, thr in sorted(thresholds.items()):
            ax.axvline(thr, linestyle="--", linewidth=1.5, label=f"FPR={fpr_t:.0%}")
        ax.set_xlabel("Reconstruction MSE")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _eff_panel(ax, raw_by_c, emb_by_c, raw_thr, emb_thr, fpr_target):
        x = np.arange(len(ANOMALY_CLASSES))
        bar_w = 0.35
        raw_effs, emb_effs = [], []
        for cls in ANOMALY_CLASSES:
            thr_r = raw_thr.get(fpr_target, np.quantile(raw_by_c[NORMAL_CLASS], 1 - fpr_target))
            thr_e = emb_thr.get(fpr_target, np.quantile(emb_by_c[NORMAL_CLASS], 1 - fpr_target))
            er = raw_by_c.get(cls, np.array([]))
            ee = emb_by_c.get(cls, np.array([]))
            raw_effs.append(float((er > thr_r).sum()) / len(er) if len(er) else 0)
            emb_effs.append(float((ee > thr_e).sum()) / len(ee) if len(ee) else 0)
        ax.bar(x - bar_w / 2, raw_effs, bar_w, label="Raw AE", color="slategray", alpha=0.8)
        ax.bar(x + bar_w / 2, emb_effs, bar_w, label="Emb AE", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_NAMES[c] for c in ANOMALY_CLASSES], fontsize=9)
        ax.set_ylabel("Signal Efficiency (TPR)")
        ax.set_title(f"Signal efficiency @ FPR={fpr_target:.0%}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    _roc_panel(axes[0, 0], raw_by_class, "ROC — Raw AE")
    _roc_panel(axes[1, 0], emb_by_class, "ROC — Embedding AE (Vanilla SupCon)")
    _mse_panel(axes[0, 1], raw_by_class, raw_thresholds, "MSE distributions — Raw AE")
    _mse_panel(axes[1, 1], emb_by_class, emb_thresholds, "MSE distributions — Embedding AE")
    _eff_panel(axes[0, 2], raw_by_class, emb_by_class, raw_thresholds, emb_thresholds, 0.05)
    _eff_panel(axes[1, 2], raw_by_class, emb_by_class, raw_thresholds, emb_thresholds, 0.01)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "raw_vs_embedding_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare raw-input AE vs embedding AE for anomaly detection"
    )
    p.add_argument("--embedding-ae-ckpt",  required=True,
                   help="AE checkpoint trained on SupCon embeddings")
    p.add_argument("--embedding-test-npz", required=True,
                   help="test_embeddings.npz produced by the pretrained encoder")
    p.add_argument("--output-dir",         required=True)

    # Raw AE: either provide pre-trained checkpoint + data, or let the script train
    raw_grp = p.add_mutually_exclusive_group(required=True)
    raw_grp.add_argument("--raw-ae-ckpt",
                         help="Pre-trained raw AE checkpoint (skip training)")
    raw_grp.add_argument("--train-raw-ae", action="store_true",
                         help="Train a fresh raw AE from scratch using anomaly_qcd_vs_higgs_raw_cern config")

    p.add_argument("--raw-data-dir", required=True,
                   help="Preprocessed root dir containing train/ val/ test/ splits "
                        "(e.g. .../v2_18class_highlevel_full/preprocessed)")
    p.add_argument("--raw-output-dir",
                   help="Output dir for raw AE training (defaults to <output-dir>/raw_ae)")
    p.add_argument("--n-samples", type=int, default=20_000,
                   help="Events per class for inference / comparison. Default: 20000")
    p.add_argument("--device",    default="cpu")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--drift-monitor", default="ae/val_drift_metric",
                   choices=["ae/val_drift_metric", "ae/val_drift_fpr01",
                            "ae/val_drift_fpr05", "ae/val_drift_fpr10"],
                   help="Metric to monitor for best checkpoint selection")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed, workers=True)

    # ── Embedding AE ──────────────────────────────────────────────────────────
    print(f"\n[EMB] Loading embedding AE checkpoint:\n    {args.embedding_ae_ckpt}")
    emb_ae = AutoencoderLitModule.load_from_checkpoint(
        args.embedding_ae_ckpt, map_location=args.device, weights_only=False
    )
    emb_ae.eval()
    emb_thresholds = dict(emb_ae._val_thresholds) if emb_ae._val_thresholds else {}

    print(f"\n[EMB] Loading test embeddings:\n    {args.embedding_test_npz}")
    emb_data = np.load(args.embedding_test_npz)
    emb_embeddings, emb_labels = emb_data["embeddings"], emb_data["labels"]
    emb_by_class = {}
    for cls in np.unique(emb_labels):
        mask = emb_labels == cls
        if int(cls) not in [NORMAL_CLASS] + ANOMALY_CLASSES:
            continue
        emb_by_class[int(cls)] = compute_errors_numpy(
            emb_ae, emb_embeddings[mask], device=args.device)
    if not emb_thresholds and NORMAL_CLASS in emb_by_class:
        emb_thresholds = derive_thresholds(emb_by_class[NORMAL_CLASS])

    # ── Raw AE ────────────────────────────────────────────────────────────────
    raw_out = Path(args.raw_output_dir) if args.raw_output_dir else out / "raw_ae"

    preprocessed_dir = Path(args.raw_data_dir)

    if args.train_raw_ae:
        print("\n[RAW] Training raw AE from scratch...")
        raw_out.mkdir(parents=True, exist_ok=True)
        raw_ckpt_path = train_raw_ae(
            preprocessed_dir=preprocessed_dir,
            output_dir=raw_out,
            seed=args.seed,
            monitor=args.drift_monitor,
        )
    else:
        raw_ckpt_path = args.raw_ae_ckpt

    print(f"\n[RAW] Loading raw AE checkpoint:\n    {raw_ckpt_path}")
    raw_ae = AutoencoderLitModule.load_from_checkpoint(
        raw_ckpt_path, map_location=args.device, weights_only=False
    )
    raw_ae.eval()
    raw_thresholds = dict(raw_ae._val_thresholds) if raw_ae._val_thresholds else {}

    # Load raw test data for inference
    test_dir = preprocessed_dir / "test"
    print(f"\n[RAW] Loading raw test data from:\n    {test_dir}")
    raw_by_class_raw = load_raw_test_data(
        test_dir, n_samples=args.n_samples, seed=args.seed
    )
    raw_by_class: dict = {}
    for cls, X in raw_by_class_raw.items():
        raw_by_class[cls] = compute_errors_numpy(raw_ae, X, device=args.device)
        print(f"    {CLASS_NAMES[cls]}: mean MSE = {raw_by_class[cls].mean():.6f}")
    if not raw_thresholds and NORMAL_CLASS in raw_by_class:
        raw_thresholds = derive_thresholds(raw_by_class[NORMAL_CLASS])

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n[METRICS] Computing comparison metrics...")
    raw_metrics = compute_metrics_for_method(raw_by_class, raw_thresholds, "raw_ae")
    emb_metrics = compute_metrics_for_method(emb_by_class, emb_thresholds, "embedding_ae")

    print_comparison(raw_metrics, emb_metrics)

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {"raw_ae": raw_metrics, "embedding_ae": emb_metrics}
    path = out / "comparison_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Comparison summary saved: {path}")

    plot_comparison(raw_by_class, emb_by_class, raw_thresholds, emb_thresholds, out)


if __name__ == "__main__":
    main()
