"""
Plot ablation study results from MLflow.

Queries the 'aug_supcon_ablation' (or 'aug_supcon_ablation') MLflow experiment
and produces per-parameter sensitivity plots.

Usage:
    python scripts/plot_ablation_results.py \
        --experiment aug_supcon_ablation \
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/ablation/aug/plots

    # Encoder ablation only:
    python scripts/plot_ablation_results.py \
        --experiment aug_supcon_ablation \
        --mode encoder

    # AE ablation only:
    python scripts/plot_ablation_results.py \
        --experiment aug_supcon_ablation \
        --mode ae
"""

import argparse
import re
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Encoder metrics to plot (Phase 1)
ENCODER_METRICS = {
    "linear_probe_accuracy":    "Linear Probe Accuracy",
    "linear_probe_auroc_macro": "Linear Probe AUROC (macro)",
    "linear_probe_f1_macro":    "Linear Probe F1 (macro)",
    "val/silhouette":           "Silhouette Score",
    "val/con_loss":             "Val Contrastive Loss",
}

# AE / anomaly detection metrics to plot (Phase 2)
AE_METRICS = {
    "ae/val_loss_normal":   "AE Val MSE (normal/QCD)",
    "ae/val_loss_anomaly":  "AE Val MSE (anomaly/Higgs)",
    "ae/separation_ratio":  "Separation Ratio (anomaly/normal MSE)",
    "ae/drift_metric":      "Drift Metric (avg over FPRs)",
    "ae/drift_fpr01":       "Drift @ FPR 1%",
    "ae/drift_fpr05":       "Drift @ FPR 5%",
    "ae/val_drift_metric":  "Val Drift Metric (unbiased)",
}

# Ordering and display labels for ablation parameters
VANILLA_ENCODER_PARAM_ORDER = [
    "d_model", "n_heads", "num_layers", "d_ff_mult",
    "dropout", "projection_dim", "hidden_projection_dim",
    "temperature", "lr", "weight_decay", "batch_size",
]
AUG_ENCODER_PARAM_ORDER = [
    "d_model", "n_heads", "num_layers", "d_ff_mult",
    "dropout", "projection_dim", "hidden_projection_dim",
    "temperature", "lr", "weight_decay", "batch_size",
    "mask_probability", "num_augmentations", "mask_full_particle",
]
AE_PARAM_ORDER = [
    "ae_compression", "ae_depth", "ae_dropout",
    "ae_lr", "ae_weight_decay", "ae_batch_size",
]

PARAM_LABELS = {
    "d_model":                 "d_model",
    "n_heads":                 "n_heads",
    "num_layers":              "num_layers",
    "d_ff_mult":               "d_ff_mult",
    "dropout":                 "dropout",
    "projection_dim":          "proj_dim",
    "hidden_projection_dim":   "hidden_proj_dim",
    "temperature":             "temperature",
    "lr":                      "lr",
    "weight_decay":            "weight_decay",
    "batch_size":              "batch_size",
    "mask_probability":        "mask_prob",
    "num_augmentations":       "num_augs",
    "mask_full_particle":      "mask_full_particle",
    "ae_compression":          "AE compression",
    "ae_depth":                "AE depth",
    "ae_dropout":              "AE dropout",
    "ae_lr":                   "AE lr",
    "ae_weight_decay":         "AE weight_decay",
    "ae_batch_size":           "AE batch_size",
}

# Anchor is identified by empty PHASE1_OVERRIDE / PHASE2_OVERRIDE
ANCHOR_TAG = "ablation_anchor"  # or detect by run_name containing "__anchor"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_value(v: str) -> float:
    """Convert sanitized value strings back to float (0p05 → 0.05, 1em4 → 1e-4)."""
    if v.lower() == "true":
        return 1.0
    if v.lower() == "false":
        return 0.0
    v2 = v.replace("p", ".").replace("em", "e-").replace("ep", "e+")
    try:
        return float(v2)
    except ValueError:
        return float("nan")


def _value_label(value: float, value_str: str) -> str:
    """Human-readable x-axis label: use original string for booleans, else numeric."""
    if value_str.lower() in ("true", "false"):
        return value_str
    if np.isnan(value):
        return value_str
    return f"{value:.3g}"


def fetch_runs(experiment_name: str, tracking_uri: str) -> pd.DataFrame:
    """
    Fetch and merge ablation runs from MLflow.

    Each encoder ablation point produces two MLflow runs:
      - Phase 1: `..._enc__<param>__<value>`      → encoder metrics
      - Phase 2: `..._enc__<param>__<value>_ae`   → AE metrics

    These are merged into a single row per (param, value) so that
    per-parameter plots can show both encoder and AE metrics together.

    Pure AE ablation runs (`..._ae__<param>__<value>`) are kept as-is.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Collect all experiments with this name (MLflow may have created duplicates)
    all_exps = client.search_experiments()
    exp_ids = [e.experiment_id for e in all_exps if e.name == experiment_name]
    if not exp_ids:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found.")

    runs = client.search_runs(experiment_ids=exp_ids, max_results=5000)

    # Buckets: key = (param, value_str)
    enc_phase1: dict = {}   # encoder metrics (Phase 1)
    enc_phase2: dict = {}   # AE metrics from encoder ablation (Phase 2, _ae suffix)
    ae_runs: dict    = {}   # pure AE ablation runs

    for r in runs:
        name    = r.info.run_name or ""
        metrics = r.data.metrics
        run_id  = r.info.run_id
        status  = r.info.status

        # Phase 2 of encoder ablation  (ends with _ae)
        m = re.search(r"ablation_enc__(.+?)__(.+?)_ae$", name)
        if m:
            key = (m.group(1), m.group(2))
            prev = enc_phase2.get(key)
            if prev is None or (status == "FINISHED" and prev["status"] != "FINISHED") or run_id > prev["run_id"]:
                enc_phase2[key] = {"run_id": run_id, "status": status, **metrics}
            continue

        # Phase 1 of encoder ablation
        m = re.search(r"ablation_enc__(.+?)__(.+?)$", name)
        if m:
            key = (m.group(1), m.group(2))
            row = {
                "run_id": run_id, "run_name": name,
                "ablation_param": m.group(1), "ablation_value_str": m.group(2),
                "ablation_value": _sanitize_value(m.group(2)),
                "mode": "encoder", "status": status,
                **metrics,
            }
            prev = enc_phase1.get(key)
            if prev is None or (status == "FINISHED" and prev["status"] != "FINISHED") or run_id > prev["run_id"]:
                enc_phase1[key] = row
            continue

        # Pure AE ablation run
        m = re.search(r"ablation_ae__(.+?)__(.+?)$", name)
        if m:
            key = (m.group(1), m.group(2))
            row = {
                "run_id": run_id, "run_name": name,
                "ablation_param": m.group(1), "ablation_value_str": m.group(2),
                "ablation_value": _sanitize_value(m.group(2)),
                "mode": "ae", "status": status,
                **metrics,
            }
            prev = ae_runs.get(key)
            if prev is None or (status == "FINISHED" and prev["status"] != "FINISHED") or run_id > prev["run_id"]:
                ae_runs[key] = row
            continue

    # Merge Phase 1 + Phase 2 metrics for encoder ablation
    records = []
    all_keys = set(enc_phase1.keys()) | set(enc_phase2.keys())
    for key in all_keys:
        param, value_str = key
        row = enc_phase1.get(key, {
            "run_id": None, "run_name": "",
            "ablation_param": param, "ablation_value_str": value_str,
            "ablation_value": _sanitize_value(value_str),
            "mode": "encoder", "status": "MISSING",
        }).copy()
        # Add AE metrics from Phase 2 (don't overwrite Phase 1 keys)
        if key in enc_phase2:
            for k, v in enc_phase2[key].items():
                if k not in row:
                    row[k] = v
            # Mark finished only if both phases finished
            if row.get("status") == "FINISHED" and enc_phase2[key]["status"] != "FINISHED":
                row["status"] = "PARTIAL"
        records.append(row)

    records.extend(ae_runs.values())
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_param_sensitivity(
    df: pd.DataFrame,
    param: str,
    metrics: dict,
    output_dir: Path,
    mode: str,
):
    """One figure per parameter: subplots for each metric."""
    sub = df[(df["ablation_param"] == param) & (df["status"] == "FINISHED")].copy()
    if sub.empty:
        print(f"  [skip] {param}: no finished runs")
        return

    sub = sub.sort_values("ablation_value")
    available = [m for m in metrics if m in sub.columns and sub[m].notna().any()]
    if not available:
        print(f"  [skip] {param}: no metric data")
        return

    ncols = min(3, len(available))
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"Ablation: {PARAM_LABELS.get(param, param)}", fontsize=14, fontweight="bold")

    for idx, metric_key in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        x = sub["ablation_value"].values
        y = sub[metric_key].values

        # Identify anchor: the middle value (or whichever matches anchor config)
        # Simple heuristic: anchor is the value closest to the median x
        anchor_idx = int(np.argmin(np.abs(x - np.median(x))))

        colors = ["#d62728" if i == anchor_idx else "#1f77b4" for i in range(len(x))]
        ax.bar(range(len(x)), y, color=colors, edgecolor="black", linewidth=0.5)
        xlabels = [_value_label(xi, si) for xi, si in zip(x, sub["ablation_value_str"].values)]
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(metrics[metric_key], fontsize=10)
        ax.set_title(metrics[metric_key], fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Annotate anchor
        ax.get_xticklabels()[anchor_idx].set_color("red")
        ax.get_xticklabels()[anchor_idx].set_fontweight("bold")

    # Hide unused subplots
    for idx in range(len(available), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    anchor_patch = mpatches.Patch(color="#d62728", label="Anchor value")
    fig.legend(handles=[anchor_patch], loc="lower right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = output_dir / f"ablation_{mode}_{param}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fname_png = output_dir / f"ablation_{mode}_{param}.png"
    fig.savefig(fname_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_sensitivity_summary(
    df: pd.DataFrame,
    param_order: list,
    metrics: dict,
    output_dir: Path,
    mode: str,
    title: str,
):
    """
    Summary heatmap: rows = parameters, cols = metrics.
    Value = (best - anchor) / |anchor| normalised delta.
    """
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return

    delta_matrix = []
    param_labels_used = []

    for param in param_order:
        sub = df[(df["ablation_param"] == param) & (df["status"] == "FINISHED")].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("ablation_value")
        x = sub["ablation_value"].values
        anchor_idx = int(np.argmin(np.abs(x - np.median(x))))

        deltas = []
        for metric_key in available_metrics:
            if metric_key not in sub.columns:
                deltas.append(float("nan"))
                continue
            y = sub[metric_key].values
            anchor_val = y[anchor_idx]
            if np.isnan(anchor_val):
                deltas.append(float("nan"))
                continue
            deltas.append(float(np.nanmax(np.abs(y - anchor_val))))

        delta_matrix.append(deltas)
        param_labels_used.append(PARAM_LABELS.get(param, param))

    if not delta_matrix:
        return

    mat = np.array(delta_matrix)  # shape: (n_params, n_metrics), absolute deltas

    # Compute global range per metric across ALL params (union of all runs)
    # pct[i,j] = max|Δ from anchor| / (global_max - global_min) * 100
    pct_matrix = mat.copy()
    for j, metric_key in enumerate(available_metrics):
        all_vals = df[df["status"] == "FINISHED"][metric_key].dropna().values if metric_key in df.columns else np.array([])
        global_range = float(np.nanmax(all_vals) - np.nanmin(all_vals)) if len(all_vals) > 1 else float("nan")
        if np.isnan(global_range) or global_range == 0:
            pct_matrix[:, j] = float("nan")
        else:
            pct_matrix[:, j] = mat[:, j] / global_range * 100.0

    COLOR_CAP = 100.0
    fig, ax = plt.subplots(figsize=(max(6, len(available_metrics) * 1.5), max(4, len(param_labels_used) * 0.6)))
    im = ax.imshow(np.clip(pct_matrix, 0, COLOR_CAP), aspect="auto", cmap="YlOrRd", vmin=0, vmax=COLOR_CAP)
    plt.colorbar(im, ax=ax, label="% of metric range covered (capped at 100%)")

    ax.set_xticks(range(len(available_metrics)))
    ax.set_xticklabels([metrics[m] for m in available_metrics], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(param_labels_used)))
    ax.set_yticklabels(param_labels_used, fontsize=9)
    ax.set_title(f"{title} — Parameter Sensitivity\n(cell = % of metric's full range affected by this parameter)",
                 fontsize=11, fontweight="bold")

    for i in range(pct_matrix.shape[0]):
        for j in range(pct_matrix.shape[1]):
            v = pct_matrix[i, j]
            if not np.isnan(v):
                norm_v = min(v, COLOR_CAP) / COLOR_CAP
                label = f"{v:.0f}%"
                ax.text(j, i, label, ha="center", va="center", fontsize=7,
                        color="black" if norm_v < 0.6 else "white")

    plt.tight_layout()
    fname = output_dir / f"ablation_{mode}_sensitivity_heatmap.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fig.savefig(output_dir / f"ablation_{mode}_sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_metric_overview(
    df: pd.DataFrame,
    param_order: list,
    metric_key: str,
    metric_label: str,
    output_dir: Path,
    mode: str,
):
    """
    Overview: two bars per parameter — anchor (red) vs best (green).
    The best value label is annotated above its bar.
    Delta % vs anchor is shown as text between the bars.
    Higher is better for all metrics except losses.
    """
    lower_is_better_metrics = {"val/con_loss", "ae/val_loss_normal"}
    higher_is_better = metric_key not in lower_is_better_metrics and "drift" not in metric_key

    records = []
    for param in param_order:
        sub = df[(df["ablation_param"] == param) & (df["status"] == "FINISHED")].copy()
        if sub.empty or metric_key not in sub.columns:
            continue
        sub = sub.sort_values("ablation_value").reset_index(drop=True)
        x = sub["ablation_value"].values
        s = sub["ablation_value_str"].values
        y = sub[metric_key].values
        if np.all(np.isnan(y)):
            continue

        anchor_idx = int(np.argmin(np.abs(x - np.median(x))))
        anchor_val = y[anchor_idx]

        if higher_is_better:
            best_idx = int(np.nanargmax(y))
        else:
            best_idx = int(np.nanargmin(y))

        best_val = y[best_idx]
        best_label = _value_label(x[best_idx], s[best_idx])
        anchor_label = _value_label(x[anchor_idx], s[anchor_idx])

        delta_pct = 100 * (best_val - anchor_val) / abs(anchor_val) \
            if (not np.isnan(anchor_val) and anchor_val != 0) else float("nan")

        records.append({
            "param_label":   PARAM_LABELS.get(param, param),
            "anchor_val":    anchor_val,
            "anchor_label":  anchor_label,
            "best_val":      best_val,
            "best_label":    best_label,
            "delta_pct":     delta_pct,
            "is_same":       best_idx == anchor_idx,
        })

    if not records:
        return

    params      = [r["param_label"]  for r in records]
    anchor_vals = [r["anchor_val"]   for r in records]
    best_vals   = [r["best_val"]     for r in records]
    deltas      = [r["delta_pct"]    for r in records]
    best_labels = [r["best_label"]   for r in records]

    x_pos = np.arange(len(params))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(params) * 1.4), 5))

    bars_anchor = ax.bar(x_pos - width / 2, anchor_vals, width, color="#d62728",
                         label="Anchor", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_best   = ax.bar(x_pos + width / 2, best_vals,   width, color="#2ca02c",
                         label="Best",   edgecolor="black", linewidth=0.5, alpha=0.85)

    # Annotate best bar with its hyperparameter value
    for bar, label in zip(bars_best, best_labels):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001 * abs(bar.get_height() or 1),
                label, ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"Ablation Overview — {metric_label}\n"
                 f"(green bar label = best hyperparameter value found)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    metric_slug = metric_key.replace("/", "_")
    fname = output_dir / f"ablation_{mode}_overview_{metric_slug}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fig.savefig(output_dir / f"ablation_{mode}_overview_{metric_slug}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def export_table(df: pd.DataFrame, output_dir: Path, model_name: str):
    """Export a tidy CSV and a formatted markdown table of all ablation results."""
    all_metric_keys = list(ENCODER_METRICS.keys()) + list(AE_METRICS.keys())
    base_cols = ["mode", "ablation_param", "ablation_value", "status"]
    metric_cols = [m for m in all_metric_keys if m in df.columns]

    tidy = df[df["status"].isin(["FINISHED", "PARTIAL"])][base_cols + metric_cols].copy()
    tidy = tidy.sort_values(["mode", "ablation_param", "ablation_value"]).reset_index(drop=True)

    # Rename metric columns to short display names
    rename = {**ENCODER_METRICS, **AE_METRICS}
    tidy = tidy.rename(columns=rename)

    csv_path = output_dir / "ablation_results.csv"
    tidy.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved table: {csv_path}")

    # Also save a per-parameter summary: anchor value + best value + delta for key metrics
    key_metrics = ["ae/separation_ratio", "ae/drift_fpr05", "linear_probe_accuracy"]
    key_metrics = [m for m in key_metrics if m in df.columns]

    summary_rows = []
    for param in tidy["ablation_param"].unique():
        sub = df[(df["ablation_param"] == param) & (df["status"].isin(["FINISHED", "PARTIAL"]))].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("ablation_value")
        x = sub["ablation_value"].values
        anchor_idx = int(np.argmin(np.abs(x - np.median(x))))
        row = {"param": param, "anchor_value": x[anchor_idx]}
        for mk in key_metrics:
            if mk not in sub.columns:
                continue
            y = sub[mk].values
            if np.all(np.isnan(y)):
                continue
            anchor_val = y[anchor_idx]
            lower_is_better = {"val/con_loss", "ae/val_loss_normal"}
            higher = mk not in lower_is_better and "drift" not in mk
            best_val = float(np.nanmax(y)) if higher else float(np.nanmin(y))
            best_val_idx = int(np.nanargmax(y)) if higher else int(np.nanargmin(y))
            row[f"{mk}_anchor"] = anchor_val
            row[f"{mk}_best"] = best_val
            row[f"{mk}_best_value"] = x[best_val_idx]
            row[f"{mk}_delta_pct"] = 100 * (best_val - anchor_val) / abs(anchor_val) if anchor_val and not np.isnan(anchor_val) else float("nan")
        summary_rows.append(row)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = output_dir / "ablation_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.4f")
        print(f"  Saved summary: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results from MLflow")
    parser.add_argument("--experiment", default="aug_supcon_ablation",
                        help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default="/eos/user/d/dgenoves/mlflow",
                        help="MLflow tracking URI")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save plots (default: <tracking_uri>/../ablation_plots/<experiment>)")
    parser.add_argument("--mode", choices=["encoder", "ae", "both"], default="both",
                        help="Which ablation section to plot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.tracking_uri).parent / "ablation_plots" / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching runs from experiment: {args.experiment}")
    df = fetch_runs(args.experiment, args.tracking_uri)
    print(f"Found {len(df)} ablation runs")

    if df.empty:
        print("No runs found. Have the jobs completed?")
        return

    # Determine model name and param order from experiment name
    is_aug = "aug" in args.experiment.lower()
    model_name = "Augmented SupCon" if is_aug else "Vanilla SupCon"
    encoder_param_order = AUG_ENCODER_PARAM_ORDER if is_aug else VANILLA_ENCODER_PARAM_ORDER

    # ── Encoder ablation plots ────────────────────────────────────
    if args.mode in ("encoder", "both"):
        print("\n[Encoder ablation plots]")
        enc_df = df[df["mode"] == "encoder"]
        all_metrics = {**ENCODER_METRICS, **AE_METRICS}

        for param in encoder_param_order:
            plot_param_sensitivity(enc_df, param, all_metrics, output_dir, mode="enc")

        plot_sensitivity_summary(
            enc_df, encoder_param_order, all_metrics, output_dir,
            mode="enc", title=f"{model_name} Encoder"
        )
        for mk, ml in all_metrics.items():
            plot_metric_overview(enc_df, encoder_param_order, mk, ml, output_dir, mode="enc")

    # ── AE ablation plots ─────────────────────────────────────────
    if args.mode in ("ae", "both"):
        print("\n[AE ablation plots]")
        ae_df = df[df["mode"] == "ae"]

        for param in AE_PARAM_ORDER:
            plot_param_sensitivity(ae_df, param, AE_METRICS, output_dir, mode="ae")

        plot_sensitivity_summary(
            ae_df, AE_PARAM_ORDER, AE_METRICS, output_dir,
            mode="ae", title=f"{model_name} AE"
        )
        for mk, ml in AE_METRICS.items():
            plot_metric_overview(ae_df, AE_PARAM_ORDER, mk, ml, output_dir, mode="ae")

    print("\n[Exporting tables]")
    export_table(df, output_dir, model_name)

    print(f"\nAll plots and tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
