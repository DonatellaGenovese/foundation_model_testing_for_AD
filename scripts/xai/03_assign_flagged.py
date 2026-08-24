#!/usr/bin/env python3
"""
XAI paper pipeline — Step 03: AE-flag signal and assign to GMM components (F1).

AE provides the anomaly flag; GMM only interprets (max responsibility).
Writes:

  flagged_assignment.pdf   (F1)
  flagged_only.pdf         (same, without the SM baseline bar)
  flagged_vs_missed.pdf    (per-component fraction: flagged vs missed signal)
  flag_rate_per_component.csv  (step 5: AE flag rate for QCD / non-QCD SM / signal, per region)
  assignments.npz
  assign_meta.json

Usage:
    python scripts/xai/03_assign_flagged.py \\
        --embeddings-dir /eos/.../embeddings \\
        --gmm-path       /eos/.../gmm_K12.pkl \\
        --ae-checkpoint  /eos/.../ae.ckpt \\
        --output-dir     /eos/.../xai_paper/assign \\
        [--signal-label 13] [--ae-threshold 1.303] [--fpr 0.10]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_XAI = Path(__file__).resolve().parent
_ROOT = _XAI.parents[1]
sys.path.insert(0, str(_XAI))
sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.ae_score import compute_ae_mse, flag_anomalies, resolve_ae_threshold
from common.constants import CLASS_NAMES, HH4B_LABEL, SIG_LABELS, SM_INDICES
from common.io_embeddings import filter_low_norm, load_train_val_test, sm_mask
from common.projection import build_sm_pca, check_gmm_dims, project
from common.style import OI
from common.utils import save_json


def plot_flagged_assignment(
    frac_flagged: np.ndarray,
    frac_all_sig: np.ndarray,
    frac_sm: np.ndarray,
    k: int,
    signal_name: str,
    path: Path,
    dominant: list | None = None,
    ylim: float | None = None,
):
    x = np.arange(k)
    width = 0.28
    fig, ax = plt.subplots(figsize=(max(7, k * 0.55), 4.2))
    ax.bar(x - width, frac_sm, width, label="SM (all)", color=OI["sm"])
    ax.bar(x, frac_all_sig, width, label=f"{signal_name} (all)",
           facecolor="none", edgecolor=OI["signal"], hatch="///", linewidth=0.8)
    ax.bar(x + width, frac_flagged, width, label=f"{signal_name} AE-flagged",
           color=OI["signal"], hatch="..", edgecolor="white", linewidth=0.0)
    ax.set_xticks(x)
    if dominant:
        labels = [f"C{i}\n{dominant[i]}" for i in range(k)]
        ax.set_xticklabels(labels, fontsize=7)
    else:
        ax.set_xticklabels([f"C{i}" for i in range(k)])
    ax.set_ylabel("Fraction of events")
    ax.set_xlabel("GMM component")
    ax.set_title(f"Where AE-flagged {signal_name} falls (K={k})")
    ax.legend(fontsize=8)
    # A fixed ylim lets two signals be shown side by side on the same scale. Without it
    # each panel is autoscaled to its own maximum and the shared SM bars, which are the
    # same numbers in both, appear at different heights.
    ax.set_ylim(0, ylim if ylim else max(0.05, 1.05 * max(
        frac_flagged.max(), frac_all_sig.max(), frac_sm.max())))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_signal_only(
    frac_flagged: np.ndarray,
    frac_all_sig: np.ndarray,
    k: int,
    signal_name: str,
    path: Path,
    dominant: list | None = None,
):
    """Same as plot_flagged_assignment but without the SM baseline bar."""
    x = np.arange(k)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, k * 0.5), 4.0))
    ax.bar(x - width / 2, frac_all_sig, width, label=f"{signal_name} (all)",
           facecolor="none", edgecolor=OI["signal"], hatch="///", linewidth=0.8)
    ax.bar(x + width / 2, frac_flagged, width, label=f"{signal_name} AE-flagged",
           color=OI["signal"])
    ax.set_xticks(x)
    if dominant:
        labels = [f"C{i}\n{dominant[i]}" for i in range(k)]
        ax.set_xticklabels(labels, fontsize=7)
    else:
        ax.set_xticklabels([f"C{i}" for i in range(k)])
    ax.set_ylabel("Fraction of events")
    ax.set_xlabel("GMM component")
    ax.set_title(f"{signal_name} distribution across GMM components (K={k})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.05, 1.05 * max(frac_flagged.max(), frac_all_sig.max())))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_flagged_vs_missed(
    frac_flag: np.ndarray,
    frac_missed: np.ndarray,
    k: int,
    signal_name: str,
    path: Path,
    dominant: list | None = None,
):
    """Per-component fraction of AE-flagged vs AE-missed signal. If the two are
    nearly identical (as observed), cluster assignment alone does not explain
    the AE's flag/miss decision — motivates the local physics analysis."""
    x = np.arange(k)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, k * 0.55), 4.2))
    ax.bar(x - width / 2, frac_flag, width, label=f"{signal_name} flagged",
           color=OI["signal"])
    ax.bar(x + width / 2, frac_missed, width, label=f"{signal_name} missed",
           facecolor="none", edgecolor=OI["signal"], hatch="\\\\", linewidth=0.8)
    ax.set_xticks(x)
    if dominant:
        ax.set_xticklabels([f"C{i}\n{dominant[i]}" for i in range(k)], fontsize=7)
    else:
        ax.set_xticklabels([f"C{i}" for i in range(k)])
    ax.set_ylabel("Fraction of events")
    ax.set_xlabel("GMM component")
    ax.set_title(f"{signal_name}: AE-flagged vs AE-missed assignment (K={k})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def flag_rate_per_component(
    assign: np.ndarray,
    anomalous: np.ndarray,
    y: np.ndarray,
    k: int,
    qcd_labels: list,
    signal_label: int,
) -> list:
    """Step 5: per component, the AE flag rate (fraction above threshold) for
    QCD-local, non-QCD-SM-local, and signal-local. Distinguishes regions where
    the flag is signal-specific (low SM flag rate) from regions the AE flags
    wholesale (high non-QCD-SM flag rate). The QCD/non-QCD split reflects that
    the AE is trained on QCD only — an intrinsic property of the setup, not
    post-hoc region labeling by dominant class."""
    qcd_mask = np.isin(y, qcd_labels)
    nonqcd_mask = np.isin(y, SM_INDICES) & ~qcd_mask
    sig_mask = y == signal_label

    def rate(group: np.ndarray, in_comp: np.ndarray):
        n = int((group & in_comp).sum())
        f = int((group & in_comp & anomalous).sum())
        return n, (f / n if n else float("nan"))

    rows = []
    for ki in range(k):
        in_comp = assign == ki
        n_qcd, r_qcd = rate(qcd_mask, in_comp)
        n_nq, r_nq = rate(nonqcd_mask, in_comp)
        n_sig, r_sig = rate(sig_mask, in_comp)
        rows.append(
            {
                "component": ki,
                "n_qcd": n_qcd, "flag_rate_qcd": r_qcd,
                "n_nonqcd_sm": n_nq, "flag_rate_nonqcd_sm": r_nq,
                "n_signal": n_sig, "flag_rate_signal": r_sig,
            }
        )
    return rows


def frac_per_component(assignments: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    counts = np.array([(assignments[mask] == c).sum() for c in range(k)], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def dominant_sm_names(assign_sm: np.ndarray, y_sm: np.ndarray, k: int) -> list:
    """Takes the already-computed assignment rather than re-running gmm.predict.

    It used to predict again from the raw embeddings, which silently bypassed the
    projection: with --pca-dim the mixture lives in the reduced space, so a second
    predict on the full embedding would either raise or, if the dimensions happened
    to agree, label the components from a different partition than the one every
    other number in this script refers to.
    """
    assign = assign_sm
    names = []
    for ki in range(k):
        mask = assign == ki
        if not mask.any():
            names.append("—")
            continue
        counts = [(y_sm[mask] == c).sum() for c in SM_INDICES]
        names.append(CLASS_NAMES[SM_INDICES[int(np.argmax(counts))]])
    return names


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--embeddings-dir", type=Path, default=None,
        help="Split tree to read the test population from. Mutually exclusive with "
             "--matched-npz; one of the two is required.",
    )
    p.add_argument(
        "--matched-npz", type=Path, default=None,
        help="Read the test population from a matched array instead of a split tree. "
             "Needed for the CASE signals: their embedding tree holds QCD plus the "
             "seven CASE processes and NO Standard-Model classes, so the SM fractions "
             "this script reports would be empty. The matched array built by "
             "build_matched_case.py carries the 12 SM classes and the signal together, "
             "which is the population the flag rates are meant to describe.",
    )
    p.add_argument("--gmm-path", type=Path, required=True)
    p.add_argument("--ae-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--signal-label", type=int, default=HH4B_LABEL)
    p.add_argument(
        "--label-components", action="store_true",
        help="Annotate each component on the x axis with its dominant SM process. "
             "OFF by default: the paper's claim is a label-free, model-agnostic "
             "characterisation, so components are identified as C0..CK-1 and "
             "described by physics, never named after a class. The composition is "
             "still written to assign_meta.json (dominant_sm_per_component) for "
             "internal validation — this flag only affects the figures.",
    )
    p.add_argument(
        "--ae-threshold", type=float, default=None,
        help="If omitted, use the val-calibrated threshold saved in --ae-checkpoint at --fpr "
             "(falls back to self-calibrating on this run's QCD if the checkpoint predates it)",
    )
    p.add_argument("--fpr", type=float, default=0.10)
    p.add_argument(
        "--pca-dim", type=float, default=0.0,
        help="Project onto this many principal components before the GMM assignment "
             "only — the AE keeps scoring the full embedding, so the flag set and the "
             "threshold are unchanged and the two runs differ solely in how the "
             "flagged events are partitioned. 0 disables; a value below 1 is read as "
             "a variance fraction. Must match the space --gmm-path was fitted in. "
             "Same convention as 04_profile_and_rank.py.",
    )
    p.add_argument(
        "--pca-embeddings-dir", type=Path, default=None,
        help="Embeddings the PCA is fitted on (SM train only). Defaults to "
             "--embeddings-dir; pass it explicitly when the mixture was fitted on a "
             "different tree, or the projection would not be the mixture's own.",
    )
    p.add_argument("--pca-seed", type=int, default=3)
    p.add_argument(
        "--ylim", type=float, default=None,
        help="Fix the y-axis upper limit of the assignment figure. Use the same value "
             "for two signals meant to be compared side by side.",
    )
    p.add_argument(
        "--qcd-labels", type=int, nargs="+", default=[0],
        help="Labels counted as QCD (the AE's trained-normal). Default [0]=QCD-inclusive; "
             "add 8 to include QCD_bb.",
    )
    p.add_argument("--filter-norm-percentile", type=float, default=0.0)
    p.add_argument("--norm-threshold", type=float, default=None, help="Reuse absolute L2 threshold")
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    if (args.embeddings_dir is None) == (args.matched_npz is None):
        raise SystemExit("pass exactly one of --embeddings-dir / --matched-npz")
    if args.matched_npz is not None:
        d = np.load(args.matched_npz)
        X_te, y_te = d["embeddings"], d["labels"]
        print(f"Test population from matched array: {X_te.shape}")
    else:
        splits = load_train_val_test(args.embeddings_dir)
        X_te, y_te = splits["test"]
    X_te, y_te, thr = filter_low_norm(
        X_te, y_te, args.filter_norm_percentile, threshold=args.norm_threshold
    )

    gmm = joblib.load(args.gmm_path)
    k = int(gmm.n_components)

    # Only the assignment is projected. compute_ae_mse below is still given X_te, the
    # full embedding, so the anomaly score and its threshold do not depend on
    # --pca-dim; what changes is which component a flagged event is attributed to.
    pca = None
    if args.pca_dim and args.pca_dim > 0:
        pca_src = args.pca_embeddings_dir or args.embeddings_dir
        if pca_src is None:
            raise SystemExit("--pca-dim with --matched-npz requires --pca-embeddings-dir")
        pca = build_sm_pca(pca_src, args.pca_dim, seed=args.pca_seed)
    Z_gmm = project(pca, X_te)
    check_gmm_dims(gmm, Z_gmm)
    assign = gmm.predict(Z_gmm)

    print("Computing AE MSE on test embeddings…")
    mse = compute_ae_mse(args.ae_checkpoint, X_te)
    ae_thr, thr_src = resolve_ae_threshold(
        mse, y_te, ae_threshold=args.ae_threshold, ae_ckpt_path=args.ae_checkpoint,
        bg_label=0, fpr=args.fpr,
    )
    anomalous = flag_anomalies(mse, ae_thr)
    print(f"AE threshold={ae_thr:.6f} (source={thr_src})")

    sig = args.signal_label
    sig_name = SIG_LABELS.get(sig, CLASS_NAMES.get(sig, str(sig)))
    mask_sig = y_te == sig
    mask_flag = mask_sig & anomalous
    mask_missed = mask_sig & ~anomalous
    mask_sm = sm_mask(y_te)

    print(
        f"{sig_name}: flagged {mask_flag.sum()}/{mask_sig.sum()} "
        f"(TPR={mask_flag.sum() / max(mask_sig.sum(), 1):.3f})"
    )

    frac_flag = frac_per_component(assign, mask_flag, k)
    frac_missed = frac_per_component(assign, mask_missed, k)
    frac_all = frac_per_component(assign, mask_sig, k)
    frac_sm = frac_per_component(assign, mask_sm, k)

    # Still computed unconditionally: it goes into assign_meta.json for internal
    # validation. It reaches the figures only with --label-components.
    y_sm = y_te[mask_sm]
    dominant = dominant_sm_names(assign[mask_sm], y_sm, k)
    plot_dominant = dominant if args.label_components else None

    plot_flagged_assignment(
        frac_flag,
        frac_all,
        frac_sm,
        k,
        sig_name,
        plots / "flagged_assignment.pdf",
        dominant=plot_dominant,
        ylim=args.ylim,
    )
    plot_signal_only(
        frac_flag,
        frac_all,
        k,
        sig_name,
        plots / "flagged_only.pdf",
        dominant=plot_dominant,
    )
    plot_flagged_vs_missed(
        frac_flag,
        frac_missed,
        k,
        sig_name,
        plots / "flagged_vs_missed.pdf",
        dominant=plot_dominant,
    )

    # ── Step 5: AE flag rate per component (QCD / non-QCD SM / signal) ────────
    flag_rates = flag_rate_per_component(assign, anomalous, y_te, k, args.qcd_labels, sig)
    fr_path = out / "flag_rate_per_component.csv"
    with open(fr_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["component", "n_qcd", "flag_rate_qcd", "n_nonqcd_sm",
                        "flag_rate_nonqcd_sm", "n_signal", "flag_rate_signal"],
        )
        w.writeheader()
        for r in flag_rates:
            w.writerow(
                {
                    **{key: r[key] for key in ("component", "n_qcd", "n_nonqcd_sm", "n_signal")},
                    "flag_rate_qcd": "" if not np.isfinite(r["flag_rate_qcd"]) else f"{r['flag_rate_qcd']:.4f}",
                    "flag_rate_nonqcd_sm": "" if not np.isfinite(r["flag_rate_nonqcd_sm"]) else f"{r['flag_rate_nonqcd_sm']:.4f}",
                    "flag_rate_signal": "" if not np.isfinite(r["flag_rate_signal"]) else f"{r['flag_rate_signal']:.4f}",
                }
            )
    print(f"Saved {fr_path}")

    np.savez_compressed(
        out / "assignments.npz",
        embeddings=X_te,
        labels=y_te,
        assignments=assign.astype(np.int32),
        ae_mse=mse.astype(np.float32),
        ae_flagged=anomalous.astype(bool),
        ae_threshold=np.array([ae_thr], dtype=np.float64),
        signal_label=np.array([sig], dtype=np.int32),
        norm_threshold=np.array([thr], dtype=np.float64),
    )

    populated = [
        {"component": int(i), "fraction": float(frac_flag[i]), "n_flagged": int((assign[mask_flag] == i).sum())}
        for i in range(k)
        if frac_flag[i] > 0
    ]
    populated.sort(key=lambda d: -d["fraction"])

    meta = {
        "k": k,
        "signal_label": sig,
        "signal_name": sig_name,
        "ae_threshold": ae_thr,
        "ae_threshold_source": thr_src,
        "fpr": args.fpr,
        "n_signal": int(mask_sig.sum()),
        "n_flagged": int(mask_flag.sum()),
        "tpr": float(mask_flag.sum() / max(mask_sig.sum(), 1)),
        "frac_flagged_per_component": frac_flag.tolist(),
        "frac_missed_per_component": frac_missed.tolist(),
        "frac_all_signal_per_component": frac_all.tolist(),
        "frac_sm_per_component": frac_sm.tolist(),
        "dominant_sm_per_component": dominant,
        "flag_rate_per_component": flag_rates,
        "qcd_labels": args.qcd_labels,
        "populated_components": populated,
        "gmm_path": str(args.gmm_path),
        "ae_checkpoint": str(args.ae_checkpoint),
        "embeddings_dir": str(args.embeddings_dir),
    }
    save_json(out / "assign_meta.json", meta)
    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
