#!/usr/bin/env python3
"""
XAI paper pipeline — Step 06 (Appendix/exploratory): AE mechanism vs GMM
geometry — do they agree on the same physical driver?

Two independent explanations exist for why AE-flagged signal is anomalous
within a GMM component:
  A) Geometric (step 04): which physics variable separates flagged signal from
     local SM background (Wasserstein ranking, already in profile_meta.json).
  B) Mechanistic (this script): which embedding dimensions does the AE fail to
     reconstruct for the signal, and what physics variable do those dimensions
     encode?

This script computes B and compares it to A, per populated component:
  1. Spearman(AE MSE, physics variable), separately for local SM and flagged
     signal in each component — "local SM" (not strict QCD-only) is used for
     sample-size reasons, matching the rest of the pipeline's local-comparison
     convention.
  2. Mean per-dimension residual (z-ẑ)^2 for local SM vs flagged (vs missed,
     if enough events) — ranks the embedding dimensions where flagged signal
     breaks reconstruction the most, within this component.
  3. For the top-N such dimensions, Spearman(raw z_j, physics variable) over
     the component's local population (SM ∪ flagged ∪ missed) — identifies
     what each high-residual dimension physically represents. Majority vote
     across the N dimensions gives this component's "AE-mechanism winner".

Convergence (A's winner == B's winner) is the paper's closing cross-check:
independent geometric and mechanistic explanations agreeing on the same
physical driver validates both; divergence means the AE is sensitive to
latent structure the 8 chosen variables don't capture.

Writes:
  ae_mechanism.json        — full detail per component
  convergence_summary.csv  — component, wasserstein_winner, ae_winner, agree

Usage:
    python scripts/xai/06_ae_mechanism.py \\
        --matched-npz /eos/.../matched_sm_hh4b.npz \\
        --gmm-path /eos/.../gmm_K12.pkl \\
        --ae-checkpoint /eos/.../ae.ckpt \\
        --profile-meta /eos/.../04_profile/profile_meta.json \\
        --output-dir /eos/.../xai_paper/ae_mechanism
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_XAI = Path(__file__).resolve().parent
_ROOT = _XAI.parents[1]
sys.path.insert(0, str(_XAI))
sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
from scipy.stats import spearmanr

from common.ae_score import compute_ae_residual, flag_anomalies
from common.constants import PHYSICS_VARS, SM_INDICES
from common.physics import load_matched_npz
from common.projection import build_sm_pca, check_gmm_dims, project
from common.utils import load_json, save_json


def spearman_safe(x: np.ndarray, y: np.ndarray, min_n: int = 5) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < min_n or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def analyse_component(
    ki: int,
    Z: np.ndarray,
    phys: dict,
    mse: np.ndarray,
    resid: np.ndarray,
    m_loc: np.ndarray,
    m_flag: np.ndarray,
    m_miss: np.ndarray,
    m_qcd: np.ndarray,
    qcd_poor: bool,
    n_top_dims: int,
    min_events: int,
) -> dict:
    # Reference background for the residual comparison = QCD-local (the AE's
    # trained-normal — the residual is by construction a departure from QCD),
    # falling back to all-SM-local where QCD is too sparse.
    m_ref = m_loc if qcd_poor else m_qcd
    ref_name = "local_sm" if qcd_poor else "local_qcd"
    result = {
        "component": int(ki),
        "n_local_sm": int(m_loc.sum()),
        "n_local_qcd": int(m_qcd.sum()),
        "n_flagged": int(m_flag.sum()),
        "n_missed": int(m_miss.sum()),
        "residual_reference": ref_name,
        "qcd_poor": bool(qcd_poor),
    }

    # ── Step 1: Spearman(AE MSE, physics variable) — reference vs flagged ────
    step1 = {ref_name: {}, "flagged": {}}
    for var in PHYSICS_VARS:
        rho, p = spearman_safe(mse[m_ref], phys[var][m_ref])
        step1[ref_name][var] = {"rho": rho, "p": p}
        rho, p = spearman_safe(mse[m_flag], phys[var][m_flag])
        step1["flagged"][var] = {"rho": rho, "p": p}
    result["spearman_mse_vs_physics"] = step1

    if m_ref.sum() < min_events or m_flag.sum() < min_events:
        result["status"] = "insufficient_events"
        result["ae_mechanism_winner"] = None
        return result

    # ── Step 2: per-dimension mean residual, flagged − reference ────────────
    resid_ref_mean = resid[m_ref].mean(axis=0)
    resid_flag_mean = resid[m_flag].mean(axis=0)
    diff = resid_flag_mean - resid_ref_mean
    top_dims = np.argsort(diff)[::-1][:n_top_dims].tolist()

    top_info = [
        {
            "dim": int(d),
            "residual_reference": float(resid_ref_mean[d]),
            "residual_flagged": float(resid_flag_mean[d]),
            "diff": float(diff[d]),
        }
        for d in top_dims
    ]
    if m_miss.sum() >= min_events:
        resid_miss_mean = resid[m_miss].mean(axis=0)
        for entry, d in zip(top_info, top_dims):
            entry["residual_missed"] = float(resid_miss_mean[d])
    result["top_residual_dims"] = top_info

    # ── Step 3: correlate raw z_j of top dims with physics vars ─────────────
    # Population = everything actually living in this component (SM + both
    # flagged and missed signal), for the most stable estimate of what each
    # dimension represents locally.
    pop = m_loc | m_flag | m_miss
    per_dim_best = []
    for d in top_dims:
        z_d = Z[pop, d]
        corrs = {}
        for var in PHYSICS_VARS:
            rho, p = spearman_safe(z_d, phys[var][pop])
            corrs[var] = {"rho": rho, "p": p}
        finite_vars = [v for v in PHYSICS_VARS if np.isfinite(corrs[v]["rho"])]
        if finite_vars:
            best_var = max(finite_vars, key=lambda v: abs(corrs[v]["rho"]))
            best_rho = corrs[best_var]["rho"]
        else:
            best_var, best_rho = None, float("nan")
        per_dim_best.append(
            {"dim": int(d), "correlations": corrs, "best_variable": best_var, "best_rho": best_rho}
        )
    result["dim_physics_correlations"] = per_dim_best

    # Majority vote across top-N dims, tie-broken by mean |rho|
    votes = [e["best_variable"] for e in per_dim_best if e["best_variable"] is not None]
    if votes:
        counts = Counter(votes)
        max_count = max(counts.values())
        candidates = [v for v, c in counts.items() if c == max_count]
        if len(candidates) == 1:
            ae_winner = candidates[0]
        else:
            avg_rho = {
                v: float(np.mean([abs(e["best_rho"]) for e in per_dim_best if e["best_variable"] == v]))
                for v in candidates
            }
            ae_winner = max(candidates, key=lambda v: avg_rho[v])
    else:
        ae_winner = None
    result["ae_mechanism_winner"] = ae_winner
    result["status"] = "ok"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matched-npz", type=Path, required=True, help="Same matched data used in step 04")
    p.add_argument("--gmm-path", type=Path, required=True)
    p.add_argument("--ae-checkpoint", type=Path, required=True)
    p.add_argument("--profile-meta", type=Path, required=True, help="profile_meta.json from step 04")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-top-dims", type=int, default=5, help="High-residual dims probed per component")
    p.add_argument("--min-events", type=int, default=20, help="Min events per group to attempt an estimate")
    p.add_argument(
        "--pca-dim", type=float, default=0.0,
        help="Project onto this many principal components before the GMM assignment "
             "only. The AE residuals below are computed on the FULL embedding, which "
             "is the point of this step: it asks whether the AE's per-dimension error "
             "lines up with the mixture's geometry. 0 disables; must match the space "
             "--gmm-path was fitted in. Requires --pca-embeddings-dir.",
    )
    p.add_argument(
        "--pca-embeddings-dir", type=Path, default=None,
        help="Embeddings the PCA is fitted on (SM train only). Required with --pca-dim: "
             "this step loads a matched npz, not a split tree, so there is nothing to "
             "fall back to.",
    )
    p.add_argument("--pca-seed", type=int, default=3)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    Z, y, phys = load_matched_npz(args.matched_npz)
    print(f"Loaded matched: {Z.shape}")

    gmm = joblib.load(args.gmm_path)
    # Only the assignment is projected; Z stays full-dimensional for the residuals.
    pca = None
    if args.pca_dim and args.pca_dim > 0:
        if args.pca_embeddings_dir is None:
            raise ValueError("--pca-dim requires --pca-embeddings-dir")
        pca = build_sm_pca(args.pca_embeddings_dir, args.pca_dim, seed=args.pca_seed)
    Z_gmm = project(pca, Z)
    check_gmm_dims(gmm, Z_gmm)
    assign = gmm.predict(Z_gmm)

    meta = load_json(args.profile_meta)
    ae_threshold = meta["ae_threshold"]
    signal_label = meta["signal_label"]
    populated = meta["populated_components"]
    winners = {e["component"]: e["dominant_variable"] for e in meta.get("top_variable_per_component", [])}
    # Reuse step 04's QCD definition and QCD-poor decision so both stages align.
    qcd_labels = meta.get("qcd_labels", [0])
    qcd_poor_map = {int(k): bool(v) for k, v in meta.get("qcd_poor_per_component", {}).items()}
    print(f"AE threshold={ae_threshold:.6f} (from {args.profile_meta})")
    print(f"Populated components: {populated} | QCD labels: {qcd_labels}")

    resid = compute_ae_residual(args.ae_checkpoint, Z)
    mse = resid.mean(axis=1)
    anomalous = flag_anomalies(mse, ae_threshold)

    mask_sig_all = y == signal_label
    mask_flag = mask_sig_all & anomalous
    mask_missed = mask_sig_all & ~anomalous
    mask_sm = np.isin(y, SM_INDICES)
    mask_qcd = np.isin(y, qcd_labels)

    all_results = []
    for ki in populated:
        m_loc = mask_sm & (assign == ki)
        m_flag = mask_flag & (assign == ki)
        m_miss = mask_missed & (assign == ki)
        m_qcd = mask_qcd & (assign == ki)
        qcd_poor = qcd_poor_map.get(int(ki), int(m_qcd.sum()) < args.min_events)
        res = analyse_component(
            ki, Z, phys, mse, resid, m_loc, m_flag, m_miss, m_qcd, qcd_poor,
            args.n_top_dims, args.min_events,
        )
        res["wasserstein_winner"] = winners.get(ki)
        if res["status"] == "ok" and res["wasserstein_winner"] is not None:
            res["agree"] = res["ae_mechanism_winner"] == res["wasserstein_winner"]
        else:
            res["agree"] = None
        all_results.append(res)
        print(
            f"C{ki}: Wasserstein winner={res['wasserstein_winner']}  "
            f"AE-mechanism winner={res['ae_mechanism_winner']}  "
            f"agree={res['agree']}  (status={res['status']})"
        )

    # Record which mixture and which projection produced these components. Without it the
    # component indices in this file cannot be tied back to a specific fit, and indices are
    # the only handle the paper uses to name a region.
    save_json(out / "ae_mechanism.json", {
        "gmm_path": str(args.gmm_path),
        "pca_dim": float(args.pca_dim) if args.pca_dim else 0.0,
        "pca_seed": args.pca_seed,
        "populated_components": populated,
        "results": all_results,
    })

    csv_path = out / "convergence_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "wasserstein_winner", "ae_mechanism_winner", "agree", "status"])
        w.writeheader()
        for r in all_results:
            w.writerow(
                {
                    "component": r["component"],
                    "wasserstein_winner": r["wasserstein_winner"],
                    "ae_mechanism_winner": r["ae_mechanism_winner"],
                    "agree": r["agree"],
                    "status": r["status"],
                }
            )
    print(f"Saved {csv_path}")

    n_ok = [r for r in all_results if r["status"] == "ok"]
    n_agree = sum(1 for r in n_ok if r["agree"])
    if n_ok:
        print(f"\nConvergence: {n_agree}/{len(n_ok)} populated components agree (of {len(all_results)} total).")
    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
