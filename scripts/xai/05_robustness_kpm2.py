#!/usr/bin/env python3
"""
XAI paper pipeline — Step 05: robustness at K±2 (Appendix).

Refits GMMs at K-2, K, K+2 (or uses provided pkls), repeats AE-flagged
assignment + top Wasserstein variable, and writes a short comparison table.

Requires a matched npz (embeddings + physics) so F2/T1 logic is consistent.

The ranking mirrors step 04 exactly — Wasserstein standardised by each
observable's global std, ranked against the QCD-local reference with the
all-SM fallback in QCD-poor regions. This step only means something as a
robustness check if it ranks the same quantity step 04 ranks: an earlier
version used raw W1 against all-SM, which made HT (GeV) win over n_bjets
(counts) by unit magnitude alone, so it reported a "stable" winner that was
not the one step 04 had concluded on.

Usage:
    python scripts/xai/05_robustness_kpm2.py \\
        --embeddings-dir /eos/.../embeddings \\
        --matched-npz /eos/.../matched_data.npz \\
        --ae-checkpoint /eos/.../ae.ckpt \\
        --k 12 \\
        --output-dir /eos/.../xai_paper/robustness
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
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture

from common.ae_score import compute_ae_mse, flag_anomalies, resolve_ae_threshold
from common.constants import HH4B_LABEL, PHYSICS_VARS, SIG_LABELS, SM_INDICES
from common.io_embeddings import filter_classes, load_train_val_test
from common.projection import build_sm_pca, check_gmm_dims, project
from common.physics import load_matched_npz
from common.utils import save_json


# Fitting protocol of select_k_profiles.py, which produced the mixtures the paper
# uses. Keep these three in step: a K fitted here must be comparable with a K loaded
# from the selection, or the K scan measures the fitting effort as well as K.
N_RESTARTS = 12
N_INIT = 5
MAX_ITER = 300


def fit_or_load(
    gmm_dir: Path | None,
    Z_train: np.ndarray,
    k: int,
    seed: int,
    Z_val: np.ndarray | None = None,
) -> GaussianMixture:
    """Load the mixture for this K if the selection already produced one, else fit it
    the way the selection would have.

    EM is a local optimiser, so a single fit lands wherever its initialisation leads.
    An earlier version fitted once with n_init=3 and no model selection, which on the
    K=7 mixture of this analysis converged to a visibly poorer optimum than the one the
    paper uses --- mean log-likelihood -120.83 against -120.69 over 240k Standard Model
    events, and a BIC worse by 6.8e4. The difference was large enough to move a signal
    between components and to look, from the outside, like an instability of the
    partition rather than an under-optimised fit. So this now mirrors the selection:
    N_RESTARTS independent fits of N_INIT initialisations each, keeping the one with the
    lowest BIC on held-out data when it is available and on the training embeddings
    otherwise."""
    if gmm_dir is not None:
        path = Path(gmm_dir) / f"gmm_K{k}.pkl"
        if path.exists():
            print(f"  Loading {path}")
            return joblib.load(path)

    print(f"  Fitting GMM K={k} ({N_RESTARTS} restarts x n_init={N_INIT}, best BIC)…")
    scorer = Z_val if Z_val is not None else Z_train
    candidates = [
        GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=seed + i,
            max_iter=MAX_ITER,
            n_init=N_INIT,
            reg_covar=1e-6,
        ).fit(Z_train)
        for i in range(N_RESTARTS)
    ]
    best = min(candidates, key=lambda g: g.bic(scorer))
    print(f"    kept BIC={best.bic(scorer):,.0f} of "
          f"[{min(g.bic(scorer) for g in candidates):,.0f}, "
          f"{max(g.bic(scorer) for g in candidates):,.0f}]")
    return best


def _w1(a: np.ndarray, b: np.ndarray, scale: float, min_n: int = 5) -> float:
    """Wasserstein-1 divided by the observable's global std — identical to the
    helper in step 04, so the two steps rank on the same quantity."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) >= min_n and len(b) >= min_n and np.isfinite(scale) and scale > 0:
        return float(wasserstein_distance(a, b)) / scale
    return float("nan")


def top_w1_variable(phys, phys_scale, mask_sig, mask_ref) -> tuple[str | None, float]:
    """Top observable by *standardised* W1 against `mask_ref`.

    Both the standardisation and the choice of reference must match step 04: this
    step exists to test whether step 04's conclusion survives a change of K, and
    it can only do that if it ranks the same quantity. Ranking raw W1 instead
    makes the observable with the largest unit magnitude win by construction
    (HT in GeV beats n_bjets in counts regardless of the physics).
    """
    best_var, best_w = None, -1.0
    for var in PHYSICS_VARS:
        w = _w1(phys[var][mask_sig], phys[var][mask_ref], phys_scale[var])
        if np.isfinite(w) and w > best_w:
            best_w, best_var = w, var
    return best_var, best_w


def analyse_k(
    gmm, Z, y, phys, phys_scale, anomalous, signal_label: int, min_frac: float,
    qcd_labels: list[int], min_qcd: int,
) -> dict:
    assign = gmm.predict(Z)
    k = gmm.n_components
    mask_flag = (y == signal_label) & anomalous
    mask_sm = np.isin(y, SM_INDICES)
    mask_qcd = np.isin(y, qcd_labels)
    counts = np.array([(assign[mask_flag] == i).sum() for i in range(k)], dtype=float)
    frac = counts / counts.sum() if counts.sum() else counts
    populated = [int(i) for i in np.argsort(frac)[::-1] if frac[i] >= min_frac]
    if not populated and len(frac):
        populated = [int(np.argmax(frac))]

    per_comp = []
    for ki in populated:
        m_sig = mask_flag & (assign == ki)
        m_qcd = mask_qcd & (assign == ki)
        m_sm = mask_sm & (assign == ki)
        # Primary reference is QCD-local (the AE's trained-normal), falling back
        # to all-SM where QCD is too thin — same rule as step 04.
        qcd_poor = int(m_qcd.sum()) < min_qcd
        rank_ref = "sm" if qcd_poor else "qcd"
        var, w = top_w1_variable(phys, phys_scale, m_sig, m_sm if qcd_poor else m_qcd)
        var_sm, w_sm = top_w1_variable(phys, phys_scale, m_sig, m_sm)
        if qcd_poor:
            print(f"  [K={k} C{ki}] QCD-poor ({int(m_qcd.sum())} < {min_qcd}) — ranking on all-SM reference")
        per_comp.append(
            {
                "component": ki,
                "fraction_flagged": float(frac[ki]),
                "top_variable": var,
                "W1": w,
                "rank_ref": rank_ref,
                "top_variable_sm": var_sm,
                "W1_sm": w_sm,
                "n_qcd": int(m_qcd.sum()),
                "n_sm": int(m_sm.sum()),
            }
        )
    return {
        "k": int(k),
        "n_flagged": int(mask_flag.sum()),
        "populated": per_comp,
        "frac_per_component": frac.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", type=Path, required=True, help="For fitting GMMs on SM train")
    p.add_argument("--matched-npz", type=Path, required=True)
    p.add_argument("--ae-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--k", type=int, required=True, help="Central K (also runs K-2 and K+2 if >=4)")
    p.add_argument("--gmm-dir", type=Path, default=None, help="Optional dir with gmm_K{k}.pkl")
    p.add_argument(
        "--ae-threshold", type=float, default=None,
        help="If omitted, use the val-calibrated threshold saved in --ae-checkpoint at --fpr "
             "(falls back to self-calibrating on this run's QCD if the checkpoint predates it)",
    )
    p.add_argument("--fpr", type=float, default=0.10)
    p.add_argument("--min-frac", type=float, default=0.05)
    p.add_argument("--gmm-seed", type=int, default=42)
    p.add_argument("--max-val", type=int, default=100_000,
                   help="Cap on held-out SM events used to score the restarts by BIC "
                        "(matches select_k_profiles.py).")
    p.add_argument("--signal-label", type=int, default=HH4B_LABEL)
    p.add_argument("--max-train", type=int, default=300_000)
    p.add_argument(
        "--pca-dim", type=float, default=0.0,
        help="Project onto this many principal components for the GMM stage. Unlike "
             "steps 03/04/06 this one REFITS mixtures at K+-2, so the projection is "
             "applied to the training embeddings before the fit as well as to the "
             "matched array before predict — otherwise the K+-2 mixtures would live "
             "in a different space than the central K. 0 disables.",
    )
    p.add_argument(
        "--pca-embeddings-dir", type=Path, default=None,
        help="Embeddings the PCA is fitted on (SM train only). Defaults to "
             "--embeddings-dir.",
    )
    p.add_argument("--pca-seed", type=int, default=3)
    p.add_argument(
        "--qcd-labels", type=int, nargs="+", default=[0],
        help="Labels counted as QCD (the AE's trained-normal, primary Wasserstein reference)",
    )
    p.add_argument(
        "--min-qcd", type=int, default=50,
        help="Below this many local QCD events the ranking falls back to the all-SM reference",
    )
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    ks = sorted({max(2, args.k - 2), args.k, args.k + 2})
    print(f"Robustness K grid: {ks}")

    splits = load_train_val_test(args.embeddings_dir)
    X_tr, _ = filter_classes(*splits["train"], SM_INDICES)
    rng = np.random.default_rng(args.gmm_seed)
    if len(X_tr) > args.max_train:
        X_tr = X_tr[rng.choice(len(X_tr), args.max_train, replace=False)]
    # Held-out SM, used only to pick among the restarts by BIC when a K has to be fitted
    # here. Selecting on the training embeddings would favour the fit that overfits them.
    X_va, _ = filter_classes(*splits["val"], SM_INDICES)
    if len(X_va) > args.max_val:
        X_va = X_va[rng.choice(len(X_va), args.max_val, replace=False)]

    Z, y, phys = load_matched_npz(args.matched_npz)
    # The AE scores the FULL embedding, exactly as in steps 03/04, so the flag set is
    # independent of --pca-dim. Only the mixtures move.
    mse = compute_ae_mse(args.ae_checkpoint, Z)
    ae_thr, thr_src = resolve_ae_threshold(
        mse, y, ae_threshold=args.ae_threshold, ae_ckpt_path=args.ae_checkpoint,
        bg_label=0, fpr=args.fpr,
    )
    anomalous = flag_anomalies(mse, ae_thr)
    print(f"AE threshold={ae_thr:.6f} ({thr_src})")

    # Global per-observable scale, computed exactly as in step 04 (std over all
    # finite matched values) so the two steps produce comparable rankings.
    phys_scale = {}
    for var in PHYSICS_VARS:
        vals = phys[var][np.isfinite(phys[var])]
        phys_scale[var] = float(np.std(vals)) if len(vals) else float("nan")

    pca = None
    if args.pca_dim and args.pca_dim > 0:
        pca = build_sm_pca(
            args.pca_embeddings_dir or args.embeddings_dir,
            args.pca_dim, seed=args.pca_seed,
        )
    X_tr_gmm = project(pca, X_tr)   # fit space
    X_va_gmm = project(pca, X_va)   # same basis, for the BIC among restarts
    Z_gmm = project(pca, Z)         # predict space, same basis

    results = []
    for k in ks:
        print(f"\n=== K={k} ===")
        gmm = fit_or_load(args.gmm_dir, X_tr_gmm, k, args.gmm_seed, Z_val=X_va_gmm)
        check_gmm_dims(gmm, Z_gmm)
        if args.gmm_dir is not None:
            (Path(args.gmm_dir)).mkdir(parents=True, exist_ok=True)
            joblib.dump(gmm, Path(args.gmm_dir) / f"gmm_K{k}.pkl")
        results.append(analyse_k(
            gmm, Z_gmm, y, phys, phys_scale, anomalous, args.signal_label,
            args.min_frac, args.qcd_labels, args.min_qcd,
        ))

    summary = {
        "central_k": args.k,
        "k_grid": ks,
        "ae_threshold": ae_thr,
        "signal_label": args.signal_label,
        "signal_name": SIG_LABELS.get(args.signal_label, str(args.signal_label)),
        "qcd_labels": args.qcd_labels,
        "min_qcd": args.min_qcd,
        "phys_scale": phys_scale,
        "results": results,
    }
    save_json(out / "robustness_kpm2.json", summary)

    csv_path = out / "robustness_kpm2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["k", "component", "fraction_flagged", "top_variable", "W1",
                        "rank_ref", "top_variable_sm", "W1_sm", "n_qcd", "n_sm"],
        )
        w.writeheader()
        for block in results:
            for row in block["populated"]:
                w.writerow(
                    {
                        "k": block["k"],
                        "component": row["component"],
                        "fraction_flagged": f"{row['fraction_flagged']:.4f}",
                        "top_variable": row["top_variable"] or "",
                        "W1": "" if row["W1"] < 0 else f"{row['W1']:.6g}",
                        "rank_ref": row["rank_ref"],
                        "top_variable_sm": row["top_variable_sm"] or "",
                        "W1_sm": "" if row["W1_sm"] < 0 else f"{row['W1_sm']:.6g}",
                        "n_qcd": row["n_qcd"],
                        "n_sm": row["n_sm"],
                    }
                )
    print(f"Saved {csv_path}")
    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
