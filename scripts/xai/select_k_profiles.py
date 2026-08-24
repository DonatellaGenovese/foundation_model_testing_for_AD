#!/usr/bin/env python3
"""
XAI paper — choose K from the physics profiles alone: no BIC, no ARI.

WHY THIS EXISTS. BIC decreases monotonically over every K in every space we have
measured, so it never selects. ARI measures reproducibility, not quality, and
reproducibility can be bought by destroying information — projecting to 16
dimensions gives ARI 0.997 at K=11, which is a warning sign, not a result. Both
answer questions the paper does not ask. What the paper needs is that the regions
it interprets are (1) genuinely different from one another and (2) each backed by
enough events to characterise. That is what this script tests, directly.

THE TWO CRITERIA.

(1) NO DUPLICATE REGIONS. Each occupied component gets a profile: the median of
    each of the eight observables, divided by that observable's global standard
    deviation so the eight numbers are commensurable (the same standardisation the
    Wasserstein ranking uses). The distance between two regions is the Euclidean
    distance between their profiles.

    The threshold is calibrated, not chosen. A fixed cut such as 0.25 cannot
    distinguish "these two regions are alike" from "I do not have enough events to
    tell them apart" — and those need opposite conclusions. So for each pair we
    pool their events, reshuffle into two groups of the original sizes, and
    recompute the distance, B times. That null is what the distance would look like
    if the two regions were one population sampled twice, at exactly their sample
    sizes. Then

        sep(i,j) = observed distance / median(null distance)

    and a pair counts as duplicated when sep < --sep-threshold. Being a ratio, sep
    is dimensionless and comparable across K and across spaces, which is what makes
    the projected and unprojected runs directly comparable. It also self-corrects
    for statistics: small regions get a wide null and must be further apart to
    count as distinct.

(2) NO UNDER-POPULATED REGION. Two floors, both on the SM sample the GMM was fit
    on — not on QCD. Which regions happen to contain QCD is a fact discovered after
    the partition exists; letting it choose K would mean the answer changed if the
    test split were resampled, i.e. the criterion would be measuring the split size
    rather than the representation. Thin QCD in a region is a problem for the
    Wasserstein reference (fix it there, e.g. by taking the reference from train),
    not for K.

        n_k >= max(--min-count, --min-share * n_SM / K)

    The relative floor is a fraction of the uniform share 1/K, so it does not
    tighten as K grows the way a fixed percentage does. The absolute floor is there
    because a profile is eight medians and a median needs events to be stable.

READING THE SCAN. Both criteria only ever rule K out from above: small K trivially
passes. The informative edge is therefore the upper one — the finest partition
whose regions are all still distinct and populated. Going below it discards
resolution the data demonstrably supports.

For the lower edge the script also reports R(K), which is informational and not a
gate. Within each region it fits a two-component mixture, profiles the two halves
and computes their sep exactly as above, then

        R(K) = max_k sep(half1, half2) / min_(i,j) sep(i, j)

R > 1 says some region hides an internal split that is *more* pronounced than the
weakest distinction between two regions already being reported separately — i.e. K
is too coarse, demonstrably rather than by preference. The split is chosen by a
mixture in embedding space, deliberately without looking at the observables, so
that a large R cannot be an artefact of optimising the very quantity being tested.

Usage (reuses the GMMs already fitted by select_k_interpretable.py where present,
fits only the missing K):

    python scripts/xai/select_k_profiles.py \\
        --embeddings-dir /eos/.../xai_embeddings_smnorm/.../embeddings \\
        --matched-npz    /eos/.../04_profile/matched_sm_hh4b.npz \\
        --gmm-dir        /eos/.../k_selection_v3/vcreg_d256_seed3_diag \\
        --output-dir     /eos/.../k_profiles/d256_raw \\
        --k-values 3 4 5 6 7 8 9 10 11 12
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import combinations
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
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from common.constants import PHYSICS_VARS, SM_INDICES
from common.io_embeddings import filter_classes, load_train_val_test
from common.physics import load_matched_npz
from common.style import OI
from common.utils import save_json


def fit_gmm(Z, k, seed, n_init, max_iter, cov_type="diag"):
    g = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=seed,
                        max_iter=max_iter, n_init=n_init, reg_covar=1e-6)
    g.fit(Z)
    return g


def bic_on_val(gmm, Z_val):
    n = len(Z_val)
    return float(-2 * gmm.score(Z_val) * n + gmm._n_parameters() * np.log(n))


def profile_of(phys_cols: np.ndarray, scales: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Standardised median profile over PHYSICS_VARS for the events in `idx`.

    Undefined entries (Mjj and deta_jj need two jets, MT needs a lepton) are
    dropped per observable, matching step 04. A variable with nothing defined in
    this subset yields NaN and is skipped by the distance.
    """
    out = np.full(phys_cols.shape[1], np.nan)
    sub = phys_cols[idx]
    for j in range(phys_cols.shape[1]):
        v = sub[:, j]
        v = v[np.isfinite(v)]
        if len(v) and scales[j] > 0:
            out[j] = np.median(v) / scales[j]
    return out


def profile_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean distance over the observables defined in both profiles."""
    m = np.isfinite(p) & np.isfinite(q)
    if not m.any():
        return float("nan")
    # Scaled to the full observable count so pairs that lose a variable to
    # undefinedness stay comparable with pairs that keep all eight.
    return float(np.linalg.norm(p[m] - q[m]) * np.sqrt(len(p) / m.sum()))


def separation(phys_cols, scales, idx_a, idx_b, n_perm, rng) -> tuple:
    """(sep, observed distance, null median) for one pair of event groups.

    sep is the observed profile distance over the median distance obtained by
    pooling the two groups and reshuffling them into the same two sizes. A value
    near 1 means the two groups are no further apart than one population sampled
    twice at these sizes.
    """
    d_obs = profile_distance(profile_of(phys_cols, scales, idx_a),
                             profile_of(phys_cols, scales, idx_b))
    pooled = np.concatenate([idx_a, idx_b])
    n_a = len(idx_a)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(pooled)
        null[b] = profile_distance(profile_of(phys_cols, scales, perm[:n_a]),
                                   profile_of(phys_cols, scales, perm[n_a:]))
    d_null = float(np.nanmedian(null))
    sep = float(d_obs / d_null) if d_null > 0 and np.isfinite(d_obs) else float("nan")
    return sep, float(d_obs), d_null


def effective_k(occ: list, seps: dict, threshold: float) -> tuple:
    """Number of physically distinct regions the partition actually delivers.

    Builds the graph on occupied components whose edges join pairs that are NOT
    separated (sep < threshold, i.e. duplicates) and counts its connected
    components. With no duplicate pair this returns len(occ); each duplicate link
    merges regions the paper would otherwise report as different physics.

    This is the readable form of the gate. K_eff tracks K while the structure is
    really there and saturates once it is not, so the K at which it leaves the
    diagonal is the resolution limit — read off a shape rather than a cut. It also
    avoids a bias that afflicts min-over-pairs: the minimum runs over K(K-1)/2
    pairs, 3 at K=3 against 66 at K=12, so it drifts down with K for purely
    combinatorial reasons even when every region is equally distinct. A count of
    groups has no such drift.

    sep is normalised per pair and so is not a metric; this is graph connectivity,
    which needs no triangle inequality.
    """
    parent = {c: c for c in occ}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), s in seps.items():
        if np.isfinite(s) and s < threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for c in occ:
        groups.setdefault(find(c), []).append(c)
    merged = [g for g in groups.values() if len(g) > 1]
    return len(groups), merged


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--matched-npz", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gmm-dir", type=Path, default=None,
                   help="Reuse gmm_K<k>.pkl from here when present (same protocol)")
    p.add_argument("--k-values", type=int, nargs="+",
                   default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    p.add_argument("--pca-dim", type=float, default=0.0,
                   help=">=1 an explicit component count, <1 a variance fraction, "
                        "0 disables. Must match the space the reused GMMs were fitted in.")
    p.add_argument("--cov-type", default="diag", choices=["diag", "full"])
    p.add_argument("--min-share", type=float, default=0.2,
                   help="Occupancy floor as a fraction of the uniform share 1/K")
    # Inert over the range the paper scans, and deliberately left in place: the floor
    # applied is max(min_count, min_share * N_SM / K), and with N_SM = 240,000 the
    # relative term stays above 200 until K > 240. The published selection is therefore
    # decided by --min-share alone, which is what the appendix describes. Kept because
    # it is what the reported scan ran with, and because it guards larger K.
    p.add_argument("--min-count", type=int, default=200,
                   help="Absolute occupancy floor: events needed for a stable median")
    p.add_argument("--sep-threshold", type=float, default=2.0,
                   help="Pairs below this separation count as duplicated")
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--within", action="store_true", default=True,
                   help="Also report R(K), the informational lower-edge diagnostic")
    p.add_argument("--no-within", dest="within", action="store_false")
    p.add_argument("--n-init", type=int, default=5)
    p.add_argument("--n-restarts", type=int, default=12,
                   help="Independent fits; the best BIC on val is kept (matches the "
                        "protocol that produced the reused pickles)")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-val", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    splits = load_train_val_test(args.embeddings_dir)
    X_tr, _ = filter_classes(*splits["train"], SM_INDICES)
    X_va, _ = filter_classes(*splits["val"], SM_INDICES)
    if len(X_va) > args.max_val:
        X_va = X_va[rng.choice(len(X_va), args.max_val, replace=False)]

    Z, y, phys = load_matched_npz(args.matched_npz)

    if args.pca_dim > 0:
        n_comp = int(args.pca_dim) if args.pca_dim >= 1 else args.pca_dim
        # Same construction as select_k_interpretable.py so a reused GMM sees the
        # identical basis: fitted on SM train only, no whitening, same seed.
        pca = PCA(n_components=n_comp, svd_solver="full", whiten=False,
                  random_state=args.seed)
        pca.fit(X_tr)
        d0 = X_tr.shape[1]
        X_tr, X_va, Z = pca.transform(X_tr), pca.transform(X_va), pca.transform(Z)
        print(f"PCA: {d0} -> {X_tr.shape[1]} dims, "
              f"{pca.explained_variance_ratio_.sum():.4f} of the variance retained")

    print(f"SM train {len(X_tr):,} | SM val {len(X_va):,} | dim {X_tr.shape[1]}")

    # Physics as a dense (N, 8) block, plus the global scales used to standardise.
    phys_cols = np.column_stack([phys[v] for v in PHYSICS_VARS])
    scales = np.array([np.std(phys[v][np.isfinite(phys[v])]) for v in PHYSICS_VARS])
    mask_sm = np.isin(y, SM_INDICES)
    Z_sm, phys_sm = Z[mask_sm], phys_cols[mask_sm]
    n_sm = len(Z_sm)
    print(f"matched SM events used for profiling: {n_sm:,}\n")

    rows = []
    for k in args.k_values:
        t0 = time.time()

        pkl = (args.gmm_dir / f"gmm_K{k}.pkl") if args.gmm_dir else None
        if pkl is not None and pkl.exists():
            gmm = joblib.load(pkl)
            src = "reused"
        else:
            cands = [fit_gmm(X_tr, k, args.seed + i, args.n_init, args.max_iter,
                             args.cov_type) for i in range(args.n_restarts)]
            gmm = min(cands, key=lambda g: bic_on_val(g, X_va))
            src = "fitted"
            joblib.dump(gmm, out / f"gmm_K{k}.pkl")

        assign = gmm.predict(Z_sm)
        counts = np.array([(assign == ki).sum() for ki in range(k)])
        floor = max(args.min_count, args.min_share * n_sm / k)
        occ = [ki for ki in range(k) if counts[ki] >= floor]
        # Smallest region in units of the uniform share: 1.0 would be perfectly
        # balanced, so this reads directly as "how lopsided is the partition".
        min_rel_share = float(counts.min() / (n_sm / k))

        # --- criterion 1: pairwise separation, permutation-calibrated ---
        seps, dup = {}, []
        for a, b in combinations(occ, 2):
            s, d_obs, d_null = separation(phys_sm, scales, np.where(assign == a)[0],
                                          np.where(assign == b)[0], args.n_perm, rng)
            seps[(a, b)] = s
            if np.isfinite(s) and s < args.sep_threshold:
                dup.append((a, b, s))
        finite = [s for s in seps.values() if np.isfinite(s)]
        min_sep = float(min(finite)) if finite else float("nan")
        worst = min(seps, key=lambda kk: seps[kk] if np.isfinite(seps[kk]) else np.inf) \
            if finite else None

        # --- informational: internal structure of each region ---
        within_max, within_arg = float("nan"), None
        if args.within and len(occ) >= 2:
            vals = []
            for ki in occ:
                idx = np.where(assign == ki)[0]
                if len(idx) < max(4 * args.min_count, 400):
                    continue
                sub = GaussianMixture(n_components=2, covariance_type=args.cov_type,
                                      random_state=args.seed, max_iter=args.max_iter,
                                      n_init=1, reg_covar=1e-6).fit(Z_sm[idx])
                lab = sub.predict(Z_sm[idx])
                i0, i1 = idx[lab == 0], idx[lab == 1]
                if min(len(i0), len(i1)) < args.min_count:
                    continue
                s, _, _ = separation(phys_sm, scales, i0, i1, args.n_perm, rng)
                if np.isfinite(s):
                    vals.append((s, ki))
            if vals:
                within_max, within_arg = max(vals)[0], max(vals)[1]
        R = float(within_max / min_sep) if np.isfinite(within_max) and min_sep > 0 \
            else float("nan")

        k_eff, merged = effective_k(occ, seps, args.sep_threshold)

        # Full pairwise matrix, so K_eff can be recomputed at another threshold
        # without repeating the permutations (they are the whole cost here).
        np.save(out / f"sep_matrix_K{k}.npy",
                np.array([[a, b, seps[(a, b)]] for a, b in seps], dtype=float))

        ok_occ = len(occ) == k
        ok_dup = len(dup) == 0
        rows.append({
            "k": k, "source": src, "n_occupied": len(occ), "k_eff": k_eff,
            "min_count": int(counts.min()), "min_rel_share": min_rel_share,
            "min_sep": min_sep,
            "worst_pair": f"{worst[0]}-{worst[1]}" if worst else "",
            "n_duplicate_pairs": len(dup),
            "merged_groups": ";".join("+".join(map(str, g)) for g in merged),
            "within_sep_max": within_max, "within_arg": within_arg,
            "R": R,
            "pass_occupancy": ok_occ, "pass_distinct": ok_dup,
            "PASS": ok_occ and ok_dup, "seconds": time.time() - t0,
        })
        verdict = "PASS" if (ok_occ and ok_dup) else (
            "fail:occupancy" if not ok_occ else "fail:duplicates")
        print(f"K={k:3d} [{src:6s}]  occupied={len(occ)}/{k}  K_eff={k_eff:3d}  "
              f"n_min={counts.min():6d} (rel {min_rel_share:.2f})  "
              f"min_sep={min_sep:6.2f} dup_pairs={len(dup)}  R={R:5.2f}   {verdict}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    feasible = [r["k"] for r in rows if r["PASS"]]
    small_ok = [r["k"] for r in rows if r["PASS"] and np.isfinite(r["R"]) and r["R"] < 1.0]
    k_max = max(feasible) if feasible else None
    k_small = min(small_ok) if small_ok else None

    print()
    print("K_eff against K — the diagonal is where every region asked for is delivered:")
    for r in rows:
        flag = "  <- leaves the diagonal" if r["k_eff"] < r["k"] else ""
        print(f"   K={r['k']:3d}   K_eff={r['k_eff']:3d}   occupied={r['n_occupied']}/{r['k']}"
              f"{flag}{('   merged: ' + r['merged_groups']) if r['merged_groups'] else ''}")
    print()
    if feasible:
        print(f"Feasible K (both criteria): {feasible}")
        print(f"  finest supported resolution  -> K = {k_max}")
        if k_max == max(args.k_values):
            print("  WARNING: at the grid edge — the limit was not found, extend --k-values")
        print(f"  smallest K with no unresolved internal structure (R<1) -> "
              f"{k_small if k_small else 'none in grid'}")
    else:
        print("No K satisfies both criteria — loosen the floors or widen the grid.")

    save_json(out / "k_profiles.json", {
        "settings": {"min_share": args.min_share, "min_count": args.min_count,
                     "sep_threshold": args.sep_threshold, "n_perm": args.n_perm,
                     "pca_dim": args.pca_dim, "cov_type": args.cov_type,
                     "n_sm_profiled": n_sm, "dims": int(X_tr.shape[1]),
                     "seed": args.seed},
        "results": rows,
        "selection": {"feasible": feasible, "k_finest": k_max,
                      "k_smallest_resolved": k_small},
    })
    with open(out / "k_profiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    # K_eff first: it is the primary read. Where it tracks the grey diagonal every
    # region asked for is delivered; where it flattens, extra components are only
    # splitting regions that were already there.
    ax[0].plot(ks, ks, ls="--", lw=1, color=OI["neutral"], label="ideal ($K_{eff}=K$)")
    ax[0].plot(ks, [r["k_eff"] for r in rows], marker="o", color=OI["signal"],
               label="measured")
    ax[0].set_ylabel("$K_{eff}$ — distinct regions delivered")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].plot(ks, [r["min_rel_share"] for r in rows], marker="o", color=OI["sm"])
    ax[1].axhline(args.min_share, ls="--", color="0.35", lw=1)
    ax[1].set_ylabel("smallest region / uniform share")
    ax[2].plot(ks, [r["R"] for r in rows], marker="o", color=OI["latent"])
    ax[2].axhline(1.0, ls="--", color="0.35", lw=1)
    ax[2].set_ylabel("R = within / between")
    for a in ax:
        a.set_xlabel("K")
        a.grid(color="0.92", lw=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "k_profiles.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}/k_profiles.{{json,csv,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
