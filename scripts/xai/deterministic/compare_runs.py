#!/usr/bin/env python3
"""Compare two runs of the interpretability chain with each other and with the published
outputs.

For every output the chain writes, reports whether run A and run B are bit-identical
(the determinism test) and how far each is from the published file (the alignment test):
embeddings, event arrays, mixture, choice of K, assignments, Wasserstein ranking, K +- 2
check, Spearman correlations, Table 6.

Each cell is "identical", the largest absolute difference (continuous arrays), the share
of entries that differ (labels, assignments, flags), or the two sizes when the arrays do
not even have the same shape (a different sample of events). A section a run has not
written yet is listed as not available rather than stopping the comparison. After the
table: event counts per class, the component alignment of each run, and the lines of
Table 6 that differ from the paper.

Usage:
    python scripts/xai/deterministic/compare_runs.py --a /eos/.../run_A --b /eos/.../run_B
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path

import joblib
import numpy as np

PUB_EMB = Path("/eos/user/d/dgenoves/anomaly_pipeline/new_exp/xai_embeddings_smnorm/"
               "vcreg_12class_nosparse_dmodel256_cern/encoder_seed_3/embeddings")
PUB_XP = Path("/eos/user/d/dgenoves/anomaly_pipeline/xai_paper")
REPO = Path(__file__).resolve().parents[3]
KDIR = "k_selection_v3/vcreg_d256_seed3_diag_pca64"

ROWS = []


def diff(x, y):
    """(value, text). value 0 means identical; text is what the table prints."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return float("inf"), (f"{len(x)} vs {len(y)}" if x.ndim and y.ndim and x.shape[1:] == y.shape[1:]
                              else f"{x.shape} vs {y.shape}")
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        d = np.where(np.isnan(x) & np.isnan(y), 0.0, d)
        v = float(np.nanmax(d)) if d.size else 0.0
        return v, ("identical" if v == 0 else f"{v:.1e}")
    frac = float(np.mean(x != y)) if x.size else 0.0
    return frac, ("identical" if frac == 0 else f"{frac:.2%} differ")


def record(name, a, b, pub):
    ROWS.append((name, diff(a, b), diff(a, pub), diff(b, pub)))


def section(title, fn):
    """Run one block of comparisons; a run that stopped early keeps the rows before it."""
    try:
        fn()
    except Exception as e:
        ROWS.append((f"{title}: not available ({type(e).__name__}: ...{str(e)[-70:]})", None, None, None))


def npz_arrays(p):
    d = np.load(p)
    return {k: d[k] for k in d.files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    A, B = args.a, args.b
    runs = (A, B, PUB_XP)
    counts = {}

    def embeddings(split):
        fa, fb, fp = (npz_arrays(d / f"{split}_embeddings.npz") for d in (A / "embeddings", B / "embeddings", PUB_EMB))
        for k in ("embeddings", "labels"):
            record(f"embeddings {split}: {k}", fa[k], fb[k], fp[k])
        counts[split] = [dict(zip(*np.unique(f["labels"], return_counts=True))) for f in (fa, fb, fp)]

    for split in ("train", "val", "test"):
        section(f"embeddings {split}", lambda s=split: embeddings(s))

    def arrays(rel):
        fa, fb, fp = (npz_arrays(d / rel) for d in runs)
        for k in sorted(fp):
            record(f"{rel.split('/')[-1]}: {k}", fa[k], fb[k], fp[k])

    for rel in ("vcreg_d256_seed3_smnorm/04_profile/matched_sm_hh4b.npz",
                "case_HVdilep_Zp1000_piD2_mumu_d256_seed3/matched_sm_HVdilep_Zp1000_piD2_mumu.npz"):
        section(rel.split("/")[-1], lambda r=rel: arrays(r))

    def mixture(K):
        ga, gb, gp = (joblib.load(d / KDIR / f"gmm_K{K}.pkl") for d in runs)
        for attr in ("weights_", "means_", "covariances_"):
            record(f"mixture K={K}: {attr}", getattr(ga, attr), getattr(gb, attr), getattr(gp, attr))

    for K in (5, 7, 9):
        section(f"mixture K={K}", lambda k=K: mixture(k))

    prof = "k_profiles/vcreg_d256_seed3_pca64/k_profiles.json"

    def choice_of_k():
        sel = [json.load(open(d / prof))["selection"] for d in runs]
        record("choice of K (k_finest)", sel[0]["k_finest"], sel[1]["k_finest"], sel[2]["k_finest"])
        shares = []
        for d in runs:
            rows = {int(r["k"]): float(r["min_rel_share"]) for r in csv.DictReader(open(d / prof.replace(".json", ".csv")))}
            shares.append([rows[k] for k in sorted(rows)])
        record("least-populated share per K", *shares)

    section("choice of K", choice_of_k)

    def signal(t):
        fa, fb, fp = (npz_arrays(d / f"k7_{t}_pca64_d256_seed3/03_assign_matched/assignments.npz") for d in runs)
        for k in ("assignments", "ae_flagged", "ae_mse"):
            record(f"{t} 03: {k}", fa[k], fb[k], fp[k])
        w = []
        for d in runs:
            rs = {(r["component"], r["variable"]): r for r in csv.DictReader(open(d / f"rank_k7_sm/{t}/wasserstein_rank.csv"))}
            w.append(np.array([[float(rs[k][c]) if rs[k][c] not in ("", "nan") else np.nan for c in ("W1_sm", "W1_qcd")]
                               for k in sorted(rs)]))
        record(f"{t} 04: Wasserstein W1", *w)
        s = []
        for d in runs:
            j = json.load(open(d / f"k7_{t}_pca64_d256_seed3/06_ae_mechanism/ae_mechanism.json"))
            s.append(np.array([r["spearman_mse_vs_physics"]["flagged"][v]["rho"] for r in j["results"]
                               for v in sorted(r["spearman_mse_vs_physics"]["flagged"])]))
        record(f"{t} 06: Spearman rho", *s)
        k5 = [np.array([float(r["W1_sm"]) for r in csv.DictReader(open(d / f"k7_{t}_pca64_d256_seed3/05_robustness/robustness_kpm2.csv"))])
              for d in runs]
        record(f"{t} 05: K +- 2 W1", *k5)

    for t in ("hh4b", "hvdilep"):
        section(f"{t} 03-06", lambda s=t: signal(s))

    tex = {}

    def table6():
        tex["A"] = (A / "figures/wasserstein_side_by_side.tex").read_text()
        tex["B"] = (B / "figures/wasserstein_side_by_side.tex").read_text()
        tex["paper"] = (REPO / "paper/sections/xai/wasserstein_side_by_side.tex").read_text()
        same = lambda x, y: (0.0, "identical") if x == y else (float("inf"), "differs")
        ROWS.append(("Table 6 .tex", same(tex["A"], tex["B"]), same(tex["A"], tex["paper"]), same(tex["B"], tex["paper"])))

    section("Table 6", table6)

    width = max(len(r[0]) for r in ROWS)
    cols = ("A vs B", "A vs paper", "B vs paper")
    lines = [f"{'output':<{width}}  " + "  ".join(f"{c:>18}" for c in cols)]
    for name, *cells in ROWS:
        lines.append(f"{name:<{width}}  " + "  ".join(f"{(c[1] if c else '-'):>18}" for c in cells))
    compared = [r for r in ROWS if r[1] is not None]
    lines.append(f"\nA and B bit-identical everywhere: {bool(compared) and all(r[1][0] == 0 for r in compared)}"
                 f"  ({len(compared)} outputs compared, {len(ROWS) - len(compared)} not available)")

    for split, (ca, cb, cp) in counts.items():
        lines.append(f"\nevents per class, {split}:  class  A  B  paper")
        for k in sorted(set(ca) | set(cb) | set(cp)):
            lines.append(f"  {int(k):>2}  {ca.get(k, 0):>8}  {cb.get(k, 0):>8}  {cp.get(k, 0):>8}")
        lines.append(f"  total  {sum(ca.values()):>8}  {sum(cb.values()):>8}  {sum(cp.values()):>8}")

    for tag, d in (("A", A), ("B", B)):
        al = d / KDIR / "gmm_K7.pkl.alignment.json"
        if al.exists():
            lines.append(f"\ncomponent alignment {tag} (K=7, refit -> paper numbering): {al.read_text().strip()}")

    if tex:
        for tag in ("A", "B"):
            if tex[tag] != tex["paper"]:
                d = list(difflib.unified_diff(tex["paper"].splitlines(), tex[tag].splitlines(),
                                              "paper", f"run {tag}", n=0, lineterm=""))
                lines.append(f"\nTable 6, run {tag} vs paper ({len(d)} diff lines, first 80):")
                lines += d[:80]

    for tag, d in (("A", A), ("B", B)):
        hosts = {p.name: p.read_text().strip() for p in sorted((d / "hosts").glob("*"))}
        for m in sorted(d.glob("**/meta.json")):
            try:
                hosts[str(m.relative_to(d))] = json.load(open(m)).get("host")
            except Exception:
                pass
        lines.append(f"hosts {tag}: {hosts}")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
