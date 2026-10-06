#!/usr/bin/env python3
"""Fraction of each SM process above the AE threshold that accepts 10% of QCD.

The paper quotes one background number: the val-calibrated threshold passes 10% of
QCD. That says nothing about the other eleven SM processes, and a signal TPR read
against QCD alone can hide that the AE also flags most of leptonic tt or W -> lv.
This measures it, per encoder seed, with the same AE and the same threshold as the
AD run: the threshold is the one stored in the AE checkpoint, never recomputed.

The embeddings are the full-SM ones of the XAI pipeline
(`xai_embeddings_smnorm/`, from `extract_xai_embeddings.py`), which hold the 20,000
test events of every SM class; the AD run's own embeddings hold QCD and the three
signals only. As a check that both are the same test events through the same
encoder, the QCD and signal rows are compared with the AD run's `result.json`. They
agree to within an event or two out of 20,000 --- events sitting on the threshold,
which the AD run scored in GPU batches and this scores on CPU --- so a seed is refused
only when a rate moves by more than TOL, ten events.

Usage:
    python3 scripts/xai/sm_acceptance.py                  # seeds 0-4, d256
    python3 scripts/xai/sm_acceptance.py --seeds 3
    python3 scripts/xai/sm_acceptance.py --from-json      # rewrite the table only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.ae_score import compute_ae_mse, load_val_thresholds  # noqa: E402
from common.constants import CLASS_NAMES  # noqa: E402

NEW_EXP = Path("/eos/user/d/dgenoves/anomaly_pipeline/new_exp")
FPR = 0.10
SIGNALS = (12, 13, 14)
TOL = 5e-4   # 10 of the 20,000 test events per class

# Rows in the order of the paper's process table, with LaTeX names.
TEX_NAMES = {
    0: r"QCD inclusive", 1: r"$Z \to \nu\nu$ + jets", 2: r"$Z \to q\bar{q}$ ($uds$)",
    3: r"$Z \to b\bar{b}$", 4: r"$Z \to c\bar{c}$", 5: r"$W \to \ell\nu$",
    6: r"$W \to q\bar{q}$", 7: r"$\gamma$", 8: r"QCD $\to b\bar{b}$",
    9: r"$t\bar{t}$ (all-hadronic)", 10: r"$t\bar{t}$ (semi-leptonic)",
    11: r"$t\bar{t}$ (dileptonic)",
}


def mean_std(vals):
    """Sample std (ddof=1), as in run_encoder_seeds_anomaly.aggregate."""
    mu = float(np.mean(vals))
    return mu, (float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"))


def score_seed(new_exp: Path, run: str, seed: int) -> dict:
    ad = new_exp / "ad_results" / run / f"encoder_seed_{seed}" / "mse_normal"
    ckpt = next(p for p in sorted((ad / "checkpoints").glob("ae-epoch*.ckpt")))
    thr = {float(k): float(v) for k, v in load_val_thresholds(ckpt).items()}[FPR]

    emb = new_exp / "xai_embeddings_smnorm" / run / f"encoder_seed_{seed}" / "embeddings"
    d = np.load(emb / "test_embeddings.npz")
    z, y = d["embeddings"], d["labels"]
    from sklearn.metrics import roc_auc_score

    mse = compute_ae_mse(ckpt, z)
    above = mse > thr
    acc = {int(c): float(above[y == c].mean()) for c in np.unique(y)}
    # AUROC of each process against the QCD test events, as the AD run computes it
    # for the signals: the process is the positive class, the AE MSE the score.
    q = mse[y == 0]
    auroc = {int(c): float(roc_auc_score(np.r_[np.zeros(len(q)), np.ones((y == c).sum())],
                                         np.r_[q, mse[y == c]]))
             for c in np.unique(y) if c != 0}

    ref = json.loads((ad / "result.json").read_text())
    ref = ref["per_signal"]
    tag = f"fpr{int(FPR * 100)}"
    expected = {0: ref[f"fpr_measured_{tag}"],
                **{c: ref[f"tpr_{tag}_cls{c}"] for c in SIGNALS}}
    diff = max(abs(acc[c] - v) for c, v in expected.items())
    bad = {c: (acc[c], v) for c, v in expected.items() if abs(acc[c] - v) > TOL}
    bad.update({f"auroc_{c}": (auroc[c], ref[f"auroc_cls{c}"]) for c in SIGNALS
                if abs(auroc[c] - ref[f"auroc_cls{c}"]) > TOL})
    if bad:
        raise RuntimeError(f"seed {seed}: does not reproduce the AD run "
                           f"(label: here, result.json) {bad}")
    print(f"seed {seed}: threshold {thr:.4f}, QCD/signals within {diff:.1e} of result.json")
    return {"threshold": thr, "ckpt": str(ckpt), "max_diff_vs_ad": diff, "acceptance": acc,
            "auroc": auroc}


def write_tex(path: Path, agg: dict, agg_auroc: dict, dmodel: int, n_seeds: int) -> None:
    # SM only, most selective first: the reader's question is which processes the AE
    # lets through, and ordering by acceptance answers it down the column.
    rows = sorted((c for c in agg if c in TEX_NAMES), key=lambda c: agg[c][0])
    pct = int(FPR * 100)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{AUROC against QCD and acceptance of each Standard Model process at the "
             rf"autoencoder threshold that accepts {pct}\% of QCD events, for VCReg with "
             rf"$d_{{\text{{model}}}} = {dmodel}$. The threshold is calibrated on QCD validation "
             r"events for each encoder seed and applied unchanged to 20,000 test events per "
             rf"process; mean $\pm$ std over {n_seeds} seeds.}}",
             r"\label{tab:sm_acceptance}",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"\textbf{SM process} & \textbf{AUROC} & \textbf{Acceptance [\%]} \\",
             r"\midrule"]
    for c in rows:
        mu, sd = agg[c]
        acc = f"${mu:.1f} \\pm {sd:.1f}$"
        au = f"${agg_auroc[c][0]:.3f} \\pm {agg_auroc[c][1]:.3f}$" if c in agg_auroc else "---"
        lines.append(f"{TEX_NAMES[c]} & {au} & {acc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmodel", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--new-exp", type=Path, default=NEW_EXP,
                    help="Root holding ad_results/ and xai_embeddings_smnorm/")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <new_exp>/sm_acceptance")
    ap.add_argument("--from-json", action="store_true",
                    help="Skip the scoring and rewrite the table from the saved JSON")
    ap.add_argument("--tex", type=Path, default=Path(__file__).resolve().parents[2]
                    / "paper" / "sections" / "xai" / "sm_acceptance.tex")
    a = ap.parse_args()

    run = f"vcreg_12class_nosparse_dmodel{a.dmodel}_cern"
    out_dir = a.out_dir or a.new_exp / "sm_acceptance"
    out = out_dir / f"{run}_fpr{int(FPR * 100)}.json"
    if a.from_json:
        saved = json.loads(out.read_text())
        agg = {int(c): tuple(v) for c, v in saved["mean_std_percent"].items()}
        agg_auroc = {int(c): tuple(v) for c, v in saved["auroc_mean_std"].items()}
        write_tex(a.tex, agg, agg_auroc, a.dmodel, len(saved["seeds"]))
        return 0

    per_seed = {s: score_seed(a.new_exp, run, s) for s in a.seeds}
    labels = sorted(next(iter(per_seed.values()))["acceptance"])
    agg = {c: mean_std([100 * per_seed[s]["acceptance"][c] for s in a.seeds])
           for c in labels}
    agg_auroc = {c: mean_std([per_seed[s]["auroc"][c] for s in a.seeds])
                 for c in labels if c != 0}

    print(f"\n{'process':14s} {'AUROC vs QCD':>16s}   acceptance at QCD {int(FPR * 100)}% [%]"
          f"  (n_seeds={len(a.seeds)})")
    for c in labels:
        mu, sd = agg[c]
        au = (f"{agg_auroc[c][0]:.3f} ± {agg_auroc[c][1]:.3f}" if c in agg_auroc else "—")
        print(f"{CLASS_NAMES[c]:14s} {au:>16s}   {mu:6.1f} ± {sd:4.1f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run": run, "fpr": FPR, "seeds": a.seeds,
        "per_seed": {str(s): {**v, "acceptance": {str(c): x for c, x in v["acceptance"].items()},
                              "auroc": {str(c): x for c, x in v["auroc"].items()}}
                     for s, v in per_seed.items()},
        "mean_std_percent": {str(c): list(agg[c]) for c in labels},
        "auroc_mean_std": {str(c): list(v) for c, v in agg_auroc.items()},
    }, indent=2))
    print(f"\nSaved {out}")

    write_tex(a.tex, agg, agg_auroc, a.dmodel, len(a.seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
