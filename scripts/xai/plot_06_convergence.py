#!/usr/bin/env python3
"""
Figure for step 06: does the AE's reconstruction error track the same observable
that the Wasserstein ranking picked?

Step 06 produces two independent explanations per component and a convergence
verdict, but only as JSON/CSV. This renders the part that carries the argument:
the Spearman correlation between the AE anomaly score and each high-level
observable, over the flagged signal in each populated component.

Why Spearman and not the per-dimension vote. Step 06 also decides its winner by a
majority vote over the top-N highest-residual embedding dimensions. That vote is
computed from 5 dimensions and can tie — on d128 component 3 it split 2-2 and was
settled by a 0.04 difference in mean |rho|, producing the run's only apparent
divergence. The Spearman correlation is estimated over thousands of flagged
events and needs no tie-break, so it is the statistic to show.

The bar for the observable that step 04 ranked first by Wasserstein distance is
drawn in the highlight colour and marked, so agreement between the geometric and
the mechanistic explanation is visible rather than asserted.

Components are identified as C<k> only: the paper characterises them by physics,
never by a dominant process label.

Usage:
    python scripts/xai/plot_06_convergence.py \\
        --run-dir /eos/.../xai_paper/vcreg_d256_seed3_smnorm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_XAI = Path(__file__).resolve().parent
_ROOT = _XAI.parents[1]
sys.path.insert(0, str(_XAI))
sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.constants import PHYSICS_LABELS, PHYSICS_VARS
from common.style import OI

# The message here is "one bar among many", so only the observable step 5 ranked
# first carries a hue; the rest recede to grey. A second saturated colour would
# compete with the highlight and blunt the point.
C_BASE = OI["neutral"]
C_HIGHLIGHT = OI["signal"]


def load(run_dir: Path):
    mech = json.loads((run_dir / "06_ae_mechanism" / "ae_mechanism.json").read_text())
    conv_path = run_dir / "06_ae_mechanism" / "convergence_summary.csv"
    winners = {}
    if conv_path.exists():
        for row in csv.DictReader(conv_path.open()):
            winners[int(row["component"])] = row["wasserstein_winner"]
    return mech, winners


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <run-dir>/06_ae_mechanism/plots/ae_convergence.pdf")
    p.add_argument("--group", default="flagged", choices=["flagged", "local_qcd"],
                   help="Population the correlation is computed over")
    args = p.parse_args()

    mech, winners = load(args.run_dir)
    results = mech["results"]
    if not results:
        print("No populated components in ae_mechanism.json — nothing to plot.")
        return 1

    out = args.output or (args.run_dir / "06_ae_mechanism" / "plots" / "ae_convergence.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Fix one observable order across panels so components can be compared by eye;
    # order by |rho| in the component holding the most flagged signal.
    lead = max(results, key=lambda r: r["n_flagged"])
    lead_rho = lead["spearman_mse_vs_physics"][args.group]
    order = sorted(PHYSICS_VARS, key=lambda v: abs(lead_rho.get(v, {}).get("rho", 0.0)))

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.9), sharex=True)
    if n == 1:
        axes = [axes]

    all_rho = [
        r["spearman_mse_vs_physics"][args.group].get(v, {}).get("rho", np.nan)
        for r in results for v in order
    ]
    lim = max(0.15, 1.15 * np.nanmax(np.abs(all_rho)))

    for ax, r in zip(axes, results):
        comp = int(r["component"])
        rho_map = r["spearman_mse_vs_physics"][args.group]
        rhos = [rho_map.get(v, {}).get("rho", np.nan) for v in order]
        win = winners.get(comp)
        colors = [C_HIGHLIGHT if v == win else C_BASE for v in order]

        y = np.arange(len(order))
        ax.barh(y, rhos, color=colors, height=0.62)
        ax.axvline(0, color="0.35", lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([PHYSICS_LABELS.get(v, v) for v in order], fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(r"Spearman $\rho$(AE score, observable)", fontsize=9)
        ax.set_title(f"component C{comp}   ($n$ = {r['n_flagged']:,})", fontsize=10)
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # Secondary encoding: the highlight is never carried by colour alone.
        if win in order:
            i = order.index(win)
            val = rhos[i]
            ax.annotate(
                "$\\bigstar$",
                xy=(val, i),
                xytext=(6 if val >= 0 else -6, 0),
                textcoords="offset points",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=11, color=C_HIGHLIGHT,
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_HIGHLIGHT),
        plt.Rectangle((0, 0), 1, 1, color=C_BASE),
    ]
    fig.legend(
        handles,
        [r"top-ranked by $W_1$ (step 5)  $\bigstar$", "other observables"],
        loc="lower center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "AE reconstruction error vs high-level observables, flagged signal",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    for r in results:
        comp = int(r["component"])
        rho_map = r["spearman_mse_vs_physics"][args.group]
        top = max(PHYSICS_VARS, key=lambda v: abs(rho_map.get(v, {}).get("rho", 0.0)))
        print(f"  C{comp}: top Spearman = {top} ({rho_map[top]['rho']:+.3f}), "
              f"W1 winner = {winners.get(comp)} → "
              f"{'agree' if top == winners.get(comp) else 'DIFFER'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
