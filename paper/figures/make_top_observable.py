#!/usr/bin/env python3
"""Figure for step 4 of Sec. 3.3: the observable the ranking put first, for each signal.

Step 04 already writes an eight-panel grid per component, but a grid of sixteen panels
across two signals is a poor use of a main-text float: it shows every observable at the
same weight, while the claim rests on exactly one of them per signal. This draws that
one --- n_b for HH->4b in C5, n_leptons for HV Z'->mumu in C2 --- as flagged signal
against the local Standard Model, the same two populations the Wasserstein distance in
the table is computed between. The full grids stay in the appendix.

Both panels are drawn as densities so that populations of very different size can be
compared. The Wasserstein distance is deliberately NOT annotated on the panels: it is
printed in the table this figure sits beside, and an earlier version that repeated it
here kept the value hardcoded, so when the lepton count was recomputed the figure went
on claiming 4.51 against the table's 4.76. One number, one place.

Usage:
    python paper/figures/make_top_observable.py
    python paper/figures/make_top_observable.py --xp <xai output root> --output <pdf>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "xai"))
sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.constants import PHYSICS_BINS, PHYSICS_LABELS, SM_INDICES
from common.style import OI

DEFAULT_XP = Path("/eos/user/d/dgenoves/anomaly_pipeline/xai_paper")
DEFAULT_OUT = Path(__file__).resolve().parent / "xai" / "top_observable_K7.pdf"


def panels(xp: Path) -> list[dict]:
    return [
        dict(tag="hh4b", label=13, comp=5, var="n_bjets",
             name=r"$HH \to 4b$",
             npz=xp / "vcreg_d256_seed3_smnorm" / "04_profile" / "matched_sm_hh4b.npz"),
        dict(tag="hvdilep", label=20, comp=2, var="n_leptons",
             name=r"$Z^{\prime} \to n(\mu\mu)$",
             npz=xp / "case_HVdilep_Zp1000_piD2_mumu_d256_seed3"
                    / "matched_sm_HVdilep_Zp1000_piD2_mumu.npz"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xp", type=Path, default=DEFAULT_XP,
                    help="Root of the XAI outputs (default: the paper's runs)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    OUT = args.output

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    for ax, cfg in zip(axes, panels(args.xp)):
        run = args.xp / f"k7_{cfg['tag']}_pca64_d256_seed3" / "03_assign_matched" / "assignments.npz"
        d = np.load(run)
        y, a, flagged = d["labels"], d["assignments"], d["ae_flagged"].astype(bool)
        vals = np.load(cfg["npz"])[f"phys_{cfg['var']}"]

        here = a == cfg["comp"]
        finite = np.isfinite(vals)
        m_sm = np.isin(y, SM_INDICES) & here & finite
        m_sig = (y == cfg["label"]) & here & flagged & finite

        bins = PHYSICS_BINS[cfg["var"]]
        ax.hist(vals[m_sm], bins=bins, density=True, histtype="step", lw=1.8,
                color=OI["sm"], label="local SM")
        ax.hist(vals[m_sig], bins=bins, density=True, histtype="step", lw=1.8, ls="--",
                color=OI["signal"],
                label=f"{cfg['name']} flagged")

        ax.set_xlabel(PHYSICS_LABELS[cfg["var"]], fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"component C{cfg['comp']}", fontsize=11)
        ax.legend(fontsize=8.5, frameon=False)
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
