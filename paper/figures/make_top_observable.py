#!/usr/bin/env python3
"""Figure for step 4 of Sec. 3.3: the observable the ranking put first, for each signal.

Step 04 already writes an eight-panel grid per component, but a grid of sixteen panels
across two signals is a poor use of a main-text float: it shows every observable at the
same weight, while the claim rests on exactly one of them per signal. This draws that
one --- n_b for HH->4b in C5, n_leptons for HV Z'->mumu in C2 --- as flagged signal
against the local Standard Model, the same two populations the Wasserstein distance in
the table is computed between. The full grids stay in the appendix.

Both panels are drawn as densities so that populations of very different size can be
compared, and each is annotated with its standardised W1 so the figure and the table
carry the same number.

Usage:
    python paper/figures/make_top_observable.py
"""

from __future__ import annotations

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

XP = Path("/eos/user/d/dgenoves/anomaly_pipeline/xai_paper")
OUT = Path(__file__).resolve().parent / "xai" / "top_observable_K7.pdf"

PANELS = [
    dict(tag="hh4b", label=13, comp=5, var="n_bjets", w1=1.28,
         name=r"$HH \to 4b$",
         npz=XP / "vcreg_d256_seed3_smnorm" / "04_profile" / "matched_sm_hh4b.npz"),
    dict(tag="hvdilep", label=20, comp=2, var="n_leptons", w1=4.51,
         name=r"HV $Z' \to \mu\mu$",
         npz=XP / "case_HVdilep_Zp1000_piD2_mumu_d256_seed3"
                / "matched_sm_HVdilep_Zp1000_piD2_mumu.npz"),
]


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    for ax, cfg in zip(axes, PANELS):
        run = XP / f"k7_{cfg['tag']}_pca64_d256_seed3" / "03_assign_matched" / "assignments.npz"
        d = np.load(run)
        y, a, flagged = d["labels"], d["assignments"], d["ae_flagged"].astype(bool)
        vals = np.load(cfg["npz"])[f"phys_{cfg['var']}"]

        here = a == cfg["comp"]
        finite = np.isfinite(vals)
        m_sm = np.isin(y, SM_INDICES) & here & finite
        m_sig = (y == cfg["label"]) & here & flagged & finite

        bins = PHYSICS_BINS[cfg["var"]]
        ax.hist(vals[m_sm], bins=bins, density=True, histtype="step", lw=1.8,
                color=OI["sm"], label=f"local SM  ($n=${m_sm.sum():,})")
        ax.hist(vals[m_sig], bins=bins, density=True, histtype="step", lw=1.8, ls="--",
                color=OI["signal"],
                label=f"{cfg['name']} flagged  ($n=${m_sig.sum():,})")

        ax.set_xlabel(PHYSICS_LABELS[cfg["var"]], fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(rf"C{cfg['comp']}  ---  $W_1^{{\mathrm{{SM}}}} = {cfg['w1']:.2f}$",
                     fontsize=11)
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
