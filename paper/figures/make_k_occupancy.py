#!/usr/bin/env python3
"""
Figure: occupancy of the least-populated GMM component vs K, in both spaces.

Plots the quantity the K selection is actually made on, so the figure and the
criterion cannot drift apart: `min_rel_share` is the size of the smallest component
divided by N_SM/K, i.e. the share it would hold under a uniform partition. The
criterion keeps a K only when this stays above `min_share` (0.2) for every component.

Dividing by the uniform share matters. The raw count of the smallest component falls
with K for the trivial reason that the same events are split more ways; the ratio
removes that and leaves the imbalance, which is what makes a component unusable.

Inputs are the CSVs written by scripts/xai/select_k_profiles.py:
    <XP>/k_profiles/vcreg_d256_seed3_raw/k_profiles.csv     (unprojected, 256 dims)
    <XP>/k_profiles/vcreg_d256_seed3_pca64/k_profiles.csv   (PCA, 64 dims)

Usage:
    python paper/figures/make_k_occupancy.py [--outdir paper/figures]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/xai"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.style import OI

XP = Path("/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/k_profiles")
FLOOR = 0.2          # must match `min_share` in select_k_profiles.py
K_SELECTED = 7


def load(tag: str):
    rows = list(csv.DictReader(open(XP / f"vcreg_d256_seed3_{tag}" / "k_profiles.csv")))
    return [int(r["k"]) for r in rows], [float(r["min_rel_share"]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()

    kr, mr = load("raw")
    kp, mp = load("pca64")
    i7 = kp.index(K_SELECTED)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axhline(FLOOR, ls="--", lw=1.1, color="0.35", zorder=1)
    ax.text(12.2, FLOOR + 0.007, "occupancy floor", ha="right", va="bottom",
            fontsize=8, color="0.35")
    ax.plot(kr, mr, "o-", ms=4, lw=1.4, color=OI.get("sm", "#0072B2"),
            label=r"unprojected ($d=256$)")
    ax.plot(kp, mp, "s-", ms=4, lw=1.4, color=OI.get("signal", "#D55E00"),
            label=r"PCA ($d=64$)")
    ax.plot([K_SELECTED], [mp[i7]], marker="*", ms=15, linestyle="none",
            color=OI.get("signal", "#D55E00"), mec="black", mew=0.6, zorder=5)
    ax.annotate(rf"$K={K_SELECTED}$", xy=(K_SELECTED, mp[i7]),
                xytext=(K_SELECTED - 0.45, mp[i7] + 0.035), fontsize=9)

    ax.set_xlabel(r"number of components $K$")
    # Plotted as a fraction of the uniform share N_SM/K, not as a raw count: the count
    # of the smallest component falls with K for the trivial reason that the same
    # events are split more ways, which would hide the imbalance the criterion is
    # actually about. A value of 1 would mean a perfectly balanced partition.
    ax.set_ylabel("least-populated component\n(fraction of uniform share)")
    ax.set_xticks(range(3, 13))
    ax.set_ylim(0, 0.42)
    ax.set_xlim(2.7, 12.3)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()

    a.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = a.outdir / f"k_occupancy.{ext}"
        fig.savefig(p, dpi=200)
        print(f"wrote {p}")
    print(f"unprojected: max = {max(mr):.3f} at K={kr[mr.index(max(mr))]} "
          f"-> never clears the {FLOOR} floor")
    print(f"PCA 64     : K=7 -> {mp[i7]:.3f} ; K=8 -> {mp[kp.index(8)]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
