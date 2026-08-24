#!/usr/bin/env python3
"""
Figure for step 4 of Sec. 3.3: how specific is the anomaly flag, region by region?

This is the one headline contribution that step 03 leaves as a CSV. It reports,
per GMM component, the fraction of local QCD, local non-QCD SM, and local signal
events above the anomaly threshold. The argument it carries: the components the
signal occupies are also the ones where ordinary non-QCD SM is flagged almost as
often, so a high score there largely measures departure from QCD rather than
sensitivity to the signal — which is what motivates the local observable ranking.

Design notes.
  - The nominal background acceptance (10% by construction, calibrated globally on
    validation QCD) is drawn as a reference line. Per-component QCD flag rates far
    above it are the point of the figure, not a bug: the global rate is an average
    over regions that behave very differently.
  - Rates estimated from fewer than --min-n events are drawn hatched and greyed.
    Several components hold a handful of QCD events, where a rate of 1.0 means
    "4 of 4" and should not be read as a strong statement.
  - Components holding the flagged signal are marked under the axis, so the
    "non-specific exactly where the signal lives" reading is visible directly.
  - Components are identified as C<k> only: the paper characterises regions by
    physics, never by a dominant process label.

Colours come from the shared Okabe-Ito scheme in `common/style.py`. This figure
is where it is anchored, because it is the only one showing all three populations
at once: QCD in sky blue, non-QCD SM in blue, signal in vermillion. Every other
figure reuses those assignments, so the scheme only has to be learned once.

Usage:
    python scripts/xai/plot_03_flagrate.py \\
        --run-dir /eos/.../xai_paper/vcreg_d256_seed3_smnorm
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_XAI = Path(__file__).resolve().parent
sys.path.insert(0, str(_XAI))
from common.style import OI  # noqa: E402

# This figure fixes the scheme the rest of the set follows: one hue per meaning.
C_QCD = OI["qcd"]        # the AE's trained normality
C_NONQCD = OI["sm"]      # background, but not what the AE was trained on
C_SIGNAL = OI["signal"]  # the signal proxy
C_LOWSTAT = OI["light"]


def load_rates(run_dir: Path) -> list[dict]:
    path = run_dir / "03_assign" / "flag_rate_per_component.csv"
    rows = []
    for r in csv.DictReader(path.open()):
        def num(key):
            v = r.get(key, "")
            return float(v) if v not in ("", None) else np.nan
        rows.append({
            "component": int(r["component"]),
            "n_qcd": int(r["n_qcd"]), "rate_qcd": num("flag_rate_qcd"),
            "n_nonqcd": int(r["n_nonqcd_sm"]), "rate_nonqcd": num("flag_rate_nonqcd_sm"),
            "n_signal": int(r["n_signal"]), "rate_signal": num("flag_rate_signal"),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="Default: <run-dir>/03_assign/plots/flag_rate_per_component.pdf")
    p.add_argument("--fpr", type=float, default=0.10,
                   help="Nominal background acceptance, drawn as reference line")
    p.add_argument("--min-n", type=int, default=200,
                   help="Below this many events a rate is drawn as low-statistics")
    p.add_argument("--signal-frac-min", type=float, default=0.05,
                   help="Mark components holding at least this fraction of the signal")
    args = p.parse_args()

    rows = load_rates(args.run_dir)
    out = args.output or (args.run_dir / "03_assign" / "plots" / "flag_rate_per_component.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    comps = [r["component"] for r in rows]
    x = np.arange(len(comps))
    width = 0.27

    total_signal = sum(r["n_signal"] for r in rows) or 1
    populated = {r["component"] for r in rows
                 if r["n_signal"] / total_signal >= args.signal_frac_min}

    fig, ax = plt.subplots(figsize=(max(8.0, len(comps) * 0.78), 4.3))

    series = [
        (-width, "rate_qcd", "n_qcd", C_QCD, "local QCD"),
        (0.0, "rate_nonqcd", "n_nonqcd", C_NONQCD, "local non-QCD SM"),
        (width, "rate_signal", "n_signal", C_SIGNAL, "local signal"),
    ]
    for off, rate_key, n_key, color, _ in series:
        vals = np.array([r[rate_key] for r in rows], dtype=float)
        ns = np.array([r[n_key] for r in rows])
        lowstat = ns < args.min_n
        # Signal carries a dotted fill as well as its hue: it sits next to the
        # non-QCD SM bar, and those two hues are only 0.07 apart in luminance, so
        # colour alone does not survive greyscale printing.
        hatch = ".." if color == C_SIGNAL else None
        ax.bar(x + off, np.where(lowstat, np.nan, vals), width,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.0, zorder=3)
        ax.bar(x + off, np.where(lowstat, vals, np.nan), width,
               color=C_LOWSTAT, hatch="///", edgecolor=OI["neutral"], linewidth=0.0, zorder=3)

    ax.axhline(args.fpr, color="black", lw=1.0, ls="--", zorder=4)
    ax.annotate(f"nominal background acceptance ({args.fpr:.0%})",
                xy=(len(comps) - 0.5, args.fpr), xytext=(-4, 5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in comps])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction above anomaly threshold")
    ax.set_xlabel("GMM component")
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Secondary encoding for the components that hold the signal.
    for i, c in enumerate(comps):
        if c in populated:
            ax.annotate("$\\bigstar$", xy=(i, 0), xytext=(0, -22),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=10, color=C_SIGNAL, annotation_clip=False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c,
                             hatch=".." if c == C_SIGNAL else None, edgecolor="white")
               for _, _, _, c, _ in series]
    labels = [lab for _, _, _, _, lab in series]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=C_LOWSTAT, hatch="///",
                                 edgecolor=OI["neutral"]))
    labels.append(f"$n < {args.min_n}$")
    handles.append(plt.Line2D([], [], marker="*", ls="none", color=C_SIGNAL, markersize=9))
    labels.append("holds flagged signal")
    ax.legend(handles, labels, fontsize=8.5, ncol=5, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.16))

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    for r in rows:
        if r["component"] in populated:
            print(f"  C{r['component']}: QCD {r['rate_qcd']:.3f} | "
                  f"non-QCD SM {r['rate_nonqcd']:.3f} | signal {r['rate_signal']:.3f} "
                  f"(n_signal={r['n_signal']:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
