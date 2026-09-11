#!/usr/bin/env python3
"""Build the side-by-side Wasserstein ranking table for Section 4.3.

Step 04 writes one standalone `table` float per signal. Two floats cannot be placed
on the same line by LaTeX, and separately they invite the reader to compare the two
signals by flipping pages. This merges them into a single float holding two
minipages, so HH->4b and HV Z'->mumu sit next to each other and the leading
observable of each is read in one glance.

Three things are fixed here that the auto-generated fragments get wrong for print:
observables are emitted as maths (`n_b`, not `n_bjets`), distances are printed to a
fixed two decimals instead of `%.3g` (which mixes `2.7` with `2.73` down a column),
and the signal names are LaTeX rather than the internal slugs.

The QCD column is the control. It is an em dash wherever the component holds too
few QCD events for the distance to be defined --- which is itself the point in C2,
where a leptonic signal lands in a region the trained normality never covered.

Usage:
    python paper/make_wasserstein_table.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_XP = Path("/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/rank_k7_sm")
DEFAULT_OUT = Path(__file__).resolve().parent / "sections" / "xai" / "wasserstein_side_by_side.tex"

# Components holding less than this share of the flagged signal are left out. Step 04
# populates the ranking down to 5%, which for the dimuon signal admits C5 at 7.8%
# against C2's 88% --- a second block carrying a fourteenth of the events, whose middle
# ranks are not separable by bootstrap and whose figure counterpart (Fig. 5b, drawn with
# --components) already omits it. At 10% the same rule keeps both HH->4b components,
# which split 55.6/43.8, and leaves the dimuon signal with C2 alone, so the asymmetry
# between the two halves of the table follows from one threshold applied to both signals
# rather than from a per-signal choice.
MIN_FRAC_FLAGGED = 0.10

# Match the notation of Table 3 in the main text. The jet gap keeps its absolute-value
# bars, as the figures have them: physics.py builds the observable as |eta_1 - eta_2|.
MATHS = {
    "HT": r"$H_T$",
    "MET": r"MET",
    "n_jets": r"$n_\mathrm{jets}$",
    # n_{b-tag}: jets passing a working point, not b quarks. \text here because this
    # string goes into LaTeX, unlike the figure labels which need \mathrm for mathtext.
    "n_bjets": r"$n_{b\text{-tag}}$",
    "n_leptons": r"$n_\mathrm{leptons}$",
    "Mjj": r"$M_{jj}$",
    "deta_jj": r"$|\Delta\eta_{jj}|$",
    "MT": r"$M_T$",
}

SIGNALS = [
    ("hh4b", r"$HH \to 4b$"),
    ("hvdilep", r"$Z^{\prime} \to n(\mu\mu)$"),
]


def num(raw: str) -> str:
    if raw in ("", "nan", "NaN"):
        return "---"
    return f"{float(raw):.2f}"


def block(xp: Path, slug: str, min_frac: float) -> list[str]:
    rows = list(csv.DictReader((xp / slug / "wasserstein_rank.csv").open()))
    ranked = [r for r in rows if r["rank"] not in ("", None)]
    by_comp: dict[int, list] = {}
    for r in ranked:
        by_comp.setdefault(int(r["component"]), []).append(r)

    # Share of the flagged signal each component holds, from step 04's own metadata
    # rather than from the CSV: the CSV only lists components that already passed the
    # step's 5% cut, so a fraction rebuilt from it would be normalised to the survivors.
    frac = json.loads((xp / slug / "profile_meta.json").read_text())["frac_per_component"]
    dropped = [c for c in by_comp if frac[c] < min_frac]
    for c in dropped:
        del by_comp[c]
    if dropped:
        print(f"  {slug}: dropped C{', C'.join(str(c) for c in sorted(dropped))} "
              f"({', '.join(f'{frac[c]:.1%}' for c in sorted(dropped))} of the flagged "
              f"signal, below {min_frac:.0%})")
    if not by_comp:
        raise SystemExit(f"{slug}: no component holds {min_frac:.0%} of the flagged signal")

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"& Observable & $W_1^{\mathrm{SM}}$ & $W_1^{\mathrm{QCD}}$ \\",
        r"\midrule",
    ]
    for comp in sorted(by_comp, key=lambda c: -frac[c]):
        blk = sorted(by_comp[comp], key=lambda r: int(r["rank"]))
        for i, r in enumerate(blk):
            head = rf"C{comp}" if i == 0 else ""
            obs = MATHS.get(r["variable"], r["variable"])
            sm, qcd = num(r["W1_sm"]), num(r["W1_qcd"])
            if i == 0:  # the leading observable carries the claim
                obs, sm = rf"\textbf{{{obs}}}", rf"\textbf{{{sm}}}"
            lines.append(rf"{head} & {obs} & {sm} & {qcd} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}"]
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-frac", type=float, default=MIN_FRAC_FLAGGED,
                   help="Drop components holding less than this share of the flagged "
                        f"signal (default {MIN_FRAC_FLAGGED:.2f})")
    p.add_argument("--xp", type=Path, default=DEFAULT_XP,
                   help="Directory holding <signal>/wasserstein_rank.csv (default: the "
                        "paper's rank_k7_sm)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    OUT = args.output

    out = [
        r"% Auto-generated by paper/make_wasserstein_table.py --- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Local discriminating power of the high-level observables within the "
        r"mixture components that hold the AE-flagged signal, for $HH \rightarrow 4b$ "
        r"(left) and $Z^{\prime} \to n(\mu\mu)$ (right). Only components holding at least "
        rf"${args.min_frac * 100:.0f}\%$ of the flagged signal are reported; "
        r"$Z^{\prime} \to n(\mu\mu)$ places $88\%$ of its flagged events in C2. "
        r"$W_1^{\text{SM}}$ is the "
        r"Wasserstein-1 distance between the flagged signal and the local Standard Model "
        r"background, standardised by each observable's global standard deviation. "
        r"$W_1^{\text{QCD}}$, computed against the local QCD background (the autoencoder's "
        r"trained normality), is reported as a control baseline; it is undefined where a "
        r"component holds too few QCD events for the distance to be estimated, which is "
        r"the case throughout C2, where a single QCD event falls among $29{,}106$ Standard "
        r"Model ones. Observables are ranked by "
        r"$W_1^{\text{SM}}$, with the top-ranked observable set in bold.}",
        r"\label{tab:wasserstein_rank}",
    ]
    for i, (slug, name) in enumerate(SIGNALS):
        out.append(r"\begin{minipage}[t]{0.48\textwidth}")
        out.append(r"\centering")
        out.append(rf"\textbf{{{name}}}\\[2pt]")
        out += block(args.xp, slug, args.min_frac)
        out.append(r"\end{minipage}")
        if i == 0:
            out.append(r"\hfill")
    out += [r"\end{table}", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
