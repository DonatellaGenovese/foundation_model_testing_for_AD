"""Shared figure style for the XAI pipeline — Okabe-Ito palette.

One hue per meaning, held across every figure: a reader who learns the scheme in
one plot can read the rest. The palette is colourblind-safe by construction, and
the three hues actually carried by data (sky blue, blue, vermillion) are well
separated in luminance, so the figures survive greyscale printing.

Import for the side effect on rcParams and use OI[...] for explicit assignments:

    from common.style import OI, DIVERGING, diverging_norm
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OI = {
    "qcd":     "#56B4E9",  # sky blue        — QCD, the AE's trained normality
    "sm":      "#0072B2",  # blue            — SM other than QCD
    "signal":  "#D55E00",  # vermillion      — signal / AE-flagged
    "latent":  "#009E73",  # bluish green
    "accent":  "#CC79A7",  # reddish purple
    "neutral": "#999999",  # everything the figure is not about
    "light":   "#DDDDDD",  # low-statistics / de-emphasised
}

mpl.rcParams.update({
    "axes.prop_cycle": mpl.cycler(color=[OI["qcd"], OI["sm"], OI["signal"],
                                         OI["latent"], OI["accent"]]),
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "font.size": 9,
})

# Deviation maps: blue below, white at zero, vermillion above. White must land on
# zero, so the norm is always centred there rather than at the midpoint of the
# data range.
DIVERGING = LinearSegmentedColormap.from_list("oi_div", [OI["sm"], "#FFFFFF", OI["signal"]])


def diverging_norm(vmax: float = 3.0) -> TwoSlopeNorm:
    """Symmetric norm centred on zero. `vmax` is made at least 3 so that figures
    from different runs share a scale and can be compared by eye."""
    v = max(3.0, float(vmax))
    return TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v)
