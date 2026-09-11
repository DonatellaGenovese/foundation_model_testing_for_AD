"""Shared labels and physics variable definitions (paper Table gmm_variables)."""

from __future__ import annotations

import numpy as np

SM_INDICES = list(range(12))
BSM_INDICES = [12, 13, 14]
HH4B_LABEL = 13

# 20+ are CASE-production signals, kept clear of the 0-14 range used by the SM
# classes and the three proxy signals so a mixed matched array stays unambiguous.
SIG_LABELS = {12: "VBFHbb", 13: "HH4b", 14: "ggHtautau",
              20: "HV Zp1000 -> mumu"}

# Rendered form for figures. Kept apart from SIG_LABELS because that one also seeds
# output filenames and JSON metadata, where maths markup has no place: a slug built
# from "$Z^{\prime} \to n(\mu\mu)$" is unreadable, and the plain name stays the
# stable handle across reruns. The dimuon signal is written as n(mumu) because the
# hidden-valley Z' decays through dark pions, giving several muon pairs per event
# rather than one -- which is why its lepton multiplicity sits at a median of seven.
SIG_LABELS_TEX = {
    12: r"VBF $H \to b\bar{b}$",
    13: r"$HH \to 4b$",
    14: r"$ggH \to \tau\tau$",
    20: r"$Z^{\prime} \to n(\mu\mu)$",
}

CLASS_NAMES = {
    0: "QCD_incl",
    1: "Z→νν+j",
    2: "Z→qq(uds)",
    3: "Z→bb",
    4: "Z→cc",
    5: "W→lν",
    6: "W→qq",
    7: "γ+jets",
    8: "QCD_bb",
    9: "tt hadr",
    10: "tt semi-l",
    11: "tt lept",
    12: "VBFHbb",
    13: "HH→4b",
    14: "ggH→ττ",
}

SM_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#9a6324", "#469990", "#dcbeff", "#800000",
]

# Folder names under vectorized/{split}/ (allSM Higgs dataset)
CLASS_FOLDERS = {
    0: "QCD_HT50toInf",
    1: "ZJetsTovv_13TeV-madgraphMLM-pythia8",
    2: "ZJetsToQQ_13TeV-madgraphMLM-pythia8",
    3: "ZJetsTobb_13TeV-madgraphMLM-pythia8",
    4: "ZJetsTocc_13TeV-madgraphMLM-pythia8",
    5: "WJetsToLNu_13TeV-madgraphMLM-pythia8",
    6: "WJetsToQQ_13TeV-madgraphMLM-pythia8",
    7: "gamma",
    8: "QCD_HT50tobb",
    9: "tt0123j_5f_ckm_LO_MLM_hadronic",
    10: "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    11: "tt0123j_5f_ckm_LO_MLM_leptonic",
    12: "VBFHbb",
    13: "HH_4b",
    14: "ggHtautau",
}

PHYSICS_VARS = ["HT", "MET", "n_jets", "n_bjets", "n_leptons", "Mjj", "deta_jj", "MT"]
PHYSICS_LABELS = {
    "HT": r"$H_T$ [GeV]",
    "MET": "MET [GeV]",
    "n_jets": r"$n_{\mathrm{jets}}$",
    # n_{b-tag}, not n_b: the count is of jets passing a b-tagging working point, not of
    # b quarks, and the paper's Table 3 and Appendix A.5 both name it that way. \mathrm
    # rather than \text to match the two labels around it; they render the same.
    "n_bjets": r"$n_{b\mathrm{-tag}}$",
    "n_leptons": r"$n_{\mathrm{leptons}}$",
    "Mjj": r"$M_{jj}$ [GeV]",
    "deta_jj": r"$|\Delta\eta_{jj}|$",
    "MT": r"$M_T$ [GeV]",
}
PHYSICS_BINS = {
    "HT": np.linspace(0, 2000, 50),
    "MET": np.linspace(0, 500, 50),
    "n_jets": np.arange(-0.5, 12.5, 1),
    # Up to 12, the jet top-k, rather than 6: 9.6% of the flagged HH->4b signal carries
    # more than six b-tags and was being dropped from the figure without warning --- on
    # the very observable the ranking puts first for that signal.
    "n_bjets": np.arange(-0.5, 12.5, 1),
    # Up to 18. Two separate reasons pushed this past the Standard Model's own range,
    # which stops at 6. First, matplotlib drops out-of-range entries silently, and the
    # original ceiling of 5 hid 81% of the flagged dimuon signal. Second, once the count
    # was read from the reconstructed lists rather than from the padded arrays, its tail
    # ran to 23 and a ceiling of 12 still cut 10%. 18 holds all but 0.3% of that signal
    # while keeping the Standard Model, which lives in the first six bins, resolved
    # enough to compare against.
    "n_leptons": np.arange(-0.5, 18.5, 1),
    "Mjj": np.linspace(0, 3000, 50),
    "deta_jj": np.linspace(0, 8, 50),
    # Up to 600, not 200. The old range was set on the Standard Model, whose transverse
    # mass sits under the W peak (95th percentile 148 GeV), but half the flagged dimuon
    # signal lies above 200 GeV. 600 covers its 95th percentile and keeps the Standard
    # Model peak resolved; the remaining few per cent of the signal tail is out of range
    # in exchange for not flattening the bulk of both distributions.
    "MT": np.linspace(0, 600, 50),
}
