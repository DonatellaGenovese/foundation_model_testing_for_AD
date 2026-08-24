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
    "n_bjets": r"$n_{b}$",
    "n_leptons": r"$n_{\mathrm{leptons}}$",
    "Mjj": r"$M_{jj}$ [GeV]",
    "deta_jj": r"$|\Delta\eta_{jj}|$",
    "MT": r"$M_T$ [GeV]",
}
PHYSICS_BINS = {
    "HT": np.linspace(0, 2000, 50),
    "MET": np.linspace(0, 500, 50),
    "n_jets": np.arange(-0.5, 12.5, 1),
    "n_bjets": np.arange(-0.5, 6.5, 1),
    "n_leptons": np.arange(-0.5, 5.5, 1),
    "Mjj": np.linspace(0, 3000, 50),
    "deta_jj": np.linspace(0, 8, 50),
    "MT": np.linspace(0, 200, 50),
}
