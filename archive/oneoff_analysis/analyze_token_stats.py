"""
Token-level statistical analysis of the 15 training classes.

For each class (1M events), computes per-object-type and total token statistics,
where a "token" is a non-padded particle slot after topk selection.

Topology:
  jets     → topk = 12  (sorted by PT)
  electrons → topk = 8
  muons    → topk = 8
  photons  → topk = 8
  MET      → always 1 (scalar, always present)
  event    → always 1 (event token, always present)

"Real tokens" per event = min(n_jets,12) + min(n_electrons,8) + min(n_muons,8) + min(n_photons,8)
Total tokens fed to transformer = real_tokens + 2 (MET + event token)
"""

import json
from pathlib import Path
import numpy as np
import awkward as ak
import pyarrow.parquet as pq

RAW_DATA_BASE = Path("/eos/project/f/foundational-model-dataset/samples/production_final")
N_EVENTS_PER_CLASS = 1_000_000

TOPK = {"jets": 12, "electrons": 8, "muons": 8, "photons": 8}
PT_COLS = {
    "jets":      "FullReco_JetPuppiAK4_PT",
    "electrons": "FullReco_Electron_PT",
    "muons":     "FullReco_MuonTight_PT",
    "photons":   "FullReco_PhotonTight_PT",
}

CLASSES = {
    "QCD_inclusive":      "QCD_HT50toInf",
    "DY_to_ll":           "DYJetsToLL_13TeV-madgraphMLM-pythia8",
    "Z_to_vv_jet":        "ZJetsTovv_13TeV-madgraphMLM-pythia8",
    "Z_to_qq_uds":        "ZJetsToQQ_13TeV-madgraphMLM-pythia8",
    "Z_to_bb":            "ZJetsTobb_13TeV-madgraphMLM-pythia8",
    "Z_to_cc":            "ZJetsTocc_13TeV-madgraphMLM-pythia8",
    "W_to_lv":            "WJetsToLNu_13TeV-madgraphMLM-pythia8",
    "W_to_qq":            "WJetsToQQ_13TeV-madgraphMLM-pythia8",
    "gamma":              "gamma",
    "QCD_bb":             "QCD_HT50tobb",
    "tt_all-hadr":        "tt0123j_5f_ckm_LO_MLM_hadronic",
    "tt_semi-lept":       "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    "tt_all-lept":        "tt0123j_5f_ckm_LO_MLM_leptonic",
    "Minbias_Soft_QCD":   "minbias",
    "Upsilon_to_Leptons": "upsilon_to_leptons",
}

PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def stats(arr: np.ndarray) -> dict:
    """Compute descriptive statistics for a 1D integer array."""
    pcts = np.percentile(arr, PERCENTILES).astype(float).tolist()
    return {
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr)),
        "min":   int(np.min(arr)),
        "p5":    pcts[1],
        "p10":   pcts[2],
        "p25":   pcts[3],
        "p50":   pcts[4],
        "p75":   pcts[5],
        "p90":   pcts[6],
        "p95":   pcts[7],
        "p99":   pcts[8],
        "max":   int(np.max(arr)),
        "frac_zero": float(np.mean(arr == 0)),
    }


def token_counts_from_file(parquet_path: Path, max_events: int) -> tuple[dict, int]:
    """Read one parquet file, return raw object counts (capped at topk) per event."""
    table = pq.read_table(str(parquet_path), columns=list(PT_COLS.values()))
    arr = ak.from_arrow(table)
    n = min(len(arr), max_events)
    counts = {}
    for obj, col in PT_COLS.items():
        raw = ak.to_numpy(ak.num(arr[col][:n]))
        counts[obj] = np.minimum(raw, TOPK[obj]).astype(np.int32)
    return counts, n


def analyze_class(class_name: str, folder: str) -> dict:
    class_dir = RAW_DATA_BASE / folder
    parquet_files = sorted(
        f for f in class_dir.glob("*.parquet") if f.name != "event.parquet"
    )
    if not parquet_files:
        print(f"  [WARN] No files for {class_name}")
        return {}

    buffers = {k: [] for k in PT_COLS}
    events_read = 0

    for pf in parquet_files:
        remaining = N_EVENTS_PER_CLASS - events_read
        if remaining <= 0:
            break
        try:
            counts, n = token_counts_from_file(pf, remaining)
        except Exception as e:
            print(f"  [WARN] {pf.name}: {e}")
            continue
        for obj in PT_COLS:
            buffers[obj].append(counts[obj])
        events_read += n

    if events_read == 0:
        return {}

    # Concatenate
    per_obj = {obj: np.concatenate(buffers[obj]) for obj in PT_COLS}

    # Real particle tokens = sum across types (each already capped at topk)
    real_tokens = sum(per_obj[obj] for obj in PT_COLS)

    # Total tokens fed to transformer = real + 2 (MET always present + event token)
    total_tokens = real_tokens + 2

    result = {
        "n_events": events_read,
        "per_type": {obj: stats(per_obj[obj]) for obj in PT_COLS},
        "real_particle_tokens": stats(real_tokens),
        "total_transformer_tokens": stats(total_tokens),
        # Fraction of events with ≤N real particle tokens (attention utility thresholds)
        "frac_real_tokens_leq": {
            str(t): float(np.mean(real_tokens <= t)) for t in [0, 1, 2, 3, 4, 5]
        },
        # Distribution: fraction of events with exactly N total tokens
        "token_count_distribution": {
            str(t): float(np.mean(total_tokens == t))
            for t in range(0, int(total_tokens.max()) + 1)
            if np.sum(total_tokens == t) > 0
        },
    }
    return result


def print_report(results: dict):
    SEP = "=" * 110

    # ------------------------------------------------------------------ #
    # TABLE 1: Mean tokens per type                                       #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("TABLE 1 — Mean real tokens per object type (capped at topk) + total real tokens + total transformer tokens")
    print(f"{'Class':<22} {'Jets≤12':>8} {'Elec≤8':>8} {'Muon≤8':>8} {'Phot≤8':>8}  {'RealTok':>9} {'TransfTok':>11}")
    print(SEP)
    for cls, r in results.items():
        if not r:
            continue
        j = r["per_type"]["jets"]["mean"]
        e = r["per_type"]["electrons"]["mean"]
        m = r["per_type"]["muons"]["mean"]
        p = r["per_type"]["photons"]["mean"]
        rt = r["real_particle_tokens"]["mean"]
        tt = r["total_transformer_tokens"]["mean"]
        print(f"{cls:<22} {j:>8.3f} {e:>8.3f} {m:>8.3f} {p:>8.3f}  {rt:>9.3f} {tt:>11.3f}")

    # ------------------------------------------------------------------ #
    # TABLE 2: Percentiles of real particle tokens                        #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("TABLE 2 — Real particle token count percentiles (min / p10 / p25 / p50 / p75 / p90 / p99 / max)")
    print(f"{'Class':<22} {'min':>5} {'p10':>5} {'p25':>5} {'p50':>5} {'p75':>5} {'p90':>5} {'p99':>5} {'max':>5}")
    print(SEP)
    for cls, r in results.items():
        if not r:
            continue
        s = r["real_particle_tokens"]
        print(f"{cls:<22} {int(s['min']):>5} {int(s['p10']):>5} {int(s['p25']):>5} "
              f"{int(s['p50']):>5} {int(s['p75']):>5} {int(s['p90']):>5} {int(s['p99']):>5} {int(s['max']):>5}")

    # ------------------------------------------------------------------ #
    # TABLE 3: Fraction of events with ≤N real particle tokens           #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("TABLE 3 — Fraction of events with ≤N real particle tokens (padding pressure)")
    print(f"{'Class':<22} {'≤0':>8} {'≤1':>8} {'≤2':>8} {'≤3':>8} {'≤4':>8} {'≤5':>8}")
    print(SEP)
    for cls, r in results.items():
        if not r:
            continue
        d = r["frac_real_tokens_leq"]
        print(f"{cls:<22} "
              f"{d['0']:>8.3f} {d['1']:>8.3f} {d['2']:>8.3f} "
              f"{d['3']:>8.3f} {d['4']:>8.3f} {d['5']:>8.3f}")

    # ------------------------------------------------------------------ #
    # TABLE 4: Padding fraction per type                                  #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("TABLE 4 — Fraction of events with ZERO objects of each type (i.e. fully padded slots)")
    print(f"{'Class':<22} {'Jets=0':>9} {'Elec=0':>9} {'Muon=0':>9} {'Phot=0':>9}")
    print(SEP)
    for cls, r in results.items():
        if not r:
            continue
        fj = r["per_type"]["jets"]["frac_zero"]
        fe = r["per_type"]["electrons"]["frac_zero"]
        fm = r["per_type"]["muons"]["frac_zero"]
        fp = r["per_type"]["photons"]["frac_zero"]
        print(f"{cls:<22} {fj:>9.3f} {fe:>9.3f} {fm:>9.3f} {fp:>9.3f}")

    # ------------------------------------------------------------------ #
    # TABLE 5: Padding overhead                                           #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("TABLE 5 — Mean padding tokens per event (padded slots = topk − real tokens per type)")
    print(f"   Fixed slots: jets_topk=12, elec_topk=8, muon_topk=8, phot_topk=8  → max real = 36")
    print(f"{'Class':<22} {'PadJets':>9} {'PadElec':>9} {'PadMuon':>9} {'PadPhot':>9}  {'TotPad':>9} {'PadFrac':>9}")
    print(SEP)
    for cls, r in results.items():
        if not r:
            continue
        pj = TOPK["jets"]      - r["per_type"]["jets"]["mean"]
        pe = TOPK["electrons"] - r["per_type"]["electrons"]["mean"]
        pm = TOPK["muons"]     - r["per_type"]["muons"]["mean"]
        pp = TOPK["photons"]   - r["per_type"]["photons"]["mean"]
        tot_pad = pj + pe + pm + pp
        frac_pad = tot_pad / 36.0
        print(f"{cls:<22} {pj:>9.2f} {pe:>9.2f} {pm:>9.2f} {pp:>9.2f}  {tot_pad:>9.2f} {frac_pad:>9.3f}")


def main():
    results = {}
    for class_name, folder in CLASSES.items():
        print(f"Analyzing {class_name}...")
        results[class_name] = analyze_class(class_name, folder)

    out_path = Path("scripts/token_stats_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {out_path}\n")

    print_report(results)


if __name__ == "__main__":
    main()
