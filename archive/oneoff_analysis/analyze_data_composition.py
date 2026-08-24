"""
Analyze the composition of the training dataset.

For each of the 15 training classes, reads parquet files and computes:
- Average number of jets, electrons, muons, photons per event
- Total counts for 1,000,000 events per class
"""

import json
from pathlib import Path
import numpy as np
import awkward as ak
import pyarrow.parquet as pq

RAW_DATA_BASE = Path("/eos/project/f/foundational-model-dataset/samples/production_final")
N_EVENTS_PER_CLASS = 1_000_000

# 15 training classes and their directory names
CLASSES = {
    "QCD_inclusive":    "QCD_HT50toInf",
    "DY_to_ll":         "DYJetsToLL_13TeV-madgraphMLM-pythia8",
    "Z_to_vv_jet":      "ZJetsTovv_13TeV-madgraphMLM-pythia8",
    "Z_to_qq_uds":      "ZJetsToQQ_13TeV-madgraphMLM-pythia8",
    "Z_to_bb":          "ZJetsTobb_13TeV-madgraphMLM-pythia8",
    "Z_to_cc":          "ZJetsTocc_13TeV-madgraphMLM-pythia8",
    "W_to_lv":          "WJetsToLNu_13TeV-madgraphMLM-pythia8",
    "W_to_qq":          "WJetsToQQ_13TeV-madgraphMLM-pythia8",
    "gamma":            "gamma",
    "QCD_bb":           "QCD_HT50tobb",
    "tt_all-hadr":      "tt0123j_5f_ckm_LO_MLM_hadronic",
    "tt_semi-lept":     "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    "tt_all-lept":      "tt0123j_5f_ckm_LO_MLM_leptonic",
    "Minbias_Soft_QCD": "minbias",
    "Upsilon_to_Leptons": "upsilon_to_leptons",
}

# Columns used to count objects (PT is present for every real object)
COUNT_COLS = {
    "jets":      "FullReco_JetPuppiAK4_PT",
    "electrons": "FullReco_Electron_PT",
    "muons":     "FullReco_MuonTight_PT",
    "photons":   "FullReco_PhotonTight_PT",
}


def count_objects_in_file(parquet_path: Path, max_events: int):
    """Return dict of {obj_type: list_of_counts_per_event} for up to max_events."""
    table = pq.read_table(str(parquet_path), columns=list(COUNT_COLS.values()))
    arr = ak.from_arrow(table)
    n = min(len(arr), max_events)
    counts = {}
    for obj_type, col in COUNT_COLS.items():
        if col in arr.fields:
            c = ak.to_numpy(ak.num(arr[col][:n]))
        else:
            c = np.zeros(n, dtype=np.int32)
        counts[obj_type] = c
    return counts, n


def analyze_class(class_name: str, folder: str) -> dict:
    class_dir = RAW_DATA_BASE / folder
    parquet_files = sorted([f for f in class_dir.glob("*.parquet") if f.name != "event.parquet"])

    if not parquet_files:
        print(f"  [WARN] No parquet files found for {class_name}")
        return {}

    totals = {k: [] for k in COUNT_COLS}
    events_read = 0

    for pf in parquet_files:
        remaining = N_EVENTS_PER_CLASS - events_read
        if remaining <= 0:
            break
        try:
            counts, n = count_objects_in_file(pf, remaining)
        except Exception as e:
            print(f"  [WARN] Skipping {pf.name}: {e}")
            continue
        for obj_type in COUNT_COLS:
            totals[obj_type].append(counts.get(obj_type, np.zeros(n)))
        events_read += n
        if events_read % 100_000 == 0:
            print(f"  {class_name}: {events_read:,} events read")

    if events_read == 0:
        return {}

    result = {"n_events": events_read}
    for obj_type in COUNT_COLS:
        all_counts = np.concatenate(totals[obj_type])
        result[obj_type] = {
            "mean":   float(np.mean(all_counts)),
            "median": float(np.median(all_counts)),
            "std":    float(np.std(all_counts)),
            "total":  int(np.sum(all_counts)),
            "max":    int(np.max(all_counts)),
            "frac_zero": float(np.mean(all_counts == 0)),
        }
    return result


def main():
    results = {}
    for class_name, folder in CLASSES.items():
        print(f"\nAnalyzing {class_name} ({folder})...")
        results[class_name] = analyze_class(class_name, folder)

    # Save raw JSON
    out_path = Path("scripts/data_composition_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Class':<22} {'Events':>10} {'Jets(mean)':>12} {'Electrons(mean)':>16} {'Muons(mean)':>13} {'Photons(mean)':>14}")
    print("=" * 100)
    for cls, r in results.items():
        if not r:
            continue
        jets  = r.get("jets", {}).get("mean", 0)
        elecs = r.get("electrons", {}).get("mean", 0)
        muons = r.get("muons", {}).get("mean", 0)
        phots = r.get("photons", {}).get("mean", 0)
        n     = r.get("n_events", 0)
        print(f"{cls:<22} {n:>10,} {jets:>12.3f} {elecs:>16.3f} {muons:>13.3f} {phots:>14.3f}")

    # Also print totals for 1M events
    print("\n" + "=" * 100)
    print(f"{'Class':<22} {'Jets(total)':>13} {'Electrons(tot)':>16} {'Muons(tot)':>13} {'Photons(tot)':>14}")
    print("=" * 100)
    for cls, r in results.items():
        if not r:
            continue
        n     = r.get("n_events", 0)
        scale = N_EVENTS_PER_CLASS / n if n > 0 else 1.0
        jets  = int(r.get("jets",  {}).get("total", 0) * scale)
        elecs = int(r.get("electrons", {}).get("total", 0) * scale)
        muons = int(r.get("muons", {}).get("total", 0) * scale)
        phots = int(r.get("photons", {}).get("total", 0) * scale)
        print(f"{cls:<22} {jets:>13,} {elecs:>16,} {muons:>13,} {phots:>14,}")

    print("\nFraction of events with ZERO objects:")
    print(f"{'Class':<22} {'frac_zero_jets':>16} {'frac_zero_elec':>16} {'frac_zero_muon':>16} {'frac_zero_phot':>16}")
    print("-" * 90)
    for cls, r in results.items():
        if not r:
            continue
        fj = r.get("jets",      {}).get("frac_zero", 0)
        fe = r.get("electrons", {}).get("frac_zero", 0)
        fm = r.get("muons",     {}).get("frac_zero", 0)
        fp = r.get("photons",   {}).get("frac_zero", 0)
        print(f"{cls:<22} {fj:>16.3f} {fe:>16.3f} {fm:>16.3f} {fp:>16.3f}")


if __name__ == "__main__":
    main()
