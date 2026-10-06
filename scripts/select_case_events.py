#!/usr/bin/env python3
"""Choose the CASE signal events a run scores, and write the choice down.

The collide_v2 samples hold 50,000 events per signal, while the proxy signals of the
main table contribute 20,000 test events each. To compare like with like, a run scores
20,000 CASE events too — and which 20,000 is a choice this script makes explicit:

    random         20,000 drawn uniformly, the signal as it comes (the paper's)
    untruncated    20,000 among the events the vectorisation does not truncate,
                   i.e. at most eight muons and eight electrons (--modes random untruncated)

The second is a check. It exists because the dimuon signal runs to 23 muons while the
encoder input keeps eight (`topk: 8`): in collide_v2, 27.6% of its events are
truncated. The two selections measure what that truncation does to the results. For the h->aa signals
nothing is truncated (at most four muons), so there the two differ only by the draw.

Indices are positions in the concatenated test shards of that class, files sorted by
name and rows in order — the order `load_shards` reads and the order the extraction
writes embeddings in, with one loader worker. The shard file and row are stored too, so
a selection can be checked, or re-applied, without relying on that order.

The defaults are the tree scripts/prepare_case_smnorm.py builds on CERN EOS.

Usage:
    python scripts/select_case_events.py --case-label HVdilep_Zp1000_piD2_mumu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAP = 8   # topk for electrons and for muons in the vectorisation config
DATA = Path("/eos/user/d/dgenoves/foundation_model_testing_data")
CASE_DATA = DATA / "v3_nosparse_case_smnorm_highlevel"


def shard_rows(class_dir: Path):
    """(file names, rows per file) of the test shards, in the order they are read."""
    files = sorted(f for f in class_dir.iterdir() if f.name.endswith("_x.npy"))
    if not files:
        raise FileNotFoundError(f"no *_x.npy under {class_dir}")
    return [f.name for f in files], [len(np.load(f, mmap_mode="r")) for f in files]


def lepton_counts(folder: Path):
    """Muon and electron multiplicity per event, before truncation, empty events dropped.

    The vectorisation drops events in which every jet, electron, muon and photon slot is
    empty, so the same events are dropped here and the caller checks the count against
    the shards.
    """
    import pyarrow.parquet as pq

    cols = ["FullReco_MuonTight_PT", "FullReco_Electron_PT",
            "FullReco_JetPuppiAK4_PT", "FullReco_PhotonTight_PT"]
    per_file = []
    for f in sorted(folder.glob("*.parquet")):
        tab = pq.read_table(f, columns=cols)
        per_file.append(np.stack([np.array([len(x) if x is not None else 0
                                            for x in tab.column(c).to_pylist()])
                                  for c in cols], axis=1))
    n = np.concatenate(per_file, axis=0)
    keep = n.sum(axis=1) > 0
    return n[keep, 0], n[keep, 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-label", required=True, help="CASE folder name of the signal")
    ap.add_argument("--preproc-dir", type=Path, default=CASE_DATA / "preprocessed" / "test",
                    help="The test split of the preprocessed tree (holds one folder per class)")
    ap.add_argument("--case-src", type=Path, default=DATA / "_case_src_v2",
                    help="Directory of symlinks to the CASE productions, for the parquet")
    ap.add_argument("--out", type=Path, default=CASE_DATA / "selections",
                    help="Directory for the selection files")
    ap.add_argument("--n", type=int, default=20_000, help="Events per selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", nargs="+", default=["random"], choices=["random", "untruncated"],
                    help="Selections to write. The paper uses `random`; `untruncated` is the "
                         "check on the eight-muon limit")
    a = ap.parse_args()

    names, rows = shard_rows(a.preproc_dir / a.case_label)
    total = int(sum(rows))
    file_of = np.repeat(np.arange(len(names)), rows)
    row_of = np.concatenate([np.arange(n) for n in rows])

    n_mu, n_el = lepton_counts(a.case_src / a.case_label)
    if len(n_mu) != total:
        raise SystemExit(f"{a.case_label}: parquet gives {len(n_mu):,} non-empty events "
                         f"against {total:,} in the shards; the two are not aligned")

    untruncated = (n_mu <= CAP) & (n_el <= CAP)
    print(f"{a.case_label}: {total:,} test events in {len(names)} shards; "
          f"truncated {int((~untruncated).sum()):,} ({(~untruncated).mean():.1%}), "
          f"muons max {int(n_mu.max())}, electrons max {int(n_el.max())}")

    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    pools = {"random": np.arange(total), "untruncated": np.flatnonzero(untruncated)}
    for mode in ("random", "untruncated"):
        if mode not in a.modes:
            continue
        pool = pools[mode]
        if len(pool) < a.n:
            raise SystemExit(f"{a.case_label}/{mode}: only {len(pool):,} events available for {a.n:,}")
        idx = np.sort(rng.choice(pool, a.n, replace=False))
        out = a.out / f"{a.case_label}_{mode}.npz"
        # 'shard', not 'file': np.savez_compressed takes the output path as `file`.
        np.savez_compressed(out, index=idx.astype(np.int32), shard=file_of[idx].astype(np.int16),
                            row=row_of[idx].astype(np.int32), n_muons=n_mu[idx].astype(np.int16),
                            n_electrons=n_el[idx].astype(np.int16), files=np.array(names))
        meta = {"case_label": a.case_label, "mode": mode, "n": int(a.n), "seed": a.seed,
                "available": int(len(pool)), "total_test_events": total,
                "muons_max": int(n_mu[idx].max()), "truncated_in_selection": int((n_mu[idx] > CAP).sum())}
        Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
        print(f"  {mode:12s} -> {out.name}  muons max {meta['muons_max']}, "
              f"truncated {meta['truncated_in_selection']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
