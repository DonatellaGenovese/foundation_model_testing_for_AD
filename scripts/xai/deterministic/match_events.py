#!/usr/bin/env python3
"""Recover which events, in which order, a published embeddings file was built from.

The published XAI embeddings were extracted through a shuffled, multi-worker DataLoader
with no seed, so the event set is not recorded anywhere. It can be recovered: encode
every event of the split once in a fixed order (encode_shards.py), then match each
published row to the encoded event it equals. Rows are matched within their class, first
by exact byte equality, then — for rows that differ only at floating-point level — by
nearest neighbour inside a narrow window on the first coordinate.

The result is an event list (file_id, row) in the published order, which encode_shards.py
--events turns back into the embeddings deterministically.

Usage:
    python scripts/xai/deterministic/match_events.py \\
        --published /eos/.../embeddings/train_embeddings.npz \\
        --encoded /eos/.../all_train --out /eos/.../train_events.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def match_class(pub: np.ndarray, cand: np.ndarray, tol0: float) -> tuple[np.ndarray, np.ndarray]:
    """For each row of pub, the index in cand of the closest row, and the L-inf distance."""
    idx = np.full(len(pub), -1, dtype=np.int64)
    dist = np.full(len(pub), np.inf, dtype=np.float64)
    # 1. exact byte equality
    table = {cand[i].tobytes(): i for i in range(len(cand))}
    for i in range(len(pub)):
        j = table.get(pub[i].tobytes())
        if j is not None:
            idx[i], dist[i] = j, 0.0
    todo = np.where(idx < 0)[0]
    if len(todo):
        # 2. nearest neighbour within a window on the first coordinate
        order = np.argsort(cand[:, 0], kind="stable")
        c0 = cand[order, 0]
        for i in todo:
            q = pub[i]
            tol = tol0
            # Exact nearest neighbour in L-inf: a candidate whose first coordinate is
            # further than the best distance found cannot be closer, so widen the window
            # until the best match lies inside it.
            while True:
                lo, hi = np.searchsorted(c0, q[0] - tol), np.searchsorted(c0, q[0] + tol)
                w = order[lo:hi]
                if len(w) == 0:
                    if tol > 1e3:
                        break
                    tol *= 10
                    continue
                d = np.abs(cand[w] - q).max(axis=1)
                k = int(np.argmin(d))
                if d[k] <= tol:
                    idx[i], dist[i] = w[k], float(d[k])
                    break
                tol = float(d[k])
    return idx, dist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", type=Path, required=True, help="published <split>_embeddings.npz")
    ap.add_argument("--encoded", type=Path, required=True, help="encode_shards.py output of the whole split")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-5, help="initial window on the first coordinate")
    a = ap.parse_args()

    p = np.load(a.published)
    pub, plab = p["embeddings"].astype(np.float32), p["labels"].astype(np.int64)
    enc = np.load(a.encoded / "embeddings.npy", mmap_mode="r")
    elab = np.load(a.encoded / "labels.npy")
    efile, erow = np.load(a.encoded / "file_id.npy"), np.load(a.encoded / "row.npy")

    file_id = np.full(len(pub), -1, dtype=np.int32)
    row = np.full(len(pub), -1, dtype=np.int32)
    dist = np.full(len(pub), np.inf)
    report = {}
    for c in np.unique(plab):
        pi = np.where(plab == c)[0]
        ci = np.where(elab == c)[0]
        idx, d = match_class(pub[pi], np.asarray(enc[ci]), a.tol)
        ok = idx >= 0
        file_id[pi[ok]], row[pi[ok]], dist[pi[ok]] = efile[ci[idx[ok]]], erow[ci[idx[ok]]], d[ok]
        used = ci[idx[ok]]
        report[int(c)] = {
            "published": int(len(pi)), "matched": int(ok.sum()), "exact": int((d[ok] == 0).sum()),
            "max_linf": float(d[ok].max()) if ok.any() else None,
            "duplicates": int(len(used) - len(np.unique(used))),
        }
        r = report[int(c)]
        print(f"  class {int(c):>2}: {r['matched']:>7,}/{r['published']:<7,} matched, {r['exact']:>7,} exact,"
              f" max |diff| {r['max_linf']:.2e}, duplicates {r['duplicates']}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, file_id=file_id, row=row, linf=dist)
    tot = {k: sum(v[k] for v in report.values()) for k in ("published", "matched", "exact", "duplicates")}
    tot["max_linf"] = max(v["max_linf"] for v in report.values() if v["max_linf"] is not None)
    Path(str(a.out) + ".json").write_text(json.dumps({"total": tot, "per_class": report}, indent=2))
    print(f"TOTAL: {tot['matched']:,}/{tot['published']:,} matched, {tot['exact']:,} exact, "
          f"max |diff| {tot['max_linf']:.2e}, duplicates {tot['duplicates']}")
    return 0 if tot["matched"] == tot["published"] and tot["duplicates"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
