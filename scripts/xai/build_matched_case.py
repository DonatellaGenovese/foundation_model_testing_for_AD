#!/usr/bin/env python3
"""
Build a matched (embeddings, labels, physics) array for a CASE signal, on the SAME
SM background the HH->4b analysis uses.

WHY THE BACKGROUND IS REUSED RATHER THAN REBUILT. The interpretability pipeline
compares flagged events against the SM that occupies the same GMM component. For
HH->4b that local SM is the twelve SM classes of the smnorm dataset. The CASE dataset
contains only QCD as Standard Model — its other seven processes are signals — so
building the background from CASE would compare this signal against QCD alone while
HH->4b is compared against all twelve. The two interpretations would then differ in
their background as well as their signal, and the claim the second signal exists to
support ("the leading observable tracks the physics of the signal") would be
ambiguous: the answer could have moved because the reference moved.

So the SM rows are taken verbatim from the existing matched file and only the signal
block is built here. Both come from the same encoder, so the embeddings live in the
same space; both datasets carry the same SM-only normalisation.

ALIGNMENT. Physics come from the raw vectorised tree and embeddings from the
preprocessed one, paired BY POSITION — the same trap documented in
common/physics.build_matched_arrays. Here both trees hold one file of 4,984 events
for the signal, read in sorted order, so the pairing is exact; the script checks the
two lengths agree and refuses to continue otherwise.

Usage:
    python scripts/xai/build_matched_case.py \\
        --case-label HVdilep_Zp1000_piD2_mumu --signal-label 20 \\
        --sm-matched /eos/.../04_profile/matched_sm_hh4b.npz \\
        --ckpt /eos/.../vcreg_12class_nosparse_dmodel256_cern/seed_3/checkpoints/... \\
        --output /eos/.../matched_sm_hvdilep.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_XAI = Path(__file__).resolve().parent
_ROOT = _XAI.parents[1]
sys.path.insert(0, str(_XAI))
sys.path.insert(0, str(_ROOT))

import numpy as np

from common.constants import PHYSICS_VARS, SM_INDICES
from common.physics import compute_physics, load_matched_npz, save_matched_npz

CASE_DATA = Path("/eos/user/d/dgenoves/foundation_model_testing_data/"
                 "v2_nosparse_case_smnorm_highlevel")
# Symlink tree the CASE config reads from, one entry per process pointing into the b2g
# production. Used here to reach the reconstructed object lists, which the vectorised
# arrays no longer carry once they have been truncated to the per-group top-k.
CASE_SRC = Path("/eos/user/d/dgenoves/foundation_model_testing_data/_case_src")


def load_shards(d: Path, max_events: int = 0) -> np.ndarray:
    files = sorted(f for f in d.iterdir() if f.name.endswith("_x.npy"))
    if not files:
        raise FileNotFoundError(f"no *_x.npy under {d}")
    X = np.concatenate([np.load(f) for f in files], axis=0)
    return X[:max_events] if max_events else X


def true_lepton_count(case_label: str, n_expect: int,
                      case_src: Path = CASE_SRC) -> np.ndarray | None:
    """Lepton multiplicity read from the CASE parquet, before the top-k truncation.

    compute_physics() counts leptons off the padded arrays, which keep at most eight
    muons and eight electrons. For every Standard Model process and for the proxy
    signals that cap is never reached --- their maxima are six and seven --- so the two
    counts agree. The dimuon signal is the exception: 38% of its events have more than
    eight muons, the true multiplicity runs to 23, and counting off the padded array
    turns a smooth distribution into a 43% spike sitting exactly on the cap. That spike
    is a property of the input pipeline, not of the events, so the observable is read
    from the reconstructed lists instead and the truncation is stated in the text as a
    limit of what the encoder sees.

    Events with no reconstructed objects at all are dropped upstream, so they are
    removed here too; the caller checks the resulting length against the embeddings.
    Returns None when the parquet cannot be located, leaving the padded count in place.
    """
    folder = case_src / case_label
    files = sorted(folder.glob("*.parquet"))
    if not files:
        print(f"  no parquet under {folder}; keeping the truncated lepton count")
        return None

    import pyarrow.parquet as pq

    cols = ["FullReco_MuonTight_PT", "FullReco_Electron_PT",
            "FullReco_JetPuppiAK4_PT", "FullReco_PhotonTight_PT"]
    counts = []
    for f in files:
        tab = pq.read_table(f, columns=cols)
        per_col = [np.array([len(x) if x is not None else 0 for x in tab.column(c).to_pylist()])
                   for c in cols]
        counts.append(np.stack(per_col, axis=1))
    n = np.concatenate(counts, axis=0)          # columns: mu, e, jet, gamma
    keep = n.sum(axis=1) > 0
    true_n = (n[:, 0] + n[:, 1])[keep]

    if len(true_n) != n_expect:
        print(f"  parquet gives {len(true_n):,} non-empty events against {n_expect:,} "
              f"embeddings; keeping the truncated lepton count")
        return None
    return true_n.astype(float)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-label", required=True, help="CASE folder name of the signal")
    p.add_argument("--signal-label", type=int, required=True,
                   help="Label to give the signal; must not collide with 0-14")
    p.add_argument("--sm-matched", type=Path, required=True,
                   help="Existing matched npz whose SM rows are reused")
    p.add_argument("--ckpt", type=Path, required=True, help="Encoder checkpoint")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-signal", type=int, default=0, help="0 = all available")
    p.add_argument("--case-data", type=Path, default=CASE_DATA,
                   help="CASE dataset tree with vectorized/ and preprocessed/ (stage 1)")
    p.add_argument("--case-src", type=Path, default=CASE_SRC,
                   help="Per-process folders of CASE parquet, for the untruncated "
                        "lepton count")
    args = p.parse_args()

    if args.signal_label <= 14:
        print(f"--signal-label {args.signal_label} collides with the 0-14 range used "
              f"by the SM classes and the three proxy signals")
        return 1

    # ── SM background, taken verbatim ────────────────────────────────────────
    Z_sm, y_sm, phys_sm = load_matched_npz(args.sm_matched)
    keep = np.isin(y_sm, SM_INDICES)
    Z_sm, y_sm = Z_sm[keep], y_sm[keep]
    phys_sm = {v: phys_sm[v][keep] for v in PHYSICS_VARS}
    print(f"SM background reused from {args.sm_matched.name}: {len(y_sm):,} events, "
          f"{len(set(y_sm.tolist()))} classes")

    # The Standard Model block is copied verbatim from another matched file rather than
    # recomputed, so a change to compute_physics does not reach it: the file has to be
    # rebuilt too. That is not hypothetical --- the b-tag definition was corrected in
    # August and the fix took a month to arrive here, during which the signal block was
    # counted at the medium working point while the Standard Model it is compared against
    # was still counted as an OR over all six, at a 24% light-jet mistag. Nothing failed;
    # the two halves of every comparison simply meant different things. So: recompute one
    # observable on a sample of the inherited block and check it against what is stored.
    # Measured on this twelve-class, class-balanced Standard Model sample: the medium
    # working point gives a mean of 0.52 b-tags per event, the OR over all six bits 1.46.
    # 1.0 sits between them with room on either side.
    sm_nb = float(np.nanmean(phys_sm["n_bjets"]))
    print(f"  SM block b-tag check: mean n_bjets {sm_nb:.3f} (medium WP gives ~0.52)")
    if sm_nb > 1.0:
        print("  WARNING: the inherited Standard Model block has n_bjets around the value "
              "the OR over all six b-tag bits produces (~1.46), not the medium working "
              "point (~0.52). It was probably built before the b-tag fix. Rebuild "
              "--sm-matched first, or the signal and the background it is compared "
              "against will be counted differently.")

    # ── Signal block, built from the CASE trees ──────────────────────────────
    vec_dir = args.case_data / "vectorized" / "test" / args.case_label
    pre_dir = args.case_data / "preprocessed" / "test" / args.case_label
    X_raw = load_shards(vec_dir, args.max_signal)
    X_pp = load_shards(pre_dir, args.max_signal)
    if len(X_raw) != len(X_pp):
        print(f"LENGTH MISMATCH: vectorised {len(X_raw)} vs preprocessed {len(X_pp)}. "
              f"Physics and embeddings are paired by position, so they cannot be used.")
        return 1
    print(f"signal {args.case_label}: {len(X_raw):,} events "
          f"(raw dim {X_raw.shape[1]}, preprocessed dim {X_pp.shape[1]})")

    phys_sig = compute_physics(X_raw)

    true_nlep = true_lepton_count(args.case_label, len(X_raw), args.case_src)
    if true_nlep is not None:
        trunc = phys_sig["n_leptons"]
        n_capped = int((trunc >= 8).sum())
        phys_sig["n_leptons"] = true_nlep
        print(f"n_leptons read from the parquet: median {np.median(true_nlep):.0f}, "
              f"max {true_nlep.max():.0f}; {n_capped:,} events ({n_capped/len(trunc)*100:.1f}%) "
              f"were sitting on the eight-muon cap")

    import torch
    sys.path.insert(0, str(_XAI))
    from importlib import import_module
    step04 = import_module("04_profile_and_rank")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = step04.load_encoder(args.ckpt, device)
    encode = step04.make_encode_fn(encoder, device)
    print(f"encoding on {device}…", flush=True)
    Z_sig = encode(X_pp)
    if Z_sig.shape[1] != Z_sm.shape[1]:
        print(f"EMBEDDING WIDTH MISMATCH: signal {Z_sig.shape[1]} vs SM {Z_sm.shape[1]} "
              f"— the SM block came from a different encoder")
        return 1

    # ── Concatenate and save ─────────────────────────────────────────────────
    Z = np.concatenate([Z_sm, Z_sig], axis=0)
    y = np.concatenate([y_sm, np.full(len(Z_sig), args.signal_label, dtype=int)])
    phys = {v: np.concatenate([phys_sm[v], phys_sig[v]]) for v in PHYSICS_VARS}

    save_matched_npz(args.output, Z, y, phys)
    print(f"\n{len(y):,} events total  ({len(y_sm):,} SM + {len(Z_sig):,} signal)")
    for v in PHYSICS_VARS:
        s = phys_sig[v]; s = s[np.isfinite(s)]
        b = phys_sm[v];  b = b[np.isfinite(b)]
        print(f"   {v:<10} signal median {np.median(s):>8.2f}   SM median {np.median(b):>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
