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


def load_shards(d: Path, max_events: int = 0) -> np.ndarray:
    files = sorted(f for f in d.iterdir() if f.name.endswith("_x.npy"))
    if not files:
        raise FileNotFoundError(f"no *_x.npy under {d}")
    X = np.concatenate([np.load(f) for f in files], axis=0)
    return X[:max_events] if max_events else X


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

    # ── Signal block, built from the CASE trees ──────────────────────────────
    vec_dir = CASE_DATA / "vectorized" / "test" / args.case_label
    pre_dir = CASE_DATA / "preprocessed" / "test" / args.case_label
    X_raw = load_shards(vec_dir, args.max_signal)
    X_pp = load_shards(pre_dir, args.max_signal)
    if len(X_raw) != len(X_pp):
        print(f"LENGTH MISMATCH: vectorised {len(X_raw)} vs preprocessed {len(X_pp)}. "
              f"Physics and embeddings are paired by position, so they cannot be used.")
        return 1
    print(f"signal {args.case_label}: {len(X_raw):,} events "
          f"(raw dim {X_raw.shape[1]}, preprocessed dim {X_pp.shape[1]})")

    phys_sig = compute_physics(X_raw)

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
