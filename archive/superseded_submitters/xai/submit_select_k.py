#!/usr/bin/env python3
"""
Stub Condor submitter for step 01 (K selection).

Fill EOS paths / image, then:
    python3 scripts/xai/submit/submit_select_k.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
# Defaults mirror the VCReg d128 mocktest run used for the paper draft.
DEFAULTS = {
    "embeddings_dir": "/eos/user/d/dgenoves/anomaly_pipeline/gmm_encoder_seeds/vcreg_nosparse_dmodel128/encoder_seed_0/embeddings",
    "output_dir": "/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d128_seed0/01_select_k",
    "image": "/eos/user/d/dgenoves/fm_testing.sif",
}


def make_sub(dry_run: bool) -> None:
    log_dir = PROJECT_DIR / "logs/condor_logs/xai_paper/select_k"
    log_dir.mkdir(parents=True, exist_ok=True)
    wrapper = PROJECT_DIR / "scripts/xai/submit/wrapper_select_k.sh"
    sub = f"""universe = vanilla
executable = {wrapper}
arguments =
output = {log_dir}/select_k.out
error  = {log_dir}/select_k.err
log    = {log_dir}/select_k.log
+JobFlavour = "tomorrow"
queue 1
"""
    sub_path = log_dir / "select_k.sub"
    sub_path.write_text(sub)
    print(f"Wrote {sub_path}")
    if dry_run:
        print("[dry-run] not submitting")
        return
    import subprocess

    subprocess.run(["condor_submit", str(sub_path)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print("Defaults:", DEFAULTS)
    print("Edit wrapper_select_k.sh paths if needed.")
    make_sub(args.dry_run)


if __name__ == "__main__":
    main()
