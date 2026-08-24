#!/usr/bin/env python3
"""
Stub Condor submitter for the full paper XAI chain (02→04).

    python3 scripts/xai/submit/submit_xai_pipeline.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULTS = {
    "embeddings_dir": "/eos/user/d/dgenoves/anomaly_pipeline/gmm_encoder_seeds/vcreg_nosparse_dmodel128/encoder_seed_0/embeddings",
    "ae_ckpt": "/eos/user/d/dgenoves/anomaly_pipeline/ad_encoder_seeds/vcreg_nosparse_dmodel128/encoder_seed_0/mse_normal/checkpoints/ae-epochepoch=49.ckpt",
    "output_root": "/eos/user/d/dgenoves/anomaly_pipeline/xai_paper/vcreg_d128_seed0",
    "k": 12,
}


def make_sub(dry_run: bool) -> None:
    log_dir = PROJECT_DIR / "logs/condor_logs/xai_paper/pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    wrapper = PROJECT_DIR / "scripts/xai/submit/wrapper_xai_pipeline.sh"
    sub = f"""universe = vanilla
executable = {wrapper}
arguments =
output = {log_dir}/pipeline.out
error  = {log_dir}/pipeline.err
log    = {log_dir}/pipeline.log
+JobFlavour = "tomorrow"
request_gpus = 1
queue 1
"""
    sub_path = log_dir / "pipeline.sub"
    sub_path.write_text(sub)
    print(f"Wrote {sub_path}")
    print("Defaults:", DEFAULTS)
    if dry_run:
        print("[dry-run] not submitting")
        return
    import subprocess

    subprocess.run(["condor_submit", str(sub_path)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    make_sub(args.dry_run)


if __name__ == "__main__":
    main()
