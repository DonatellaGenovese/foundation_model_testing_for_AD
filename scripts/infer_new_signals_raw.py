#!/usr/bin/env python3
"""
Raw-AE baseline on the additional signal proxies. Inference only.

The counterpart of `infer_new_signals.py` for the baseline that skips the encoder
entirely: the autoencoder is trained directly on the preprocessed kinematic
features, so evaluating a new process needs no embedding step at all — load the
features, reconstruct, score against the val-calibrated threshold.

Strategy. The raw baseline was run under six strategies; the one reported in the
paper is `mse_qcd`, the autoencoder trained on QCD alone with an MSE monitor,
which is the direct analogue of `mse_normal` in the embedding pipeline. It is
also the one that reproduces the published numbers exactly (HH->4b AUROC
0.931 +- 0.001, TPR 77.3 +- 0.7), which is how it was identified.

Both datasets carry 340 features normalised with the same SM-only statistics
(531,750 fit events), so the trained baseline transfers without rescaling. The
script checks the feature width against the checkpoint and refuses to run on a
mismatch rather than silently reconstructing the wrong thing.

Usage:
    python3 scripts/infer_new_signals_raw.py --seed 1337
    python3 scripts/infer_new_signals_raw.py --seed 1337 --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "xai"))
from common.ae_score import compute_ae_mse, load_val_thresholds  # noqa: E402

from src.train_full_anomaly_pipeline import _compose_cfg

NEW_EXP    = Path("/eos/user/d/dgenoves/anomaly_pipeline/new_exp")
STRATEGY   = "mse_qcd"
SEEDS      = [7, 42, 137, 1337, 31337]

# Same two held-out sets as infer_new_signals.py, so the raw baseline row is built on
# exactly the events the learned-embedding rows are scored on. `classes` follows the
# order of `to_classify` in the matching experiment config.
DATASETS = {
    "newsig": {
        "experiment": "new_exp/anomaly_newsig_smnorm",
        "out_root":   "ad_results_newsig",
        "classes": {0: "QCD_inclusive", 1: "VVV", 2: "VH", 3: "tttt", 4: "ttH",
                    5: "HH_bbtautau"},
    },
    "case": {
        "experiment": "new_exp/anomaly_case_smnorm",
        "out_root":   "ad_results_case",
        "classes": {0: "QCD_inclusive", 1: "hToAA_4b_ma60", 2: "hToAA_4tau_ma15",
                    3: "HVdilep_Zp1000_piD2_mumu", 4: "RPV_squark300_UDD",
                    5: "RPV_squark600_cascade_LSP550",
                    6: "SUEPlike_HV_mPhi400_mX2_Lam4", 7: "Zprime_qq_m500"},
    },
}
NORMAL_LABEL = 0


def find_ckpt(seed: int) -> Path | None:
    d = NEW_EXP / "ad_results" / "raw" / f"seed_{seed}" / "raw_baseline" / STRATEGY / "checkpoints"
    cands = [p for p in d.glob("ae-epoch*.ckpt") if p.name != "last.ckpt"]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def load_test_features(out_dir: Path, experiment: str) -> tuple[np.ndarray, np.ndarray]:
    """Preprocessed feature vectors and labels for the test split, every class."""
    cfg = _compose_cfg("anomaly_detection.yaml", [f"experiment={experiment}"], output_dir=out_dir)
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup("test")
    xs, ys = [], []
    for batch in dm.test_dataloader():
        x, y = batch[0], batch[1]
        xs.append(x.reshape(len(x), -1).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337, choices=SEEDS)
    ap.add_argument("--dataset", default="newsig", choices=list(DATASETS),
                    help="Which held-out signal set to score")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    EXPERIMENT, CLASS_NAMES = ds["experiment"], ds["classes"]
    ckpt = find_ckpt(a.seed)
    out_dir = NEW_EXP / ds["out_root"] / "raw" / f"seed_{a.seed}"
    print(f"dataset  : {a.dataset}  ({EXPERIMENT})")
    print(f"strategy : {STRATEGY}   seed: {a.seed}")
    print(f"AE       : {ckpt}")
    print(f"output   : {out_dir}")
    if ckpt is None:
        print("MISSING: no ae-epoch*.ckpt for this seed")
        return 1
    if a.dry_run:
        print("\n[dry-run] stopping before inference")
        return 0

    X, y = load_test_features(out_dir, EXPERIMENT)
    print(f"\ntest features: {X.shape}")

    expected = torch.load(ckpt, map_location="cpu", weights_only=False) \
        .get("hyper_parameters", {}).get("input_dim")
    if expected is not None and X.shape[1] != expected:
        print(f"FEATURE WIDTH MISMATCH: data has {X.shape[1]}, checkpoint expects {expected}")
        return 1

    mse = compute_ae_mse(ckpt, X.astype(np.float32))
    thresholds = {float(k): float(v) for k, v in load_val_thresholds(ckpt).items()}
    if not thresholds:
        print("No val_thresholds in the checkpoint — cannot transfer the operating point.")
        return 1
    print(f"val-calibrated thresholds: { {k: round(v, 5) for k, v in thresholds.items()} }")

    qcd = mse[y == NORMAL_LABEL]
    if len(qcd) == 0:
        print("No QCD events in the test split.")
        return 1
    qcd_mean = float(np.mean(qcd))

    metrics = {f"mse_mean_cls{NORMAL_LABEL}": qcd_mean,
               f"mse_std_cls{NORMAL_LABEL}": float(np.std(qcd)),
               f"n_cls{NORMAL_LABEL}": int(len(qcd))}
    for fpr, thr in thresholds.items():
        tag = f"fpr{int(fpr*100):02d}"
        metrics[f"threshold_{tag}"] = thr
        metrics[f"fpr_measured_{tag}"] = float(np.mean(qcd > thr))

    print(f"\n{'process':14s} {'n':>7s} {'AUROC':>7s} " +
          "  ".join(f"TPR@{int(f*100)}%" for f in sorted(thresholds)))
    rows = []
    for cls in sorted(set(y.tolist())):
        if cls == NORMAL_LABEL:
            continue
        sig = mse[y == cls]
        if len(sig) == 0:
            continue
        tag = f"cls{cls}"
        auroc = float(roc_auc_score(np.r_[np.zeros(len(qcd)), np.ones(len(sig))], np.r_[qcd, sig]))
        sep = float(np.mean(sig) / qcd_mean) if qcd_mean > 0 else float("nan")
        metrics[f"auroc_{tag}"] = auroc
        metrics[f"sep_ratio_{tag}"] = sep
        metrics[f"n_{tag}"] = int(len(sig))
        tprs = []
        for fpr, thr in sorted(thresholds.items()):
            t = float(np.mean(sig > thr))
            metrics[f"tpr_fpr{int(fpr*100):02d}_{tag}"] = t
            tprs.append(t)
        name = CLASS_NAMES.get(int(cls), str(cls))
        print(f"{name:14s} {len(sig):7,d} {auroc:7.3f} " + "  ".join(f"{t:8.3f}" for t in tprs))
        rows.append({"label": int(cls), "process": name, "n": int(len(sig)),
                     "auroc": auroc, "sep_ratio": sep,
                     "tpr": {f"{f:.2f}": metrics[f'tpr_fpr{int(f*100):02d}_{tag}']
                             for f in sorted(thresholds)}})

    print(f"\nQCD: n={len(qcd):,}  measured FPR " +
          "  ".join(f"@{int(f*100)}%={metrics[f'fpr_measured_fpr{int(f*100):02d}']:.4f}"
                    for f in sorted(thresholds)))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"result_{a.dataset}.json").write_text(json.dumps({
        "model": "raw", "strategy": STRATEGY, "seed": a.seed, "ae_ckpt": str(ckpt),
        "experiment": EXPERIMENT, "inference_only": True,
        "class_names": CLASS_NAMES, "summary": rows, "per_signal": metrics,
    }, indent=2))
    print(f"\nSaved {out_dir}/result_{a.dataset}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
