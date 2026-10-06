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
# order of `to_classify` in the matching experiment config; see the note there on why
# `newsig` writes to its own v3 root.
DATASETS = {
    "newsig": {
        "experiment": "new_exp/anomaly_newsig_smnorm",
        "out_root":   "ad_results_newsig_v3",
        "classes": {0: "QCD_inclusive", 1: "HH_bbtautau"},
    },
    "case": {
        "experiment": "new_exp/anomaly_case_smnorm",
        "out_root":   "ad_results_case",
        "classes": {0: "QCD_inclusive", 1: "hToAA_4b_ma60", 2: "hToAA_4tau_ma15",
                    3: "HVdilep_Zp1000_piD2_mumu"},
    },
    # The same CASE processes from the collide_v2 production (50,000 events per signal),
    # the paper's held-out table, with the same selections as infer_new_signals.py.
    "case_v2": {
        "experiment": "new_exp/anomaly_case_v2_smnorm",
        "out_root":   "ad_results_case_v2",
        "classes": {0: "QCD_inclusive", 1: "hToAA_4b_ma60", 2: "hToAA_4tau_ma15",
                    3: "HVdilep_Zp1000_piD2_mumu"},
        # The events the paper scores: 20,000 per signal drawn by scripts/select_case_events.py
        # (seed 42), the statistics of the proxy signals, and the QCD reference HH->4b was
        # scored against. Passing --select or --qcd-reference overrides them.
        "selections": {1: "/eos/user/d/dgenoves/foundation_model_testing_data/v3_nosparse_case_smnorm_highlevel/selections/hToAA_4b_ma60_random.npz",
                       2: "/eos/user/d/dgenoves/foundation_model_testing_data/v3_nosparse_case_smnorm_highlevel/selections/hToAA_4tau_ma15_random.npz",
                       3: "/eos/user/d/dgenoves/foundation_model_testing_data/v3_nosparse_case_smnorm_highlevel/selections/HVdilep_Zp1000_piD2_mumu_random.npz"},
        "qcd_reference": "proxy",
        "out_sub": "random",
    },
}
# The experiment the published raw baseline was trained and evaluated with; its test QCD
# is the reference the raw HH->4b row was scored against (RawNpyDataModule, seed 42).
RAW_PROXY_EXPERIMENT = "anomaly_qcd_vs_higgs_raw_smnorm_nosparse_cern"
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


def load_proxy_qcd(out_dir: Path) -> np.ndarray:
    """The QCD test events the raw HH->4b row was scored against, rebuilt the way
    run_raw_ae_baseline builds them: RawNpyDataModule over the same tree and split size,
    with its default seed."""
    from src.data.raw_npy_datamodule import RawNpyDataModule
    cfg = _compose_cfg("anomaly_detection.yaml", [f"experiment={RAW_PROXY_EXPERIMENT}"], output_dir=out_dir)
    names = list(cfg.data.to_classify)
    p2f = dict(cfg.data.get("process_to_folder", {}))
    dm = RawNpyDataModule(
        preprocessed_dir=Path(cfg.paths.eos_data_dir) / cfg.data.label / "preprocessed",
        normal_classes=list(cfg.normal_classes), anomaly_classes=[],
        class_folders={i: p2f.get(n, n) for i, n in enumerate(names)},
        n_train=1, n_val=1, n_test=list(cfg.data.train_val_test_split_per_class)[2],
        batch_size=cfg.data.batch_size)
    dm.setup("test")
    x, y = dm.test_ds.tensors
    return x[y == NORMAL_LABEL].numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337, choices=SEEDS)
    ap.add_argument("--dataset", default="newsig", choices=list(DATASETS),
                    help="Which held-out signal set to score")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--select", action="append", default=[], metavar="LABEL=FILE",
                    help="Score only the events an npz from scripts/select_case_events.py lists, "
                         "for the class with that label; repeatable")
    ap.add_argument("--qcd-reference", choices=["dataset", "proxy"], default=None,
                    help="'proxy': score against the QCD test events of the published raw HH->4b row")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    EXPERIMENT, CLASS_NAMES = ds["experiment"], ds["classes"]
    if not a.select:
        a.select = [f"{lbl}={path}" for lbl, path in ds.get("selections", {}).items()]
    a.qcd_reference = a.qcd_reference or ds.get("qcd_reference", "dataset")
    ckpt = find_ckpt(a.seed)
    out_dir = a.out_dir or (NEW_EXP / ds["out_root"] / "raw" / f"seed_{a.seed}" / ds.get("out_sub", ""))
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
    selections = {}
    if a.select:
        keep = np.ones(len(y), bool)
        for sel in a.select:
            lbl, path = sel.split("=", 1)
            lbl = int(lbl)
            idx = np.load(path)["index"].astype(int)
            pos = np.flatnonzero(y == lbl)
            if len(pos) == 0 or idx.max() >= len(pos):
                print(f"--select {sel}: {len(pos):,} events of label {lbl}, index up to {idx.max()}")
                return 1
            mask = np.zeros(len(pos), bool)
            mask[idx] = True
            keep[pos[~mask]] = False
            selections[lbl] = str(path)
            print(f"  label {lbl}: {len(idx):,} of {len(pos):,} events selected")
        X, y = X[keep], y[keep]
    if a.qcd_reference == "proxy":
        q = load_proxy_qcd(out_dir)
        print(f"QCD reference: {len(q):,} events of {RAW_PROXY_EXPERIMENT} "
              f"(replacing {int((y == NORMAL_LABEL).sum()):,} of the dataset's own)")
        X = np.concatenate([X[y != NORMAL_LABEL], q.astype(X.dtype)])
        y = np.concatenate([y[y != NORMAL_LABEL], np.full(len(q), NORMAL_LABEL, dtype=y.dtype)])

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
    # Declared classes only, as in infer_new_signals.py. The datamodule already reads
    # just the declared folders, so here this is a guard rather than a filter.
    undeclared = sorted(set(y.tolist()) - set(CLASS_NAMES))
    if undeclared:
        print(f"Ignoring labels {undeclared}: not declared for '{a.dataset}'.")
    rows = []
    for cls in sorted(CLASS_NAMES):
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
        "selections": selections, "qcd_reference": a.qcd_reference,
    }, indent=2))
    print(f"\nSaved {out_dir}/result_{a.dataset}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
