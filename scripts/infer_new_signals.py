#!/usr/bin/env python3
"""
Anomaly-detection inference on additional signal proxies. Nothing is trained.

The encoder and the autoencoder already exist, the AE having been fitted on QCD
embeddings alone, and the operating threshold is the val-calibrated value stored
in the AE checkpoint. Evaluating a new process is therefore a forward pass:
encode, reconstruct, score, compare against a threshold that was fixed before the
process was ever seen. Re-running the training pipeline would refit the AE and
recalibrate the threshold, which is both wasteful and worse — the threshold would
no longer be the one the published results use.

Metric definitions are those of `COLLIDE2VAutoEncoderLitModule.on_test_epoch_end`,
reproduced here so numbers are comparable with `ad_results/*/result.json`:
  AUROC        signal vs QCD, scored by reconstruction MSE
  TPR @ FPR    fraction of signal above the val-calibrated threshold
  sep_ratio    mean signal MSE / mean QCD MSE
The measured FPR on this run's QCD is reported as a check that the transferred
threshold still lands where it should.

Usage:
    python3 scripts/infer_new_signals.py --dmodel 256 --seed 3
    python3 scripts/infer_new_signals.py --dmodel 256 --seed 3 --dry-run
"""

import argparse
import collections
import functools
import json
import sys
import typing
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import omegaconf
import torch

torch.serialization.add_safe_globals([
    functools.partial,
    torch.optim.AdamW, torch.optim.Adam,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    omegaconf.ListConfig, omegaconf.DictConfig, omegaconf.dictconfig.DictConfig,
    omegaconf.nodes.AnyNode, omegaconf.base.Metadata, omegaconf.base.ContainerMetadata,
    collections.defaultdict, typing.Any,
    list, dict, int,
])
torch.set_float32_matmul_precision("high")

from sklearn.metrics import roc_auc_score

# The XAI pipeline already scores embeddings against a trained AE and reads the
# val-calibrated thresholds out of its checkpoint; reuse that rather than
# reimplementing the forward pass, so both paths cannot drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent / "xai"))
from common.ae_score import compute_ae_mse, load_val_thresholds  # noqa: E402

from scripts.run_encoder_seeds_anomaly import find_best_ckpt, load_encoder
from src.train_full_anomaly_pipeline import extract_and_save_embeddings

NEW_EXP    = Path("/eos/user/d/dgenoves/anomaly_pipeline/new_exp")

# Held-out signal sets. Both are inference-only on processes normalised with the
# SM-only statistics; they differ in provenance. `newsig` are proxies drawn from the
# same COLLIDE-2V production as the training classes, `case` come from the separate
# CASE production and cover topologies the proxies never reach — soft resolved b-jets,
# a dimuon-rich hidden valley, high-multiplicity RPV cascades.
#
# `classes` must follow the order of `to_classify` in the matching experiment config:
# the labels saved in the embeddings are positions in that list.
DATASETS = {
    "newsig": {
        "experiment": "new_exp/anomaly_newsig_smnorm",
        "emb_root":   "newsig_embeddings",
        "out_root":   "ad_results_newsig",
        "classes": {0: "QCD_inclusive", 1: "VVV", 2: "VH", 3: "tttt", 4: "ttH",
                    5: "HH_bbtautau"},
    },
    "case": {
        "experiment": "new_exp/anomaly_case_smnorm",
        "emb_root":   "case_embeddings",
        "out_root":   "ad_results_case",
        "classes": {0: "QCD_inclusive", 1: "hToAA_4b_ma60", 2: "hToAA_4tau_ma15",
                    3: "HVdilep_Zp1000_piD2_mumu", 4: "RPV_squark300_UDD",
                    5: "RPV_squark600_cascade_LSP550",
                    6: "SUEPlike_HV_mPhi400_mX2_Lam4", 7: "Zprime_qq_m500"},
    },
}

# Seed sets differ by model: the VCReg runs are indexed 0-4 while the contrastive
# runs carry the seed value itself, so the directory names are not interchangeable.
MODELS = {
    "vcreg": {
        "class": "src.models.collide2v_vicreg.COLLIDE2VVCRegLitModule",
        "run":   "vcreg_12class_nosparse_dmodel{d}_cern",
        "seeds": [0, 1, 2, 3, 4],
    },
    "supcon": {
        "class": "src.models.collide2v_augmented_supcon.COLLIDE2VAugmentedSupConLitModule",
        "run":   "supcon_12class_nosparse_dmodel{d}_cern",
        "seeds": [7, 42, 137, 1337, 31337],
    },
    # VICReg row of the paper. Same encoders the vicreg AD reports, so the
    # extra held-out signals land in the same table rows as HH->4b, VBF H->bb and
    # ggH->tautau rather than describing a different training campaign.
    "vicreg": {
        "class": "src.models.collide2v_vicreg.COLLIDE2VVICRegLitModule",
        "run":   "vicreg_12class_nosparse_dmodel{d}_cern",
        "seeds": [7, 42, 12345, 1337, 31337],
    },
    # Self-supervised SimCLR — "SimCLR" in the paper, "simclr" on disk.
    "simclr": {
        "class": "src.models.collide2v_augmented_selfsupcon.COLLIDE2VAugmentedSelfSupConLitModule",
        "run":   "simclr_12class_nosparse_dmodel{d}_cern",
        "seeds": [7, 42, 137, 1337, 31337],
    },
}


def find_ae_ckpt(ad_dir: Path) -> Path | None:
    """The AE checkpoint keeps the epoch it stopped at, which is not always 49 —
    d32 has runs ending at 44, 45 and 48. Glob for it rather than assuming."""
    ck_dir = ad_dir / "mse_normal" / "checkpoints"
    cands = sorted(p for p in ck_dir.glob("ae-epoch*.ckpt") if p.name != "last.ckpt")
    if not cands:
        return None
    # More than one only if a run was resumed; the highest epoch is the best one.
    return max(cands, key=lambda p: p.stat().st_mtime)

NORMAL_LABEL = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vcreg", choices=list(MODELS))
    ap.add_argument("--dmodel", type=int, default=256)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--dataset", default="newsig", choices=list(DATASETS),
                    help="Which held-out signal set to score")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    spec = MODELS[a.model]
    ds = DATASETS[a.dataset]
    EXPERIMENT, CLASS_NAMES = ds["experiment"], ds["classes"]
    if a.seed not in spec["seeds"]:
        print(f"Seed {a.seed} is not one of the {a.model} seeds {spec['seeds']}")
        return 1

    run = spec["run"].format(d=a.dmodel)
    seed_dir = NEW_EXP / run / f"seed_{a.seed}"
    ad_dir = NEW_EXP / "ad_results" / run / f"encoder_seed_{a.seed}"
    ae_ckpt = find_ae_ckpt(ad_dir)
    emb_dir = NEW_EXP / ds["emb_root"] / run / f"encoder_seed_{a.seed}" / "embeddings"
    out_dir = NEW_EXP / ds["out_root"] / run / f"encoder_seed_{a.seed}"

    enc_ckpt = find_best_ckpt(seed_dir)
    print(f"dataset     : {a.dataset}  ({EXPERIMENT})")
    print(f"model       : {a.model}  ({spec['class'].rsplit('.', 1)[-1]})")
    print(f"d_model     : {a.dmodel}   seed: {a.seed}")
    print(f"encoder     : {enc_ckpt}")
    print(f"AE          : {ae_ckpt}")
    print(f"embeddings  : {emb_dir}")
    print(f"output      : {out_dir}")

    if ae_ckpt is None:
        print(f"MISSING: no ae-epoch*.ckpt under {ad_dir/'mse_normal'/'checkpoints'}")
        return 1
    if not Path(enc_ckpt).exists():
        print(f"MISSING: {enc_ckpt}")
        return 1
    if a.dry_run:
        print("\n[dry-run] stopping before inference")
        return 0

    if not (emb_dir / "test_embeddings.npz").exists():
        emb_dir.mkdir(parents=True, exist_ok=True)
        encoder = load_encoder(enc_ckpt, spec["class"])
        extract_and_save_embeddings(model=encoder, phase2_experiment=EXPERIMENT, output_dir=emb_dir)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\nEmbeddings already present — reusing them.")

    d = np.load(emb_dir / "test_embeddings.npz")
    Z, y = d["embeddings"], d["labels"]
    print(f"\ntest embeddings: {Z.shape}")
    mse = compute_ae_mse(ae_ckpt, Z)
    thresholds = {float(k): float(v) for k, v in load_val_thresholds(ae_ckpt).items()}
    if not thresholds:
        print("No val_thresholds in the AE checkpoint — cannot transfer the operating point.")
        return 1
    print(f"val-calibrated thresholds: { {k: round(v, 4) for k, v in thresholds.items()} }")

    qcd = mse[y == NORMAL_LABEL]
    if len(qcd) == 0:
        print("No QCD events in the test split — cannot evaluate.")
        return 1
    qcd_mean = float(np.mean(qcd))

    metrics = {
        f"mse_mean_cls{NORMAL_LABEL}": qcd_mean,
        f"mse_std_cls{NORMAL_LABEL}": float(np.std(qcd)),
        f"n_cls{NORMAL_LABEL}": int(len(qcd)),
    }
    for fpr, thr in thresholds.items():
        tag = f"fpr{int(fpr*100):02d}"
        metrics[f"threshold_{tag}"] = thr
        metrics[f"fpr_measured_{tag}"] = float(np.mean(qcd > thr))

    print(f"\n{'process':14s} {'n':>7s} {'AUROC':>7s} " +
          "  ".join(f"TPR@{int(f*100)}%" for f in sorted(thresholds)) + f" {'sep':>6s}")
    rows = []
    for cls in sorted(set(y.tolist())):
        if cls == NORMAL_LABEL:
            continue
        sig = mse[y == cls]
        if len(sig) == 0:
            continue
        tag = f"cls{cls}"
        metrics[f"mse_mean_{tag}"] = float(np.mean(sig))
        metrics[f"mse_std_{tag}"] = float(np.std(sig))
        metrics[f"n_{tag}"] = int(len(sig))
        auroc = float(roc_auc_score(np.r_[np.zeros(len(qcd)), np.ones(len(sig))], np.r_[qcd, sig]))
        metrics[f"auroc_{tag}"] = auroc
        sep = float(np.mean(sig) / qcd_mean) if qcd_mean > 0 else float("nan")
        metrics[f"sep_ratio_{tag}"] = sep
        tprs = []
        for fpr, thr in sorted(thresholds.items()):
            t = float(np.mean(sig > thr))
            metrics[f"tpr_fpr{int(fpr*100):02d}_{tag}"] = t
            metrics[f"fnr_fpr{int(fpr*100):02d}_{tag}"] = 1.0 - t
            tprs.append(t)
        name = CLASS_NAMES.get(int(cls), str(cls))
        print(f"{name:14s} {len(sig):7,d} {auroc:7.3f} " +
              "  ".join(f"{t:8.3f}" for t in tprs) + f" {sep:6.2f}")
        rows.append({"label": int(cls), "process": name, "n": int(len(sig)),
                     "auroc": auroc, "sep_ratio": sep,
                     "tpr": {f"{f:.2f}": metrics[f'tpr_fpr{int(f*100):02d}_{tag}']
                             for f in sorted(thresholds)}})

    print(f"\nQCD: n={len(qcd):,}  mean MSE={qcd_mean:.4f}  measured FPR " +
          "  ".join(f"@{int(f*100)}%={metrics[f'fpr_measured_fpr{int(f*100):02d}']:.4f}"
                    for f in sorted(thresholds)))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"result_{a.dataset}.json").write_text(json.dumps({
        "model": a.model, "model_class": spec["class"],
        "d_model": a.dmodel, "encoder_seed": a.seed,
        "encoder_ckpt": str(enc_ckpt), "ae_ckpt": str(ae_ckpt),
        "experiment": EXPERIMENT, "inference_only": True,
        "class_names": CLASS_NAMES, "summary": rows, "per_signal": metrics,
    }, indent=2))
    print(f"\nSaved {out_dir}/result_{a.dataset}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
