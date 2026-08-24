"""
Run GMM anomaly detection across all encoder seeds.

For each encoder seed N:
  1. Load encoder from encoder_seeds_dir/seed_N/checkpoints/best (epoch_XXX.ckpt)
  2. Extract embeddings for all 12 SM + 3 BSM classes
     → output_dir/encoder_seed_N/embeddings/
  3. For each K in K_VALUES, fit a GMM on train SM embeddings (diag covariance)
     → output_dir/encoder_seed_N/K{k}/gmm.npz + results.json
  4. Compute two anomaly scores on test set:
       score1 = NLL  = -gmm.score_samples(z)
       score3 = 1 - max_k r_k(z)   (1 minus max responsibility)
  5. Compute AUROC for each BSM class with two SM backgrounds:
       qcd   — QCD only (label 0) as normal reference
       allSM — all 12 SM classes as normal reference
  6. Save summary to output_dir/encoder_seed_N/summary.json (all K results)

Aggregates mean ± std across encoder seeds per K → output_dir/aggregated_summary.json

Supports resume: per-K results.json already present are skipped.

Usage:
    python scripts/run_gmm_encoder_seeds.py \\
        --phase2-experiment anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_allsm_dmodel128_cern \\
        --encoder-seeds-dir /eos/user/d/dgenoves/anomaly_pipeline/encoder_seeds/aug_supcon_12class_nosparse_dmodel128_cern \\
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/gmm_encoder_seeds/augsupcon_nosparse_dmodel128
"""

import argparse
import collections
import functools
import gc
import json
import math
import os
import typing
from importlib import import_module
from pathlib import Path

import numpy as np
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import omegaconf

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
torch.set_float32_matmul_precision('high')

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture

from src.train_full_anomaly_pipeline import extract_and_save_embeddings, _compose_cfg
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

FPR_TARGETS = [0.01, 0.05, 0.10]
K_VALUES    = [6, 10, 12, 15]
COV_TYPE    = "diag"


# ─── Encoder loading ─────────────────────────────────────────────────────────

def find_best_ckpt(seed_dir: Path) -> Path:
    """Return the best checkpoint (epoch_XXX.ckpt saved by ModelCheckpoint),
    falling back to last.ckpt only if no epoch checkpoint exists."""
    ckpt_dir = seed_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints dir: {ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt"))
    if ckpts:
        return ckpts[-1]   # ModelCheckpoint saves only the best (save_top_k=1)
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def load_encoder(ckpt_path: Path, model_class_str: str) -> torch.nn.Module:
    module_path, class_name = model_class_str.rsplit(".", 1)
    model_cls = getattr(import_module(module_path), class_name)
    model = model_cls.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    return model


# ─── GMM fitting ─────────────────────────────────────────────────────────────

def fit_gmm_k(z_train: np.ndarray, k: int, seed: int) -> GaussianMixture:
    """Fit a single GMM with K components (diag covariance)."""
    g = GaussianMixture(
        n_components=k, covariance_type=COV_TYPE,
        random_state=seed, max_iter=300, n_init=3, reg_covar=1e-6,
    )
    g.fit(z_train)
    return g


# ─── Scoring & AUROC ─────────────────────────────────────────────────────────

def compute_scores(gmm: GaussianMixture, z: np.ndarray):
    """Return score1 (NLL) and score3 (1 - maxResp) for each sample."""
    score1 = -gmm.score_samples(z).astype(np.float64)
    resp   = gmm.predict_proba(z)
    score3 = (1.0 - resp.max(axis=1)).astype(np.float64)
    return score1, score3


def evaluate_auroc(scores_normal: np.ndarray, scores_bsm_dict: dict) -> dict:
    """Compute AUROC and TPR@FPR for each BSM class and combined."""
    out = {}

    all_bsm = np.concatenate(list(scores_bsm_dict.values()))
    lbl_all = np.concatenate([np.zeros(len(scores_normal)), np.ones(len(all_bsm))])
    sc_all  = np.concatenate([scores_normal, all_bsm])
    out["all"] = {"auroc": float(roc_auc_score(lbl_all, sc_all))}

    for cls_name, scores_bsm in scores_bsm_dict.items():
        lbl = np.concatenate([np.zeros(len(scores_normal)), np.ones(len(scores_bsm))])
        sc  = np.concatenate([scores_normal, scores_bsm])
        fpr_arr, tpr_arr, _ = roc_curve(lbl, sc)
        tpr_at = {
            str(fpr_t): float(np.interp(fpr_t, fpr_arr, tpr_arr))
            for fpr_t in FPR_TARGETS
        }
        out[cls_name] = {
            "auroc":      float(roc_auc_score(lbl, sc)),
            "tpr_at_fpr": tpr_at,
        }
    return out


# ─── Per-seed pipeline ───────────────────────────────────────────────────────

def run_one_encoder_seed(
    encoder_seed: int,
    ckpt_path: Path,
    model_class_str: str,
    phase2_experiment: str,
    output_dir: Path,
    normal_classes: list,
    anomaly_classes: list,
    class_names: list,
    gmm_seed: int,
) -> dict:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"  [encoder_seed {encoder_seed}] Already done — loading {summary_path}")
        with open(summary_path) as f:
            return json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1 — extract embeddings (all SM + BSM)
    embedding_dir = output_dir / "embeddings"
    if not (embedding_dir / "train_embeddings.npz").exists():
        print(f"  [encoder_seed {encoder_seed}] Loading encoder from {ckpt_path.name}")
        model = load_encoder(ckpt_path, model_class_str)
        extract_and_save_embeddings(
            model=model,
            phase2_experiment=phase2_experiment,
            output_dir=embedding_dir,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"  [encoder_seed {encoder_seed}] Embeddings already extracted")

    # Step 2 — load embeddings
    train_data = np.load(embedding_dir / "train_embeddings.npz")
    val_data   = np.load(embedding_dir / "val_embeddings.npz")
    test_data  = np.load(embedding_dir / "test_embeddings.npz")

    z_train, lbl_train = train_data["embeddings"].astype(np.float64), train_data["labels"]
    z_val,   lbl_val   = val_data["embeddings"].astype(np.float64),   val_data["labels"]
    z_test,  lbl_test  = test_data["embeddings"].astype(np.float64),  test_data["labels"]

    sm_mask_train = np.isin(lbl_train, normal_classes)
    sm_mask_test  = np.isin(lbl_test,  normal_classes)
    qcd_mask_test = lbl_test == 0

    z_sm_train = z_train[sm_mask_train]

    bsm_cls_names = {idx: class_names[idx] for idx in anomaly_classes}
    z_bsm_test = {
        cls_name: z_test[lbl_test == idx]
        for idx, cls_name in bsm_cls_names.items()
        if (lbl_test == idx).any()
    }

    print(f"  [encoder_seed {encoder_seed}] SM train={len(z_sm_train)}")
    for cls_name, z in z_bsm_test.items():
        print(f"    BSM test {cls_name}: {len(z)}")
    missing = [n for _, n in bsm_cls_names.items() if n not in z_bsm_test]
    if missing:
        print(f"    WARNING: BSM classes with 0 test samples (skipped): {missing}")

    # Step 3 — fit one GMM per K, save results separately
    results_by_k = {}
    for k in K_VALUES:
        k_key = f"K{k}"
        k_dir = output_dir / k_key
        k_results_path = k_dir / "results.json"

        if k_results_path.exists():
            print(f"  [encoder_seed {encoder_seed}][{k_key}] Already done")
            with open(k_results_path) as f:
                results_by_k[k_key] = json.load(f)
            continue

        os.makedirs(k_dir, exist_ok=True)
        print(f"  [encoder_seed {encoder_seed}][{k_key}] Fitting GMM...")
        try:
            gmm = fit_gmm_k(z_sm_train, k=k, seed=gmm_seed)
        except Exception as e:
            print(f"  [encoder_seed {encoder_seed}][{k_key}] FAILED: {e}")
            continue

        np.savez_compressed(
            k_dir / "gmm.npz",
            means=gmm.means_, covariances=gmm.covariances_, weights=gmm.weights_,
        )

        s1_qcd,   s3_qcd   = compute_scores(gmm, z_test[qcd_mask_test])
        s1_allsm, s3_allsm = compute_scores(gmm, z_test[sm_mask_test])
        s1_bsm = {cls_name: compute_scores(gmm, z)[0] for cls_name, z in z_bsm_test.items()}
        s3_bsm = {cls_name: compute_scores(gmm, z)[1] for cls_name, z in z_bsm_test.items()}

        k_results = {
            "gmm_k":         k,
            "gmm_cov":       COV_TYPE,
            "qcd_nll":       evaluate_auroc(s1_qcd,   s1_bsm),
            "allsm_nll":     evaluate_auroc(s1_allsm, s1_bsm),
            "qcd_maxresp":   evaluate_auroc(s3_qcd,   s3_bsm),
            "allsm_maxresp": evaluate_auroc(s3_allsm, s3_bsm),
        }

        for bg_score in ("qcd_nll", "allsm_nll", "qcd_maxresp", "allsm_maxresp"):
            res = k_results[bg_score]
            auroc_all = res.get("all", {}).get("auroc", float("nan"))
            per = "  ".join(f"{n}={v['auroc']:.4f}" for n, v in res.items() if n != "all")
            print(f"    [{k_key}][{bg_score}]  all={auroc_all:.4f}  {per}")

        with open(k_results_path, "w") as f:
            json.dump(k_results, f, indent=2)
        results_by_k[k_key] = k_results

        gc.collect()

    summary = {
        "encoder_seed":  encoder_seed,
        "ckpt_path":     str(ckpt_path),
        "gmm_seed":      gmm_seed,
        "results_by_k":  results_by_k,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ─── Aggregation ─────────────────────────────────────────────────────────────

def _mean_std(values):
    vals = [v for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan")
    mu = sum(vals) / len(vals)
    if len(vals) == 1:
        return mu, float("nan")
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return mu, math.sqrt(var)


def aggregate(summaries: list, output_dir: Path) -> dict:
    # k_key → bg_score → cls_name → [auroc values across seeds]
    collected = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for s in summaries:
        for k_key, k_res in s.get("results_by_k", {}).items():
            for bg_score in ("qcd_nll", "allsm_nll", "qcd_maxresp", "allsm_maxresp"):
                if bg_score not in k_res:
                    continue
                for cls_name, metrics in k_res[bg_score].items():
                    if isinstance(metrics, dict) and "auroc" in metrics:
                        collected[k_key][bg_score][cls_name].append(metrics["auroc"])

    agg = {}
    for k_key, bg_dict in collected.items():
        agg[k_key] = {}
        for bg_score, cls_dict in bg_dict.items():
            agg[k_key][bg_score] = {
                cls_name: {
                    "mean":   _mean_std(vals)[0],
                    "std":    _mean_std(vals)[1],
                    "n":      len(vals),
                    "values": vals,
                }
                for cls_name, vals in cls_dict.items()
            }

    result = {"aggregated": agg, "n_encoder_seeds": len(summaries)}
    with open(output_dir / "aggregated_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  GMM aggregated results — {len(summaries)} encoder seeds")
    print(f"{'='*80}")
    for k_key in sorted(agg):
        print(f"\n  [{k_key}]")
        for bg_score, cls_dict in agg[k_key].items():
            print(f"    [{bg_score}]")
            for cls_name, v in cls_dict.items():
                std_str = f"±{v['std']:.4f}" if not math.isnan(v["std"]) else ""
                print(f"      {cls_name:20s}: {v['mean']:.4f}{std_str}  (n={v['n']})")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-experiment", required=True,
                        help="Hydra experiment config (must have normal_classes=[0..11])")
    parser.add_argument("--encoder-seeds-dir", required=True,
                        help="Dir with seed_0/, seed_1/, ... encoder checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gmm-seed", type=int, default=42)
    parser.add_argument("--encoder-seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    encoder_seeds_dir = Path(args.encoder_seeds_dir)

    cfg = _compose_cfg("anomaly_detection.yaml",
                       [f"experiment={args.phase2_experiment}"],
                       output_dir=output_dir)
    model_class_str = cfg.get("model_class")
    if not model_class_str:
        raise ValueError(f"Config '{args.phase2_experiment}' must define 'model_class'")
    normal_classes  = list(cfg.get("normal_classes", []))
    anomaly_classes = list(cfg.get("anomaly_classes", []))
    class_names     = list(cfg.data.to_classify)

    print(f"\n{'='*80}")
    print(f"  GMM encoder-seeds pipeline")
    print(f"  Phase2 experiment : {args.phase2_experiment}")
    print(f"  Encoder seeds dir : {encoder_seeds_dir}")
    print(f"  Encoder seeds     : {args.encoder_seeds}")
    print(f"  GMM seed          : {args.gmm_seed}")
    print(f"  Normal classes    : {normal_classes}")
    print(f"  Anomaly classes   : {anomaly_classes}")
    print(f"  K values          : {K_VALUES}  cov: {COV_TYPE}")
    print(f"  Output dir        : {output_dir}")
    print(f"{'='*80}\n")

    summaries = []
    for enc_seed in args.encoder_seeds:
        seed_input_dir  = encoder_seeds_dir / f"seed_{enc_seed}"
        seed_output_dir = output_dir / f"encoder_seed_{enc_seed}"

        print(f"\n{'─'*80}")
        print(f"  ENCODER SEED {enc_seed}")
        print(f"{'─'*80}")

        try:
            ckpt_path = find_best_ckpt(seed_input_dir)
            print(f"  Checkpoint: {ckpt_path}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        summary = run_one_encoder_seed(
            encoder_seed=enc_seed,
            ckpt_path=ckpt_path,
            model_class_str=model_class_str,
            phase2_experiment=args.phase2_experiment,
            output_dir=seed_output_dir,
            normal_classes=normal_classes,
            anomaly_classes=anomaly_classes,
            class_names=class_names,
            gmm_seed=args.gmm_seed,
        )
        summaries.append(summary)

    if summaries:
        aggregate(summaries, output_dir)

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
