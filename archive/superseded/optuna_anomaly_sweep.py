"""
Multi-objective Optuna sweep for the full anomaly detection pipeline.

Optimises hyperparameters in both Phase 1 (encoder) and Phase 2 (AE)
jointly, targeting two objectives:
    - ae/separation_ratio  → maximise  (anomaly MSE / normal MSE)
    - ae/drift_metric      → minimise  (FPR stability at fixed thresholds)

Each trial runs the complete two-phase pipeline and writes results to a
dedicated sub-directory.  Each encoder type runs trials sequentially in one Condor job.
Results are persisted to a SQLite database on EOS.

Usage (single trial, from Condor):
    python src/optuna_anomaly_sweep.py \\
        --encoder-type aug \\
        --storage /eos/user/d/dgenoves/anomaly_pipeline/optuna/study_aug.db \\
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/optuna \\
        --n-trials 1

Encoder types:
    aug      →  aug_supcon_12class_nosparse_dmodel128_cern
    selfsup  →  aug_supcon_12class_nosparse_selfsup_dmodel128_cern
"""

import argparse
import os
import math
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import optuna

from src.train_full_anomaly_pipeline import run_full_anomaly_pipeline
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# ─── Experiment registry ──────────────────────────────────────────────────────

ENCODER_CONFIGS = {
    "aug":     "aug_supcon_12class_nosparse_dmodel128_cern",
    "selfsup": "aug_supcon_12class_nosparse_selfsup_dmodel128_cern",
}

PHASE2_EXPERIMENT = "anomaly_qcd_vs_higgs_embedding_augsupcon_nosparse_dmodel128_cern"

# HPO uses half the default splits to keep trial time manageable.
# Best HPs found here are then re-used for a full-data final run.
HPO_PHASE1_SPLITS = "data.train_val_test_split_per_class=[500000,50000,50000]"
HPO_PHASE2_SPLITS = "data.train_val_test_split_per_class=[100000,10000,10000]"


# ─── Hyperparameter search spaces ─────────────────────────────────────────────

def _sample_phase1_overrides(trial: optuna.Trial, encoder_type: str) -> list:
    """
    Sample Phase 1 (encoder) hyperparameters.

    Architecture (common to all encoder types):
        d_model, n_heads, num_layers, d_ff, dropout, projection_dim

    Training (common to all):
        temperature, lr, weight_decay

    Aug / selfsup only:
        mask_probability

    Constraints:
        n_heads is chosen from values that always divide d_model exactly.
        d_ff is expressed as a multiplier of d_model so it scales consistently.
    """
    overrides = []

    # ── Architecture ──────────────────────────────────────────────────────────
    # d_model in {256, 512}; n_heads in {4, 8} — both divide either d_model
    # (256/4=64, 256/8=32, 512/4=128, 512/8=64 tokens per head — all valid).
    d_model = trial.suggest_categorical("p1_d_model", [256, 512])
    overrides.append(f"model.d_model={d_model}")

    n_heads = trial.suggest_categorical("p1_n_heads", [4, 8])
    overrides.append(f"model.n_heads={n_heads}")

    num_layers = trial.suggest_int("p1_num_layers", 2, 8)
    overrides.append(f"model.num_layers={num_layers}")

    # d_ff as a multiplier of d_model (2x or 4x — standard Transformer choices)
    d_ff_mult = trial.suggest_categorical("p1_d_ff_mult", [2, 4])
    overrides.append(f"model.d_ff={d_model * d_ff_mult}")

    # Encoder dropout
    dropout = trial.suggest_float("p1_dropout", 0.0, 0.3)
    overrides.append(f"model.dropout={dropout}")

    # Projection head sizes
    projection_dim = trial.suggest_categorical("p1_projection_dim", [32, 64, 128, 256])
    overrides.append(f"model.projection_dim={projection_dim}")

    hidden_projection_dim = trial.suggest_categorical("p1_hidden_projection_dim", [128, 256, 512])
    overrides.append(f"model.hidden_projection_dim={hidden_projection_dim}")

    # ── Training ──────────────────────────────────────────────────────────────
    # Contrastive temperature
    temperature = trial.suggest_float("p1_temperature", 0.03, 0.3, log=True)
    overrides.append(f"model.temperature={temperature}")

    # Optimiser
    lr = trial.suggest_float("p1_lr", 5e-5, 5e-4, log=True)
    overrides.append(f"model.optimizer.lr={lr}")

    wd = trial.suggest_float("p1_weight_decay", 1e-4, 0.1, log=True)
    overrides.append(f"model.optimizer.weight_decay={wd}")

    # ── Aug-specific ──────────────────────────────────────────────────────────
    if encoder_type in ("aug", "selfsup"):
        mask_prob = trial.suggest_float("p1_mask_probability", 0.1, 0.5)
        overrides.append(f"model.mask_probability={mask_prob}")

    return overrides


def _sample_phase2_overrides(trial: optuna.Trial) -> list:
    """
    Sample Phase 2 (autoencoder) hyperparameters.

        compression  — bottleneck ratio  (input_dim / bottleneck_dim)
        depth        — number of encoder hidden layers
        dropout      — AE dropout
        lr           — AE learning rate
        weight_decay — AE weight decay
    """
    overrides = []

    compression = trial.suggest_categorical("ae_compression", [2, 4, 8, 16])
    overrides.append(f"model.compression={compression}")

    depth = trial.suggest_int("ae_depth", 1, 4)
    overrides.append(f"model.depth={depth}")

    dropout = trial.suggest_float("ae_dropout", 0.0, 0.3)
    overrides.append(f"model.dropout={dropout}")

    lr = trial.suggest_float("ae_lr", 1e-4, 1e-2, log=True)
    overrides.append(f"model.lr={lr}")

    wd = trial.suggest_float("ae_weight_decay", 1e-6, 1e-2, log=True)
    overrides.append(f"model.weight_decay={wd}")

    return overrides


# ─── Objective ────────────────────────────────────────────────────────────────

def objective(
    trial: optuna.Trial,
    encoder_type: str,
    output_base_dir: Path,
) -> tuple:
    """
    Run the full two-phase anomaly pipeline for one Optuna trial.

    Returns:
        (separation_ratio, drift_metric)
        Directions: maximise separation_ratio, minimise drift_metric.
    """
    p1_overrides = _sample_phase1_overrides(trial, encoder_type) + [HPO_PHASE1_SPLITS]
    p2_overrides = _sample_phase2_overrides(trial) + [HPO_PHASE2_SPLITS]

    trial_dir = output_base_dir / encoder_type / f"trial_{trial.number}"

    log.info(f"[Trial {trial.number}] encoder_type  = {encoder_type}")
    log.info(f"[Trial {trial.number}] phase1 overrides: {p1_overrides}")
    log.info(f"[Trial {trial.number}] phase2 overrides: {p2_overrides}")
    log.info(f"[Trial {trial.number}] output_dir    = {trial_dir}")

    metrics = run_full_anomaly_pipeline(
        phase1_experiment=ENCODER_CONFIGS[encoder_type],
        phase2_experiment=PHASE2_EXPERIMENT,
        output_dir=trial_dir,
        phase1_overrides=p1_overrides,
        phase2_overrides=p2_overrides,
    )

    sep       = metrics["separation_ratio"]
    val_drift = metrics["val_drift_metric"]   # val holdout — unbiased HPO metric
    drift     = metrics["drift_metric"]       # test set — final report only

    log.info(
        f"[Trial {trial.number}] separation_ratio={sep:.4f}  "
        f"val_drift={val_drift:.4f}  test_drift={drift:.4f}"
    )
    # Return val_drift as the second objective — test drift is never seen by Optuna
    return sep, val_drift


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective Optuna sweep for the anomaly detection pipeline"
    )
    parser.add_argument(
        "--encoder-type",
        choices=list(ENCODER_CONFIGS.keys()),
        required=True,
        help="Encoder type: aug | selfsup",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of trials to run in this worker (default: 1 for Condor jobs)",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Path to JournalFileStorage for shared study coordination across "
            "parallel Condor jobs (e.g. "
            "/eos/user/d/dgenoves/anomaly_pipeline/optuna/study_aug.db). "
            "If omitted, an in-memory (non-persistent) study is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for trial outputs (one sub-dir per trial)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the Optuna TPE sampler",
    )
    args = parser.parse_args()

    # Ensure project root is cwd (Hydra / rootutils requirement)
    os.chdir(Path(__file__).parent.parent)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"anomaly_sweep_{args.encoder_type}"

    if args.storage is not None:
        db_path = Path(args.storage).with_suffix(".db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"
        log.info(f"Optuna storage : {db_path}")
    else:
        storage = None
        log.info("Optuna storage : in-memory (results will not be persisted)")

    # NSGAIISampler is Optuna's recommended multi-objective sampler.
    # population_size controls the Pareto-front population; 10 is enough for
    # early exploration (increase once you have >20 trials).
    sampler = optuna.samplers.NSGAIISampler(
        population_size=10,
        seed=args.seed,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "minimize"],   # separation_ratio ↑, drift_metric ↓
        sampler=sampler,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args.encoder_type, output_dir),
        n_trials=args.n_trials,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    pareto = study.best_trials
    log.info(f"\n{'='*70}")
    log.info(f"Study '{study_name}' — {len(pareto)} Pareto-optimal trial(s)")
    log.info(f"{'='*70}")
    log.info(f"  {'Trial':>6}  {'sep_ratio':>10}  {'drift':>8}  params")
    log.info(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*40}")
    for t in sorted(pareto, key=lambda t: t.values[0], reverse=True):
        sep, drift = t.values
        log.info(f"  {t.number:>6}  {sep:>10.4f}  {drift:>8.4f}  {t.params}")
    log.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
