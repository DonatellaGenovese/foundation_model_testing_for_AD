"""
Match Optuna trials to MLflow runs and extract per-FPR drift values.

For each completed Optuna trial, finds the corresponding MLflow run in the
anomaly_detection experiment by matching timestamps, then reads:
    ae/drift_fpr01, ae/drift_fpr05, ae/drift_fpr10

Saves results to logs/optuna_results/drift_details.csv

Usage:
    python scripts/extract_drift_from_mlflow.py --db-dir /tmp
"""

import argparse
import csv
from pathlib import Path
from datetime import timezone

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

MLFLOW_DIR = Path("logs/mlflow/mlruns")
ANOMALY_EXPERIMENT_NAME = "anomaly_detection"
OUTPUT_DIR = Path("logs/optuna_results")

ENCODERS = {
    "aug":     "anomaly_sweep_aug",
    "selfsup": "anomaly_sweep_selfsup",
}


def find_experiment_id(name: str) -> str:
    for meta in MLFLOW_DIR.glob("*/meta.yaml"):
        lines = meta.read_text().splitlines()
        for line in lines:
            if line.startswith("name:") and line.split(":", 1)[1].strip() == name:
                for l in lines:
                    if l.startswith("experiment_id:"):
                        return l.split(":", 1)[1].strip().strip("'\"")
    raise ValueError(f"MLflow experiment '{name}' not found in {MLFLOW_DIR}")


def load_mlflow_runs(exp_id: str) -> list:
    """Load all runs from a MLflow experiment as list of dicts."""
    runs = []
    exp_dir = MLFLOW_DIR / exp_id
    for run_dir in exp_dir.iterdir():
        meta_path = run_dir / "meta.yaml"
        if not meta_path.exists() or run_dir.name == "meta.yaml":
            continue
        meta = {}
        for line in meta_path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip("'\"")

        # Read drift metrics
        metrics = {}
        for metric_name in ["drift_metric", "drift_fpr01", "drift_fpr05", "drift_fpr10", "separation_ratio"]:
            mpath = run_dir / "metrics" / "ae" / metric_name
            if mpath.exists():
                last_line = mpath.read_text().strip().splitlines()[-1]
                metrics[metric_name] = float(last_line.split()[1])
            else:
                metrics[metric_name] = float("nan")

        runs.append({
            "run_id":     meta.get("run_id", ""),
            "start_time": int(meta.get("start_time", 0)),   # ms
            "end_time":   int(meta.get("end_time", 0)),     # ms
            "status":     int(meta.get("status", 0)),
            **metrics,
        })
    return runs


def match_trial_to_run(trial, mlflow_runs: list) -> dict | None:
    """Find MLflow run whose start_time falls within the trial's time window."""
    if trial.datetime_start is None or trial.datetime_complete is None:
        return None

    # Convert trial datetimes to ms timestamps
    t_start = int(trial.datetime_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    t_end   = int(trial.datetime_complete.replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Add tolerance: MLflow run for Phase 2 starts after Phase 1 finishes,
    # so it's always within the trial window but possibly close to the end.
    best = None
    best_delta = float("inf")
    for run in mlflow_runs:
        if run["status"] != 3:  # 3 = FINISHED
            continue
        rs = run["start_time"]
        if t_start <= rs <= t_end:
            delta = abs(rs - t_start)
            if delta < best_delta:
                best_delta = delta
                best = run
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True,
                        help="Directory containing study_aug.db, study_selfsup.db")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load MLflow anomaly_detection runs once
    exp_id = find_experiment_id(ANOMALY_EXPERIMENT_NAME)
    print(f"MLflow experiment '{ANOMALY_EXPERIMENT_NAME}' → id={exp_id}")
    mlflow_runs = load_mlflow_runs(exp_id)
    print(f"Loaded {len(mlflow_runs)} MLflow runs")

    rows = []

    for enc, study_name in ENCODERS.items():
        db_path = db_dir / f"study_{enc}.db"
        if not db_path.exists():
            print(f"WARNING: {db_path} not found, skipping {enc}")
            continue

        study = optuna.load_study(study_name, f"sqlite:///{db_path}")
        done = [t for t in study.trials if t.state.name == "COMPLETE"]
        print(f"\n{enc}: {len(done)} completed trials")

        for trial in done:
            run = match_trial_to_run(trial, mlflow_runs)
            if run is None:
                print(f"  trial {trial.number}: no MLflow match found")
                drift_fpr01 = drift_fpr05 = drift_fpr10 = drift_mean = float("nan")
            else:
                drift_fpr01 = run["drift_fpr01"]
                drift_fpr05 = run["drift_fpr05"]
                drift_fpr10 = run["drift_fpr10"]
                drift_mean  = run["drift_metric"]
                print(f"  trial {trial.number}: matched run {run['run_id'][:8]}  "
                      f"fpr01={drift_fpr01:.4f}  fpr05={drift_fpr05:.4f}  fpr10={drift_fpr10:.4f}")

            rows.append({
                "encoder":       enc,
                "trial":         trial.number,
                "sep_ratio":     trial.values[0],
                "val_drift":     trial.values[1],
                "drift_mean":    drift_mean,
                "drift_fpr01":   drift_fpr01,
                "drift_fpr05":   drift_fpr05,
                "drift_fpr10":   drift_fpr10,
                "pareto":        trial in study.best_trials,
            })

    # Save CSV
    out_csv = OUTPUT_DIR / "drift_details.csv"
    fieldnames = ["encoder", "trial", "sep_ratio", "val_drift",
                  "drift_mean", "drift_fpr01", "drift_fpr05", "drift_fpr10", "pareto"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_csv}")

    # Print summary table
    print(f"\n{'enc':<8} {'#':>4}  {'sep':>7}  {'val_drift':>9}  {'fpr01':>7}  {'fpr05':>7}  {'fpr10':>7}  pareto")
    print("-" * 72)
    for r in rows:
        p = "YES" if r["pareto"] else ""
        print(f"{r['encoder']:<8} {r['trial']:>4}  {r['sep_ratio']:>7.4f}  "
              f"{r['val_drift']:>9.6f}  {r['drift_fpr01']:>7.4f}  "
              f"{r['drift_fpr05']:>7.4f}  {r['drift_fpr10']:>7.4f}  {p}")


if __name__ == "__main__":
    main()
