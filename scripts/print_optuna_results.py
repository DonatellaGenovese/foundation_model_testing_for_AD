#!/usr/bin/env python3
"""
Simple script to print Optuna study results.

Usage:
    python scripts/print_optuna_results.py
    
Or with custom paths:
    python scripts/print_optuna_results.py --db logs/optuna_augmented_supcon_probes.db --study augmented_supcon_probes
"""

import argparse
from pathlib import Path
import optuna
import pandas as pd

def print_study_results(db_path: str, study_name: str, top_n: int = 10):
    """Print Optuna study results."""
    
    # Check if database exists
    db_file = Path(db_path.replace("sqlite:///", ""))
    if not db_file.exists():
        print(f"❌ Database not found: {db_file}")
        print(f"   Make sure you've run the Optuna sweep first!")
        return
    
    # Load study
    print(f"📊 Loading Optuna study from: {db_path}")
    print(f"   Study name: {study_name}\n")
    
    storage = f"sqlite:///{db_file.absolute()}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Print summary
    print("=" * 80)
    print("OPTUNA STUDY SUMMARY")
    print("=" * 80)
    print(f"Study name:        {study.study_name}")
    print(f"Direction:         {study.direction}")
    print(f"Number of trials:  {len(study.trials)}")
    print(f"Best value:        {study.best_value:.4f}")
    print(f"Best trial:        #{study.best_trial.number}")
    print()
    
    # Print best parameters
    print("=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    for param, value in study.best_params.items():
        print(f"  {param:40s} = {value}")
    print()
    
    # Print top trials
    print("=" * 80)
    print(f"TOP {top_n} TRIALS")
    print("=" * 80)
    
    df = study.trials_dataframe()
    
    # Sort by value (descending for maximize, ascending for minimize)
    ascending = study.direction == optuna.study.StudyDirection.MINIMIZE
    df_sorted = df.sort_values("value", ascending=ascending).head(top_n)
    
    # Select relevant columns
    cols = ["number", "value", "state"]
    param_cols = [col for col in df.columns if col.startswith("params_")]
    display_cols = cols + param_cols + ["datetime_complete", "duration"]
    display_cols = [col for col in display_cols if col in df.columns]
    
    print(df_sorted[display_cols].to_string(index=False))
    print()
    
    # Print statistics
    print("=" * 80)
    print("PARAMETER STATISTICS")
    print("=" * 80)
    
    # Calculate parameter importances
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter Importances:")
        for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param:40s} = {importance:.4f}")
    except Exception as e:
        print(f"Could not calculate parameter importances: {e}")
    
    print()
    print("=" * 80)
    print(f"✅ Results saved to: {db_file}")
    print(f"💡 View full results in notebook: notebooks/optuna_sweep_results.ipynb")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Print Optuna study results")
    parser.add_argument(
        "--db",
        type=str,
        default="logs/optuna_augmented_supcon_probes.db",
        help="Path to Optuna database (default: logs/optuna_augmented_supcon_probes.db)"
    )
    parser.add_argument(
        "--study",
        type=str,
        default="augmented_supcon_probes",
        help="Study name (default: augmented_supcon_probes)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top trials to display (default: 10)"
    )
    
    args = parser.parse_args()
    print_study_results(args.db, args.study, args.top)


if __name__ == "__main__":
    main()
