"""Save Optuna hyperparameter sweep results as a formatted table and CSV."""
import argparse
import csv
import io
import shutil
import tempfile
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

EOS_STUDY_DIR = Path("/eos/user/d/dgenoves/anomaly_pipeline/optuna")

P1_PARAMS = [
    "p1_d_model", "p1_n_heads", "p1_num_layers", "p1_d_ff_mult",
    "p1_dropout", "p1_projection_dim", "p1_hidden_projection_dim",
    "p1_temperature", "p1_lr", "p1_weight_decay", "p1_mask_probability",
]
AE_PARAMS = [
    "ae_compression", "ae_depth", "ae_dropout", "ae_lr", "ae_weight_decay",
]
ALL_PARAMS = P1_PARAMS + AE_PARAMS


def fmt(val):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.2e}" if val < 0.01 else f"{val:.4f}"
    return str(val)


def build_study_output(enc: str, db_dir: Path) -> tuple[str, list[dict]]:
    """Returns (formatted_text, list_of_row_dicts) for one encoder study."""
    db_path = db_dir / f"study_{enc}.db"
    if not db_path.exists():
        return f"\n=== {enc} — DB not found at {db_path} ===\n", []

    summaries = optuna.get_all_study_summaries(f"sqlite:///{db_path}")
    if not summaries:
        return f"\n=== {enc} — no study found ===\n", []

    study_name = summaries[0].study_name
    s = optuna.load_study(study_name, f"sqlite:///{db_path}")
    done = [t for t in s.trials if t.state.name == "COMPLETE"]
    pareto = set(t.number for t in s.best_trials)

    present_params = [p for p in ALL_PARAMS if any(p in t.params for t in done)]

    buf = io.StringIO()
    buf.write(f"\n{'='*140}\n")
    buf.write(f"  {enc.upper()} ({study_name}) — {len(done)} completed trials, {len(pareto)} on Pareto front\n")
    buf.write(f"{'='*140}\n")

    header = f"{'#':>4}  {'sep':>7}  {'drift':>8}  {'Pareto':>6}"
    for p in present_params:
        short = p.replace("p1_", "").replace("ae_", "ae.")
        header += f"  {short:>13}"
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")

    rows = []
    for t in sorted(done, key=lambda x: x.number):
        star = "*" if t.number in pareto else " "
        row_str = f"{t.number:>4}{star} {t.values[0]:>7.4f}  {t.values[1]:>8.6f}  {star:>6}"
        for p in present_params:
            row_str += f"  {fmt(t.params.get(p)):>13}"
        buf.write(row_str + "\n")

        rows.append({
            "encoder": enc,
            "trial": t.number,
            "sep_ratio": t.values[0],
            "val_drift": t.values[1],
            "pareto": t.number in pareto,
            **{p: t.params.get(p) for p in ALL_PARAMS},
        })

    return buf.getvalue(), rows


def main():
    parser = argparse.ArgumentParser(description="Save Optuna sweep results for anomaly detection")
    parser.add_argument(
        "--encoders", nargs="+",
        default=["aug", "selfsup"],
        help="Which encoder types to include (default: all)"
    )
    parser.add_argument(
        "--db-dir", type=Path, default=None,
        help="Directory containing study_*.db files (default: copies from EOS to /tmp)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("logs/optuna_results"),
        help="Directory to save results (default: logs/optuna_results)"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def run(db_dir: Path):
        all_rows = []
        full_text = ""
        for enc in args.encoders:
            text, rows = build_study_output(enc, db_dir)
            full_text += text
            all_rows.extend(rows)

        # Save formatted text table
        txt_path = args.out_dir / "results.txt"
        txt_path.write_text(full_text)
        print(f"Saved table  → {txt_path}")

        # Save CSV
        if all_rows:
            csv_path = args.out_dir / "results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"Saved CSV    → {csv_path}")

        print(full_text)

    if args.db_dir is not None:
        run(args.db_dir)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for enc in args.encoders:
                src = EOS_STUDY_DIR / f"study_{enc}.db"
                dst = tmpdir / f"study_{enc}.db"
                try:
                    shutil.copy(src, dst)
                except Exception as e:
                    print(f"Could not copy {src}: {e}")
                    print(f"  Try: cp {src} /tmp/ && python scripts/print_optuna_results.py --db-dir /tmp")
                    continue
            run(tmpdir)


if __name__ == "__main__":
    main()
