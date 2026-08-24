#!/usr/bin/env python3
"""
Aggregate LR and weight_decay ablation probe results and write LaTeX AUROC tables.

Usage:
    python3 scripts/ablation/aggregate_probe_results_lr_wd.py
    python3 scripts/ablation/aggregate_probe_results_lr_wd.py --ablation lr
    python3 scripts/ablation/aggregate_probe_results_lr_wd.py --ablation wd
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ABLATIONS = {
    "lr": {
        "train_base": Path("/eos/user/d/dgenoves/anomaly_pipeline/ablation/lr/logs/train/runs"),
        "output_dir": Path("/eos/user/d/dgenoves/anomaly_pipeline/ablation/lr/results"),
        "param_key": "lr_slug",
        "param_values": ["1e4", "5e4", "1e3", "3e3"],
        "run_name_re": re.compile(r"(\w+)_(lr)_(\w+)_seed(\d+)"),
        "tex_param": {
            "1e4": r"10^{-4}",
            "5e4": r"5\times10^{-4}",
            "1e3": r"10^{-3}",
            "3e3": r"3\times10^{-3}",
        },
        "caption_name": "learning rate",
        "caption_param": r"\mathrm{lr}",
        "label": "tab:ablation_lr_auroc",
        "table_stem": "lr_auroc_table",
    },
    "wd": {
        "train_base": Path("/eos/user/d/dgenoves/anomaly_pipeline/ablation/wd/logs/train/runs"),
        "output_dir": Path("/eos/user/d/dgenoves/anomaly_pipeline/ablation/wd/results"),
        "param_key": "wd_slug",
        "param_values": ["0", "1e5", "1e3", "1e2"],
        "run_name_re": re.compile(r"(\w+)_(wd)_(\w+)_seed(\d+)"),
        "tex_param": {
            "0": "0",
            "1e5": r"10^{-5}",
            "1e3": r"10^{-3}",
            "1e2": r"10^{-2}",
        },
        "caption_name": "weight decay",
        "caption_param": r"\mathrm{wd}",
        "label": "tab:ablation_wd_auroc",
        "table_stem": "wd_auroc_table",
    },
    "bs": {
        "train_base": Path("/eos/user/d/dgenoves/anomaly_pipeline/ablation/bs/logs/train/runs"),
        "output_dir": Path("/eos/user/d/dgenoves/anomaly_pipeline/ablation/bs/results"),
        "param_key": "bs_slug",
        "param_values": ["512", "1024", "2048"],
        "run_name_re": re.compile(r"(\w+)_(bs)_(\d+)_seed(\d+)"),
        "tex_param": {
            "512": "512",
            "1024": "1024",
            "2048": "2048",
        },
        "caption_name": "batch size",
        "caption_param": r"B",
        "label": "tab:ablation_bs_auroc",
        "table_stem": "bs_auroc_table",
        "caption_extra": "",
    },
}

MODEL_ORDER = ["vcreg", "aug_supcon", "selfsupcon", "vicreg"]
MODEL_LABEL = {
    "vcreg": "VCReg",
    "aug_supcon": "Aug. SupCon",
    "selfsupcon": "Self-SupCon",
    "vicreg": "VICReg",
}


def collect_rows(cfg: dict) -> list[dict]:
    rows = []
    best: dict[tuple, Path] = {}
    for run_dir in sorted(cfg["train_base"].iterdir()):
        if not run_dir.is_dir():
            continue
        hydra_cfg = run_dir / ".hydra" / "config.yaml"
        probe = run_dir / "probe_evaluation" / "probe_results.json"
        if not hydra_cfg.exists() or not probe.exists():
            continue

        text = hydra_cfg.read_text()
        run_name_m = re.search(r"run_name:\s*(\S+)", text)
        if not run_name_m:
            continue
        m = cfg["run_name_re"].match(run_name_m.group(1))
        if not m:
            continue

        model_short, _, param_slug, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        key = (model_short, param_slug, seed)
        mtime = run_dir.stat().st_mtime
        if key not in best or mtime > best[key].stat().st_mtime:
            best[key] = run_dir

    for (model_short, param_slug, seed), run_dir in sorted(best.items()):
        with open(run_dir / "probe_evaluation" / "probe_results.json") as f:
            metrics = json.load(f)
        rows.append({
            "model_short": model_short,
            "model": MODEL_LABEL[model_short],
            cfg["param_key"]: param_slug,
            "seed": seed,
            "auroc": float(metrics["linear_probe_auroc_macro"]),
            "run_dir": run_dir.name,
        })
    return rows


def aggregate(rows: list[dict], param_key: str) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["model_short"], row[param_key])].append(row["auroc"])

    out = []
    for (model_short, param_slug) in sorted(groups):
        vals = groups[(model_short, param_slug)]
        out.append({
            "model_short": model_short,
            "model": MODEL_LABEL[model_short],
            param_key: param_slug,
            "n_seeds": len(vals),
            "auroc_mean": float(np.mean(vals)),
            "auroc_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
        })
    return out


def fmt_cell(mean: float, std: float, n: int) -> str:
    return f"${mean:.4f} \\pm {std:.4f}$"


def write_latex_table(cfg: dict, agg: list[dict], path: Path) -> None:
    param_key = cfg["param_key"]
    param_values = cfg["param_values"]
    lookup = {(r["model_short"], r[param_key]): r for r in agg}

    col_spec = "l" + "c" * len(param_values)
    header = " & ".join(
        [f"${cfg['caption_param']}={cfg['tex_param'][p]}$" for p in param_values]
    )

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        (
            f"  \\caption{{Linear probe AUROC (macro) for the {cfg['caption_name']} ablation "
            r"($d_{\mathrm{model}}=128$, $n_{\mathrm{layers}}=6$, $n_{\mathrm{heads}}=8$, "
            r"$d_{\mathrm{ff}}=512$, dropout $=0.1$). Mean $\pm$ std over seeds."
            f"{cfg.get('caption_extra', '')}}}"
        ),
        f"  \\label{{{cfg['label']}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Model & {header} \\\\",
        r"    \midrule",
    ]

    for model_short in MODEL_ORDER:
        cells = [MODEL_LABEL[model_short]]
        for param_slug in param_values:
            row = lookup.get((model_short, param_slug))
            if row is None:
                cells.append("---")
            else:
                cells.append(fmt_cell(row["auroc_mean"], row["auroc_std"], row["n_seeds"]))
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_ablation(name: str) -> None:
    cfg = ABLATIONS[name]
    rows = collect_rows(cfg)
    if not rows:
        raise SystemExit(f"No probe results found for {name} ablation.")

    agg = aggregate(rows, cfg["param_key"])
    out_dir = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / f"{name}_auroc_per_run.csv", rows)
    write_csv(out_dir / f"{name}_auroc_aggregated.csv", agg)
    tex_path = out_dir / f"{cfg['table_stem']}.tex"
    write_latex_table(cfg, agg, tex_path)

    print(f"\n=== {name.upper()} ablation ===")
    print(f"Runs: {len(rows)}")
    print(f"Saved: {tex_path}")
    for row in agg:
        print(
            f"  {row['model_short']:<12} {cfg['param_key']}={row[cfg['param_key']]:>4} "
            f"n={row['n_seeds']}  {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["lr", "wd", "bs", "all"], default="both")
    args = parser.parse_args()

    if args.ablation == "all":
        names = ["lr", "wd", "bs"]
    elif args.ablation == "both":
        names = ["lr", "wd"]
    else:
        names = [args.ablation]
    for name in names:
        process_ablation(name)


if __name__ == "__main__":
    main()
