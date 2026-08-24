"""
Linear probe on raw preprocessed features (lower bound baseline).

Trains nn.Linear(340, 12) directly on the frozen raw 340-dim feature vector,
with no encoder. Reports macro-averaged AUROC over 12 SM classes.

Usage:
    python scripts/eval_raw_linear_probe.py \
        --data-dir /eos/user/d/dgenoves/foundation_model_testing_data/v2_12class_nosparse_highlevel/preprocessed \
        --output-dir /eos/user/d/dgenoves/anomaly_pipeline/raw_linear_probe/seed_0 \
        --seed 0
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score

CLASS_NAMES = [
    "QCD_inclusive", "Z_to_vv_jet", "Z_to_qq_uds", "Z_to_bb", "Z_to_cc",
    "W_to_lv", "W_to_qq", "gamma", "QCD_bb",
    "tt_all-hadr", "tt_semi-lept", "tt_all-lept",
]

FOLDER_MAP = {
    "QCD_inclusive":  "QCD_HT50toInf",
    "Z_to_vv_jet":    "ZJetsTovv_13TeV-madgraphMLM-pythia8",
    "Z_to_qq_uds":    "ZJetsToQQ_13TeV-madgraphMLM-pythia8",
    "Z_to_bb":        "ZJetsTobb_13TeV-madgraphMLM-pythia8",
    "Z_to_cc":        "ZJetsTocc_13TeV-madgraphMLM-pythia8",
    "W_to_lv":        "WJetsToLNu_13TeV-madgraphMLM-pythia8",
    "W_to_qq":        "WJetsToQQ_13TeV-madgraphMLM-pythia8",
    "gamma":          "gamma",
    "QCD_bb":         "QCD_HT50tobb",
    "tt_all-hadr":    "tt0123j_5f_ckm_LO_MLM_hadronic",
    "tt_semi-lept":   "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    "tt_all-lept":    "tt0123j_5f_ckm_LO_MLM_leptonic",
}

N_TRAIN = 1_000_000
N_VAL   =   100_000
N_TEST  =   100_000


def load_split(data_dir: Path, split: str, n_per_class: int, seed: int):
    xs, ys = [], []
    rng = np.random.default_rng(seed)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        folder = FOLDER_MAP[cls_name]
        cls_dir = data_dir / split / folder
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing: {cls_dir}")
        files = sorted(f for f in cls_dir.iterdir() if f.name.endswith("_x.npy"))
        chunks = [np.load(f) for f in files]
        X = np.concatenate(chunks, axis=0)
        n = min(n_per_class, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        xs.append(torch.from_numpy(X[idx]).float())
        ys.append(torch.full((n,), cls_idx, dtype=torch.long))
        print(f"  {split}/{folder}: {n} events")
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def train_linear_probe(train_ds, val_ds, input_dim: int, n_classes: int,
                       seed: int, max_epochs: int = 50,
                       lr: float = 1e-3, batch_size: int = 1024, device: str = "cpu"):
    torch.manual_seed(seed)
    model = nn.Linear(input_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(y)
                n += len(y)
        val_loss /= n
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch+1:3d}/{max_epochs}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, test_ds, n_classes: int, batch_size: int = 1024, device: str = "cpu"):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    auroc_macro  = MulticlassAUROC(num_classes=n_classes, average="macro").to(device)
    auroc_per_cls = MulticlassAUROC(num_classes=n_classes, average=None).to(device)
    acc          = MulticlassAccuracy(num_classes=n_classes, average="macro").to(device)
    f1_macro     = MulticlassF1Score(num_classes=n_classes, average="macro").to(device)

    model.eval()
    model.to(device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1)
        auroc_macro.update(probs, y)
        auroc_per_cls.update(probs, y)
        acc.update(probs, y)
        f1_macro.update(logits, y)

    results = {
        "linear_probe_accuracy":    acc.compute().item(),
        "linear_probe_f1_macro":    f1_macro.compute().item(),
        "linear_probe_auroc_macro": auroc_macro.compute().item(),
    }
    per_cls = auroc_per_cls.compute()
    for i, cls_name in enumerate(CLASS_NAMES):
        results[f"linear_probe_auroc_{cls_name}"] = per_cls[i].item()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[seed={args.seed}] Loading data from {args.data_dir}")
    print("  Loading train split...")
    train_ds = load_split(args.data_dir, "train", N_TRAIN // len(CLASS_NAMES), args.seed)
    print("  Loading val split...")
    val_ds   = load_split(args.data_dir, "val",   N_VAL   // len(CLASS_NAMES), args.seed)
    print("  Loading test split...")
    test_ds  = load_split(args.data_dir, "test",  N_TEST  // len(CLASS_NAMES), args.seed)

    input_dim = train_ds[0][0].shape[0]
    n_classes = len(CLASS_NAMES)
    print(f"  input_dim={input_dim}, n_classes={n_classes}, device={args.device}")

    print(f"\n[seed={args.seed}] Training linear probe (max_epochs={args.max_epochs})...")
    model = train_linear_probe(
        train_ds, val_ds, input_dim, n_classes,
        seed=args.seed, max_epochs=args.max_epochs,
        lr=args.lr, batch_size=args.batch_size, device=args.device,
    )

    print(f"\n[seed={args.seed}] Evaluating on test set...")
    results = evaluate(model, test_ds, n_classes, args.batch_size, args.device)

    out_file = args.output_dir / "probe_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[seed={args.seed}] Results:")
    print(f"  AUROC macro = {results['linear_probe_auroc_macro']:.4f}")
    print(f"  Accuracy    = {results['linear_probe_accuracy']:.4f}")
    print(f"  F1 macro    = {results['linear_probe_f1_macro']:.4f}")
    print(f"  Saved to {out_file}")


if __name__ == "__main__":
    main()
