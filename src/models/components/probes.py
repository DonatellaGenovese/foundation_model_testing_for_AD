from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification import (
    Accuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassROC,
)


class LinearProbe(LightningModule):
    """Linear classifier on frozen embeddings to evaluate embedding quality."""

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_classes: int,
        class_names: Optional[list] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.test_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)
        self.test_f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None)
        self.test_auroc_macro = MulticlassAUROC(num_classes=num_classes, average='macro')
        self.test_roc = MulticlassROC(num_classes=num_classes, average=None)

        self.val_acc_best = MaxMetric()
        self.cached_test_metrics = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.encoder.get_embeddings(x)
        return self.classifier(embeddings)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(torch.argmax(logits, dim=1), y)
        self.log("probe/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("probe/train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(torch.argmax(logits, dim=1), y)
        self.log("probe/val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("probe/val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.test_f1_per_class(preds, y)
        self.test_f1_macro(preds, y)
        self.test_auroc_per_class(probs, y)
        self.test_auroc_macro(probs, y)
        self.test_roc.update(probs, y)
        self.log("probe/test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("probe/test_f1_macro", self.test_f1_macro, on_step=False, on_epoch=True)
        self.log("probe/test_auroc_macro", self.test_auroc_macro, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("probe/val_acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_end(self):
        accuracy = self.test_acc.compute()
        f1_macro = self.test_f1_macro.compute()
        f1_per_class = self.test_f1_per_class.compute()
        auroc_macro = self.test_auroc_macro.compute()
        auroc_per_class = self.test_auroc_per_class.compute()
        fprs, tprs, thresholds = self.test_roc.compute()
        self.cached_test_metrics = {
            'accuracy': accuracy.cpu().numpy(),
            'f1_macro': f1_macro.cpu().numpy(),
            'f1_per_class': f1_per_class.cpu().numpy(),
            'auroc_macro': auroc_macro.cpu().numpy(),
            'auroc_per_class': auroc_per_class.cpu().numpy(),
            'fprs': [fpr.cpu().numpy() for fpr in fprs],
            'tprs': [tpr.cpu().numpy() for tpr in tprs],
            'thresholds': [th.cpu().numpy() for th in thresholds],
        }
        for i in range(len(f1_per_class)):
            self.log(f"probe/test_f1_{self.class_names[i]}", f1_per_class[i])
            self.log(f"probe/test_auroc_{self.class_names[i]}", auroc_per_class[i])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def get_detailed_metrics(self) -> Dict[str, Any]:
        if self.cached_test_metrics is None:
            raise RuntimeError("No cached test metrics. Run trainer.test() first.")
        return self.cached_test_metrics

    def plot_roc_curves(self, output_dir: Path) -> None:
        import matplotlib.pyplot as plt
        if self.cached_test_metrics is None:
            raise RuntimeError("No cached test metrics. Run trainer.test() first.")
        fprs = self.cached_test_metrics['fprs']
        tprs = self.cached_test_metrics['tprs']
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for c in range(len(fprs)):
            plt.figure()
            plt.plot(fprs[c], tprs[c])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC – {self.class_names[c]}")
            plt.grid(True)
            plt.savefig(output_dir / f"class_{self.class_names[c]}_roc.png")
            plt.close()
        plt.figure(figsize=(8, 6))
        for c in range(len(fprs)):
            plt.plot(fprs[c], tprs[c], label=self.class_names[c], linewidth=1.5)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (All Classes) - Linear Probe")
        plt.legend(fontsize="small", ncol=2)
        plt.grid(True)
        plt.savefig(output_dir / "roc_all_classes.png", bbox_inches="tight")
        plt.close()

