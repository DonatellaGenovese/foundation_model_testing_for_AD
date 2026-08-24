from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as L


class RawNpyDataModule(L.LightningDataModule):
    """
    DataModule that loads preprocessed .npy shards directly from
    train/ val/ test/ directories (each containing <class_folder>/*_x.npy).

    Training DataLoader contains only normal classes; val/test contain
    normal + anomaly classes.
    """

    def __init__(self, preprocessed_dir: Path,
                 normal_classes: list, anomaly_classes: list,
                 n_train: int = 100_000, n_val: int = 20_000, n_test: int = 20_000,
                 batch_size: int = 512, seed: int = 42,
                 class_folders: dict = None):
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.normal_classes   = normal_classes
        self.anomaly_classes  = anomaly_classes
        self.n_train, self.n_val, self.n_test = n_train, n_val, n_test
        self.batch_size = batch_size
        self.seed = seed
        self.input_dim: int = None
        self.CLASS_FOLDERS: dict = class_folders or {}

    def _probe_input_dim(self) -> int:
        for cls in self.normal_classes:
            cls_dir = self.preprocessed_dir / "train" / self.CLASS_FOLDERS[cls]
            first = next(cls_dir.glob("*_x.npy"), None)
            if first is not None:
                return int(np.load(first, mmap_mode="r").shape[1])
        raise RuntimeError("Could not probe input dimension from preprocessed data.")

    def _load_split(self, split: str, cls_id: int, n_max: int) -> np.ndarray:
        folder = self.CLASS_FOLDERS[cls_id]
        cls_dir = self.preprocessed_dir / split / folder
        if not cls_dir.exists():
            return np.zeros((0,), dtype=np.float32)
        x_files = sorted(cls_dir.glob("*_x.npy"))
        if not x_files:
            return np.zeros((0,), dtype=np.float32)
        xs = [np.load(f) for f in x_files]
        X = np.concatenate(xs, axis=0)
        rng = np.random.default_rng(self.seed)
        n = min(n_max, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx]

    def _make_dataset(self, split: str, classes: list, n_per_class: int):
        xs, ys = [], []
        for cls in classes:
            X = self._load_split(split, cls, n_per_class)
            if len(X) == 0:
                print(f"    Warning: no data for class {cls} in {split}")
                continue
            print(f"    {split}/{self.CLASS_FOLDERS[cls]}: {len(X)} events, dim={X.shape[1]}")
            if self.input_dim is None:
                self.input_dim = X.shape[1]
            xs.append(torch.from_numpy(X).float())
            ys.append(torch.full((len(X),), cls, dtype=torch.long))
        if not xs:
            return None
        return TensorDataset(torch.cat(xs), torch.cat(ys))

    def setup(self, stage=None):
        needs_train_val = stage in (None, "fit", "validate")
        needs_test = stage in (None, "test", "predict")
        if needs_train_val and not hasattr(self, "train_ds"):
            print("  [RawNpyDataModule] Loading train split (normal classes only)...")
            self.train_ds = self._make_dataset("train", self.normal_classes, self.n_train)
        if needs_train_val and not hasattr(self, "val_ds"):
            print("  [RawNpyDataModule] Loading val split (normal + anomaly)...")
            self.val_ds = self._make_dataset("val", self.normal_classes + self.anomaly_classes, self.n_val)
        if needs_test and not hasattr(self, "test_ds"):
            print("  [RawNpyDataModule] Loading test split (normal + anomaly)...")
            self.test_ds = self._make_dataset("test", self.normal_classes + self.anomaly_classes, self.n_test)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    def train_dataloader(self): return self._loader(self.train_ds, shuffle=True)
    def val_dataloader(self):   return self._loader(self.val_ds,   shuffle=False)
    def test_dataloader(self):  return self._loader(self.test_ds,  shuffle=False)
