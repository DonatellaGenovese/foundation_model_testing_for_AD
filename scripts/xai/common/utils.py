"""Small shared utilities for scripts/xai entry points."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def setup_imports() -> Path:
    """Put scripts/xai and project root on sys.path. Returns project root."""
    xai_dir = Path(__file__).resolve().parent
    root = xai_dir.parents[1]
    for p in (str(xai_dir), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
