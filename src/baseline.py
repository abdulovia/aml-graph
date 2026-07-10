"""LightGBM baseline on hand-crafted graph/motif edge features.

Training runs in a **subprocess** (``src.lgbm_worker``) so LightGBM's OpenMP
runtime never coexists with PyTorch's in one process — that combination
segfaults under Rosetta. The main process here only marshals arrays to/from the
worker and never imports ``lightgbm``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from src import config


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[None, np.ndarray]:
    """Train a class-balanced LightGBM in a subprocess and score the test set.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels.
        x_test: Test feature matrix.

    Returns:
        ``(None, scores)`` — the model stays in the worker; only the positive-
        class probabilities on ``x_test`` are returned.

    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.npz"
        out_path = Path(tmp) / "out.npz"
        np.savez(
            in_path,
            x_train=x_train.astype(np.float32),
            y_train=y_train.astype(np.int64),
            x_test=x_test.astype(np.float32),
        )
        subprocess.run(
            [sys.executable, "-m", "src.lgbm_worker", str(in_path), str(out_path)],
            check=True,
            cwd=str(config.ROOT),
        )
        scores = np.load(out_path)["scores"]
    return None, scores
