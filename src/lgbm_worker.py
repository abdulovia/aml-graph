"""Subprocess worker that trains LightGBM in isolation.

Run as ``python -m src.lgbm_worker <input.npz> <output.npz>``.

Why a subprocess: LightGBM (Homebrew ``libomp``) and PyTorch each bundle their
own OpenMP runtime. Loading both in one process segfaults under Rosetta. Keeping
LightGBM in its own process means the main (PyTorch/GNN) process never loads a
second OpenMP runtime. Input npz carries ``x_train``/``y_train``/``x_test``;
output npz carries ``scores`` (positive-class probabilities for the test rows).
"""

from __future__ import annotations

import sys

import numpy as np

from src.config import SEED


def main() -> None:
    """Train a class-balanced LightGBM and write test-set scores."""
    import lightgbm as lgb  # imported only here, never in the main process

    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=2,
        verbose=-1,
    )
    model.fit(data["x_train"], data["y_train"])
    scores = model.predict_proba(data["x_test"])[:, 1]
    np.savez(out_path, scores=scores.astype(np.float64))


if __name__ == "__main__":
    main()
