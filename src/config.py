"""Project-wide paths, seeds and constants.

Centralises every filesystem location and tunable so the notebook, the
Streamlit app and the test-suite share one source of truth.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- Native-runtime safety (must run before torch/lightgbm load their OpenMP) --
# LightGBM (Homebrew libomp) and torch each ship an OpenMP runtime; loading both
# in one process aborts with a duplicate-runtime segfault, especially under
# Rosetta. Allow the duplicate and cap threads to avoid over-subscription.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

# --- Reproducibility -------------------------------------------------------
SEED: int = 42


def set_seeds(seed: int = SEED) -> None:
    """Fix RNG seeds across python, numpy and (if present) torch.

    Args:
        seed: The integer seed to apply everywhere.

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional at import time (e.g. during light unit tests)
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - torch always present in full env
        pass


# --- Filesystem layout -----------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
OUTPUTS: Path = ROOT / "outputs"
FIGURES: Path = OUTPUTS / "figures"
METRICS_JSON: Path = OUTPUTS / "metrics.json"


def ensure_dirs() -> None:
    """Create all output/data directories if they do not yet exist."""
    for path in (DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


# --- Dataset identifiers ---------------------------------------------------
KAGGLE_DATASET: str = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
KAGGLE_ELLIPTIC: str = "ellipticco/elliptic-data-set"
HI_SMALL_TRANS: str = "HI-Small_Trans.csv"
HI_SMALL_PATTERNS: str = "HI-Small_Patterns.txt"


@dataclass(frozen=True)
class MotifWindows:
    """Time windows (in hours) used by the motif detectors."""

    fan_window_h: float = 24.0
    cycle_window_h: float = 168.0  # one week
    gather_scatter_window_h: float = 72.0


@dataclass(frozen=True)
class RunConfig:
    """Top-level knobs controlling a single MVP run.

    Attributes:
        sample_edges: Cap on transactions loaded for the MVP (keeps memory and
            wall-clock bounded, per the master-context "work on a subsample").
        temporal_train_frac: Fraction of the (time-sorted) edges used for
            training; the remainder is the strictly-later test set.
        min_fan_degree: Minimum out/in degree for a fan-out/fan-in motif.
        precision_at_k: k values reported for precision@k.
        fixed_recall: Recall level at which alert-reduction is reported.
        windows: Motif time windows.

    """

    sample_edges: int = 250_000
    temporal_train_frac: float = 0.7
    min_fan_degree: int = 3
    precision_at_k: tuple[int, ...] = (50, 100, 500, 1000)
    fixed_recall: float = 0.50
    windows: MotifWindows = field(default_factory=MotifWindows)


DEFAULT_RUN = RunConfig()
