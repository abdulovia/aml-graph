"""Strictly temporal train/test splitting (no leakage).

The master context mandates: train on early timestamps, test on strictly later
ones. This module computes the split boundary and exposes helpers the leakage
test asserts against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalSplit:
    """A temporal split of edges by time.

    Attributes:
        boundary_t: The cutoff epoch-second; train edges have ``t <= boundary``,
            test edges have ``t > boundary``.
        train_idx: Row indices (into the time-sorted edge array) for training.
        test_idx: Row indices for testing.

    """

    boundary_t: float
    train_idx: np.ndarray
    test_idx: np.ndarray


def temporal_split(times: np.ndarray, train_frac: float) -> TemporalSplit:
    """Split edge indices by a time quantile so test is strictly later.

    Args:
        times: 1-D array of edge timestamps (epoch seconds).
        train_frac: Fraction of the timeline assigned to training.

    Returns:
        A :class:`TemporalSplit`. The boundary is the ``train_frac`` quantile of
        the timestamps; ties on the boundary go to *train* to guarantee the
        test set is strictly after ``boundary_t``.

    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    boundary = float(np.quantile(times, train_frac))
    train_idx = np.nonzero(times <= boundary)[0]
    test_idx = np.nonzero(times > boundary)[0]
    return TemporalSplit(boundary, train_idx, test_idx)


@dataclass
class TriTemporalSplit:
    """A temporal train/val/test split (tr < val < test in time).

    Attributes:
        tr_idx: Inner-train indices (earliest).
        val_idx: Validation indices (for early stopping and threshold choice).
        test_idx: Test indices (latest).

    """

    tr_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def temporal_tri_split(
    times: np.ndarray, train_frac: float, val_frac_of_train: float = 0.2
) -> TriTemporalSplit:
    """Split edges into strictly time-ordered inner-train / val / test slices.

    The test boundary is the ``train_frac`` quantile; within the resulting train
    portion, the latest ``val_frac_of_train`` (by time) becomes validation so the
    F1 threshold and early stopping never see the test set.

    Args:
        times: 1-D array of edge timestamps (epoch seconds).
        train_frac: Fraction of the timeline assigned to train (rest is test).
        val_frac_of_train: Fraction of the train timeline held out for validation.

    Returns:
        A :class:`TriTemporalSplit`; guarantees ``max(t_tr) <= min(t_val)`` and
        ``max(t_val) <= min(t_test)``.

    """
    base = temporal_split(times, train_frac)
    train_idx = base.train_idx
    val_boundary = float(np.quantile(times[train_idx], 1.0 - val_frac_of_train))
    tr_idx = train_idx[times[train_idx] <= val_boundary]
    val_idx = train_idx[times[train_idx] > val_boundary]
    return TriTemporalSplit(tr_idx, val_idx, base.test_idx)


def assert_no_leakage(train_times: np.ndarray, test_times: np.ndarray) -> None:
    """Raise ``AssertionError`` if any test edge is not strictly after train.

    This is the invariant enforced by ``tests/test_no_leakage.py``.
    """
    if test_times.size == 0 or train_times.size == 0:
        return
    assert train_times.max() <= test_times.min(), (
        f"Temporal leakage: max(train_t)={train_times.max()} > min(test_t)={test_times.min()}"
    )
