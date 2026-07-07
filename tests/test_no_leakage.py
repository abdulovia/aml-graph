"""Tests guaranteeing the temporal split introduces no future-information leak."""

from __future__ import annotations

import numpy as np
import pytest

from src.splits import assert_no_leakage, temporal_split


def test_split_is_strictly_temporal():
    times = np.arange(100, dtype=float)
    split = temporal_split(times, train_frac=0.7)
    train_t = times[split.train_idx]
    test_t = times[split.test_idx]
    # every test timestamp must be strictly after every train timestamp
    assert train_t.max() <= split.boundary_t
    assert test_t.min() > split.boundary_t
    assert train_t.max() <= test_t.min()
    assert_no_leakage(train_t, test_t)  # must not raise


def test_split_covers_all_edges_without_overlap():
    times = np.random.default_rng(0).random(500)
    split = temporal_split(times, train_frac=0.6)
    idx = np.concatenate([split.train_idx, split.test_idx])
    assert sorted(idx.tolist()) == list(range(500))
    assert set(split.train_idx).isdisjoint(set(split.test_idx))


def test_assert_no_leakage_flags_overlap():
    train_t = np.array([0.0, 1.0, 5.0])
    test_t = np.array([4.0, 6.0])  # 4.0 < max(train)=5.0 -> leakage
    with pytest.raises(AssertionError):
        assert_no_leakage(train_t, test_t)


def test_invalid_train_frac_raises():
    with pytest.raises(ValueError):
        temporal_split(np.arange(10, dtype=float), train_frac=1.5)
