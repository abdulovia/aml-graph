"""Unit tests for the imbalance-aware metrics and the narrative template."""

from __future__ import annotations

import numpy as np

from src.metrics import alert_reduction_at_recall, best_f1_threshold, precision_at_k
from src.narrative import ChainFacts, template_narrative


def test_precision_at_k_perfect_and_random():
    y = np.array([1, 1, 0, 0, 0, 0])
    scores_perfect = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05])
    out = precision_at_k(y, scores_perfect, ks=(2, 4))
    assert out["p@2"] == 1.0
    assert out["p@4"] == 0.5  # both positives inside top-4 of 6


def test_alert_reduction_at_recall():
    # 100 edges, 10 positives ranked at the very top -> to reach recall 0.5 we
    # only need to review the first 5+1 edges => ~94% reduction.
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    scores = np.linspace(1.0, 0.0, 100)
    out = alert_reduction_at_recall(y, scores, target_recall=0.5)
    assert out["review_fraction"] <= 0.06
    assert out["alert_reduction"] >= 0.94
    # degenerate case: no positives -> nothing can be reduced
    none = alert_reduction_at_recall(np.zeros(10, dtype=int), np.arange(10.0), 0.5)
    assert none["alert_reduction"] == 0.0


def test_best_f1_threshold_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    thr, f1 = best_f1_threshold(y, scores)
    assert 0.3 < thr <= 0.8
    assert f1 == 1.0


def test_template_narrative_contains_facts_and_recommendation():
    facts = ChainFacts(
        motif_type="fan_out",
        n_accounts=7,
        n_transactions=12,
        total_amount=1_234_567.0,
        span_hours=36.5,
        confidence=0.91,
        focus_account="001_ABC",
    )
    text = template_narrative(facts)
    assert "001_ABC" in text and "fan-out" in text
    assert "FILE SAR" in text  # confidence >= 0.5
    low = template_narrative(
        ChainFacts("cycle", 3, 3, 10.0, 1.0, confidence=0.2, focus_account="X")
    )
    assert "ENHANCED REVIEW" in low
