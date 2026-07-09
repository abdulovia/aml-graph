"""Evaluation metrics tailored to extreme class imbalance (AML).

Primary metrics per the master context: minority-class F1 and PR-AUC. Never
report accuracy/ROC-AUC as headline numbers on <1% prevalence.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve


def minority_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 of the positive (illicit) class."""
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, y_score))


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return the threshold maximising minority-F1 and that F1 value.

    Chosen on the given (train/val) scores; applied unchanged to test to avoid
    tuning on the test set.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    # precision_recall_curve returns len(thr) == len(prec) - 1
    if thr.size == 0:
        return 0.5, 0.0
    best = int(np.argmax(f1[:-1])) if f1.size > 1 else 0
    return float(thr[best]), float(f1[best])


def precision_at_k(
    y_true: np.ndarray, y_score: np.ndarray, ks: tuple[int, ...]
) -> dict[str, float]:
    """Precision within the top-k highest-scored alerts (analyst queue)."""
    order = np.argsort(-y_score)
    out: dict[str, float] = {}
    for k in ks:
        k_eff = min(k, order.size)
        if k_eff == 0:
            out[f"p@{k}"] = 0.0
            continue
        top = order[:k_eff]
        out[f"p@{k}"] = float(y_true[top].sum() / k_eff)
    return out


def alert_reduction_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float
) -> dict[str, float]:
    """Alert-reduction achievable while holding a fixed recall.

    Returns the fraction of the population that still needs review to reach
    ``target_recall`` and the implied reduction versus reviewing everything.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # rec is decreasing over thresholds; find the highest threshold that still
    # attains at least the target recall.
    order = np.argsort(-y_score)
    n = order.size
    positives = int(y_true.sum())
    if positives == 0:
        return {"target_recall": target_recall, "review_fraction": 1.0, "alert_reduction": 0.0}
    cum_tp = np.cumsum(y_true[order])
    needed = np.searchsorted(cum_tp, int(np.ceil(target_recall * positives)))
    needed = min(needed + 1, n)
    review_fraction = needed / n
    _ = (prec, rec, thr)
    return {
        "target_recall": target_recall,
        "review_fraction": float(review_fraction),
        "alert_reduction": float(1.0 - review_fraction),
    }


def recall_by_motif(
    edge_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    motif_edge_map: dict[str, set[int]],
) -> dict[str, float]:
    """Recall of illicit edges restricted to each motif type.

    Args:
        edge_ids: Edge ids aligned with ``y_true``/``y_pred``.
        y_true: Ground-truth illicit labels.
        y_pred: Binary predictions.
        motif_edge_map: motif_type -> set of edge_ids belonging to that motif.

    Returns:
        motif_type -> recall over illicit edges of that motif (NaN-free; 0 if no
        illicit edges of that type are present in the evaluated set).

    """
    out: dict[str, float] = {}
    eid_to_pos = {int(e): i for i, e in enumerate(edge_ids)}
    for motif, eids in motif_edge_map.items():
        idx = [eid_to_pos[e] for e in eids if e in eid_to_pos]
        if not idx:
            out[motif] = 0.0
            continue
        idx_arr = np.array(idx)
        illicit = y_true[idx_arr] == 1
        denom = int(illicit.sum())
        if denom == 0:
            out[motif] = 0.0
            continue
        hit = int(((y_pred[idx_arr] == 1) & illicit).sum())
        out[motif] = hit / denom
    return out


def summarise(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    ks: tuple[int, ...],
    fixed_recall: float,
) -> dict[str, object]:
    """Bundle all headline metrics for one model into a JSON-serialisable dict."""
    y_pred = (y_score >= threshold).astype(int)
    result: dict[str, object] = {
        "model": name,
        "threshold": float(threshold),
        "minority_f1": minority_f1(y_true, y_pred),
        "pr_auc": pr_auc(y_true, y_score),
        "n_test": int(y_true.size),
        "n_illicit": int(y_true.sum()),
    }
    result.update(precision_at_k(y_true, y_score, ks))
    result.update(alert_reduction_at_recall(y_true, y_score, fixed_recall))
    return result
