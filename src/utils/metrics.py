"""
utils/metrics.py

Metric computation utilities: AUC, ACC, AP, per-generator breakdown.
Thin wrappers around sklearn that also handle edge cases.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(
    labels: np.ndarray | list,
    probs: np.ndarray | list,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute AUC, ACC, AP and related metrics.
    labels: ground truth binary (0=real, 1=fake)
    probs:  predicted fake probability in [0, 1]
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    has_both_classes = len(np.unique(labels)) > 1

    auc = float(roc_auc_score(labels, probs)) if has_both_classes else 0.0
    acc = float(accuracy_score(labels, preds))
    ap = float(average_precision_score(labels, probs)) if has_both_classes else 0.0

    # True positives, false positives, etc.
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "auc": round(auc, 4),
        "acc": round(acc, 4),
        "ap": round(ap, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": len(labels),
    }


def compute_optimal_threshold(
    labels: np.ndarray, probs: np.ndarray
) -> tuple[float, dict]:
    """
    Find the threshold that maximises the F1 score.
    Returns (threshold, metrics_at_threshold).
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = float(thresholds[best_idx])
    return best_thresh, compute_metrics(labels, probs, threshold=best_thresh)


def per_generator_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    generators: list[str],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """
    Break down metrics by generator name.
    generators: list of generator names aligned with labels/probs.
    """
    generators = np.asarray(generators)
    results = {}
    for gen in np.unique(generators):
        mask = generators == gen
        results[gen] = compute_metrics(labels[mask], probs[mask], threshold)
    return results
