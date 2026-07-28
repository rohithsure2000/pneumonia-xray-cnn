"""Tests for metric computation.

The confusion-matrix numbers below are taken directly from the Basic CNN
evaluation cell in the original exploratory notebook ([[190, 44], [4, 386]])
so this test doubles as a regression check that ``compute_metrics`` matches
numbers we already know are correct by hand.
"""

from __future__ import annotations

import numpy as np

from pneumonia_cnn.utils import compute_metrics


def test_compute_metrics_matches_known_confusion_matrix():
    # 190 true negatives, 44 false positives, 4 false negatives, 386 true positives.
    y_true = np.array([0] * (190 + 44) + [1] * (4 + 386))
    y_pred_proba = np.array(
        [0] * 190 + [1] * 44  # NORMAL class: 190 correct, 44 misclassified
        + [0] * 4 + [1] * 386  # PNEUMONIA class: 4 misclassified, 386 correct
    )

    metrics = compute_metrics(y_true, y_pred_proba)

    assert metrics.true_negatives == 190
    assert metrics.false_positives == 44
    assert metrics.false_negatives == 4
    assert metrics.true_positives == 386
    assert round(metrics.accuracy * 100, 2) == 92.31
    assert round(metrics.precision * 100, 2) == 89.77
    assert round(metrics.recall * 100, 2) == 98.97


def test_compute_metrics_handles_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.01, 0.02, 0.98, 0.99])

    metrics = compute_metrics(y_true, y_pred_proba)

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
