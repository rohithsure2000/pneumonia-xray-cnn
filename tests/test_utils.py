"""Tests for metric computation. The confusion matrix in the first test
is from the Basic CNN's evaluation in the original notebook
([[190, 44], [4, 386]]) -- doubles as a check against known-good numbers.
"""

from __future__ import annotations

import numpy as np

from pneumonia_cnn.utils import compute_metrics


def test_compute_metrics_matches_known_confusion_matrix():
    # NORMAL class (label 0): 190 correct, 44 misclassified as PNEUMONIA
    # PNEUMONIA class (label 1): 4 misclassified as NORMAL, 386 correct
    y_true = np.array([0] * (190 + 44) + [1] * (4 + 386))
    y_pred_proba = np.array([0] * 190 + [1] * 44 + [0] * 4 + [1] * 386)

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
