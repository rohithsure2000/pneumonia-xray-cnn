"""Small shared helpers: reproducibility, metrics, and plotting."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed python/numpy/tensorflow RNGs for a reproducible run."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        logger.debug("TensorFlow not installed; skipping tf seed.")


@dataclass
class ClassificationMetrics:
    """Confusion-matrix-derived metrics for a binary classifier."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> ClassificationMetrics:
    """Accuracy/precision/recall/F1 from predicted probabilities, rounded
    to a hard 0/1 label at the standard 0.5 threshold.
    """

    from sklearn.metrics import confusion_matrix

    y_pred = np.round(y_pred_proba).astype(int).ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return ClassificationMetrics(
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
    )


def plot_training_history(history, output_path: Path) -> None:
    """Save accuracy/loss curves for a completed training run."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="val")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
