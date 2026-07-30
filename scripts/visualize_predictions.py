"""Generate a grid image of example predictions for the README.

Picks a handful of test images (a mix of correct and, if any exist,
incorrect predictions), labels each with the true class and the model's
predicted probability, and saves the grid as a single PNG.

Example:
    python scripts/visualize_predictions.py \\
        --model-path artifacts/improved/model.keras \\
        --data-dir data/chest_xray \\
        --output docs/assets/example_predictions.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_cnn.config import CLASS_NAMES, TrainingConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument("--num-correct", type=int, default=6)
    parser.add_argument("--num-incorrect", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/example_predictions.png"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow.keras.models import load_model

    from pneumonia_cnn.data import _load_test_arrays

    config = TrainingConfig(data_dir=args.data_dir, image_size=(args.image_size, args.image_size))

    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)

    print(f"Loading test images from {config.test_dir}")
    images, labels = _load_test_arrays(config, cv2)

    predictions = model.predict(images).ravel()
    predicted_labels = np.round(predictions).astype(int)
    correct_mask = predicted_labels == labels

    rng = np.random.default_rng(args.seed)
    correct_idx = rng.choice(
        np.flatnonzero(correct_mask), size=min(args.num_correct, correct_mask.sum()), replace=False
    )
    incorrect_pool = np.flatnonzero(~correct_mask)
    incorrect_idx = rng.choice(
        incorrect_pool, size=min(args.num_incorrect, len(incorrect_pool)), replace=False
    )
    chosen = np.concatenate([correct_idx, incorrect_idx])
    rng.shuffle(chosen)

    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, chosen):
        image = images[idx]
        true_label = CLASS_NAMES[labels[idx]]
        pred_label = CLASS_NAMES[predicted_labels[idx]]
        probability = predictions[idx] if predicted_labels[idx] == 1 else 1 - predictions[idx]
        is_correct = labels[idx] == predicted_labels[idx]

        ax.imshow(image)
        ax.axis("off")
        color = "seagreen" if is_correct else "crimson"
        ax.set_title(
            f"true: {true_label}\npred: {pred_label} ({probability:.0%})",
            color=color,
            fontsize=10,
        )

    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle("Example test-set predictions (green = correct, red = incorrect)", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
