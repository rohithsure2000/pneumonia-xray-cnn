"""Run inference on a single chest X-ray image.

Example:
    python -m pneumonia_cnn.predict --model-path artifacts/improved/model.keras \\
        --image path/to/xray.jpeg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import CLASS_NAMES, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold above which the image is classified PNEUMONIA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(image_size=(args.image_size, args.image_size))

    import cv2
    from tensorflow.keras.models import load_model

    from .data import preprocess_single_image

    model = load_model(args.model_path)
    batch = preprocess_single_image(args.image, config, cv2)
    probability = float(model.predict(batch)[0][0])
    label = CLASS_NAMES[1] if probability >= args.threshold else CLASS_NAMES[0]

    print(f"Prediction: {label}")
    print(f"Pneumonia probability: {probability:.4f}")


if __name__ == "__main__":
    main()
