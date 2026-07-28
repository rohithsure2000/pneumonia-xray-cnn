"""Evaluate a saved model against the test split and print/save a report.

Example:
    python -m pneumonia_cnn.evaluate --model-path artifacts/improved/model.keras \\
        --data-dir data/chest_xray
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import TrainingConfig
from .utils import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(data_dir=args.data_dir, image_size=(args.image_size, args.image_size))

    from tensorflow.keras.models import load_model

    from .data import _load_test_arrays
    import cv2

    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)

    logger.info("Loading test images from %s", config.test_dir)
    test_images, test_labels = _load_test_arrays(config, cv2)

    predictions = model.predict(test_images)
    metrics = compute_metrics(test_labels, predictions)

    report = metrics.as_dict()
    print(json.dumps(report, indent=2))

    if args.output:
        metrics.save(args.output)
        logger.info("Wrote metrics to %s", args.output)


if __name__ == "__main__":
    main()
