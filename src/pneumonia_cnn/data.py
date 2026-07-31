"""Data loading and augmentation, built on Keras's ImageDataGenerator --
same approach as the original notebook, wrapped behind a typed, testable
interface instead of inline notebook cells.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class Datasets:
    """Everything a training/evaluation run needs."""

    train_generator: "object"
    val_generator: "object"
    test_generator: "object"
    test_images: np.ndarray
    test_labels: np.ndarray


def _check_dataset_layout(config: TrainingConfig) -> None:
    # val/ is deliberately not required -- see build_generators() for why.
    missing = [str(d) for d in (config.train_dir, config.test_dir) if not d.is_dir()]
    if not missing:
        return

    # The Kaggle archive nests an extra chest_xray/ folder inside itself,
    # so this is worth checking for directly rather than a generic error.
    nested = config.data_dir / "chest_xray"
    hint = ""
    if (nested / "train").is_dir():
        hint = (
            f" Found train/ one level deeper at {nested} -- move its "
            f"contents up into {config.data_dir}, or pass --data-dir {nested}."
        )

    raise FileNotFoundError(
        "Expected a 'chest_xray' style dataset with train/val/test "
        f"folders, but could not find: {', '.join(missing)}.{hint} "
        "See the README's 'Getting the data' section."
    )


def build_generators(config: TrainingConfig) -> Datasets:
    """Build Keras data generators plus an in-memory test set.

    Validation comes from a split of the *training* folder rather than the
    dataset's official val/ folder, which only has 16 images -- too few for
    EarlyStopping/ReduceLROnPlateau to get a stable signal from. test/ is
    untouched by this and is what the reported metrics are based on.

    The test set is also loaded into memory as numpy arrays, since
    computing a confusion matrix needs a fixed, non-shuffled ordering.
    """

    # Imported lazily so `pneumonia_cnn.models` (and anything that only
    # needs the config/metrics helpers) can be imported without requiring
    # TensorFlow/OpenCV to be installed.
    import cv2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    _check_dataset_layout(config)

    # Separate generators so the training split gets augmented but the
    # held-out validation slice doesn't -- both use the same split/seed so
    # subset="training"/"validation" stay non-overlapping.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=config.rotation_range,
        zoom_range=config.zoom_range,
        horizontal_flip=config.horizontal_flip,
        vertical_flip=config.vertical_flip,
        validation_split=config.validation_split,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=config.validation_split)
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=str(config.train_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=config.seed,
    )
    val_generator = val_datagen.flow_from_directory(
        directory=str(config.train_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=config.seed,
    )
    test_generator = eval_datagen.flow_from_directory(
        directory=str(config.test_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode="binary",
        shuffle=False,
    )

    test_images, test_labels = _load_test_arrays(config, cv2)

    return Datasets(
        train_generator=train_generator,
        val_generator=val_generator,
        test_generator=test_generator,
        test_images=test_images,
        test_labels=test_labels,
    )


def _load_test_arrays(config: TrainingConfig, cv2) -> Tuple[np.ndarray, np.ndarray]:
    """Load every test image into memory for evaluation.

    Split out from build_generators() so tests can exercise the resize/
    normalize logic on a couple of synthetic images without the real
    dataset on disk.
    """

    images, labels = [], []
    for label_index, class_name in enumerate(("NORMAL", "PNEUMONIA")):
        class_dir = config.test_dir / class_name
        for path in sorted(class_dir.iterdir()):
            image = cv2.imread(str(path))
            image = cv2.resize(image, config.image_size)
            image = image.astype("float32") / 255.0
            images.append(image)
            labels.append(label_index)

    if not images:
        raise FileNotFoundError(f"No test images found under {config.test_dir}")

    return np.array(images), np.array(labels)


def preprocess_single_image(path: Path, config: TrainingConfig, cv2) -> np.ndarray:
    """Load and preprocess one image for inference (see predict.py)."""

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image at {path}")
    image = cv2.resize(image, config.image_size)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)
