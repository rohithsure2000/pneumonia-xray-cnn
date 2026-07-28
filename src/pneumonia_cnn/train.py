"""Train one of the five CNN architectures on the chest X-ray dataset.

Example:
    python -m pneumonia_cnn.train --model improved --epochs 15 \\
        --data-dir data/chest_xray --output-dir artifacts/improved
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import TrainingConfig
from .models import MODEL_REGISTRY, build_model
from .utils import compute_metrics, plot_training_history, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=150, help="Square side length in pixels.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="For transfer-learning models, unfreeze the backbone instead of "
        "training only the classification head.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir / args.model,
    )
    set_seed(config.seed)

    # Imported here so `--help` works without TensorFlow installed.
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    from .data import build_generators

    logger.info("Loading dataset from %s", config.data_dir)
    datasets = build_generators(config)

    logger.info("Building model '%s'", args.model)
    model_kwargs = {"fine_tune": args.fine_tune} if args.model in {"vgg16", "vgg19", "resnet50"} else {}
    model = build_model(args.model, config.input_shape, **model_kwargs)
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "best_weights.weights.h5"
    callbacks = [
        ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, save_weights_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    steps_per_epoch = datasets.train_generator.samples // config.batch_size
    validation_steps = max(1, datasets.val_generator.samples // config.batch_size)

    history = model.fit(
        datasets.train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config.epochs,
        validation_data=datasets.val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    plot_training_history(history, config.output_dir / "training_curves.png")

    logger.info("Evaluating on held-out test set")
    predictions = model.predict(datasets.test_images)
    metrics = compute_metrics(datasets.test_labels, predictions)
    metrics.save(config.output_dir / "test_metrics.json")
    logger.info("Test accuracy: %.4f | precision: %.4f | recall: %.4f | F1: %.4f",
                metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score)

    final_model_path = config.output_dir / "model.keras"
    model.save(final_model_path)
    logger.info("Saved trained model to %s", final_model_path)


if __name__ == "__main__":
    main()
