"""The 5 CNN architectures compared in this project: basic CNN, improved
CNN (BatchNorm + Dropout), VGG-16, VGG-19, ResNet-50.

A few things were fixed here relative to the original notebook (details in
docs/results.md):
- Improved CNN now matches the project report's architecture exactly. An
  earlier version of this used a different, deeper network by mistake.
- VGG-19 builds its own base model instead of reusing VGG-16's, which was
  a copy-paste bug in the original.
- Backbones freeze consistently via `base_model.trainable = False`,
  instead of "freeze the first 20 layers," which silently didn't scale to
  ResNet-50 (~175 layers vs. VGG's ~19).

`weights=None` skips the ImageNet download -- used by the tests so they
don't need network access. TensorFlow is imported inside each function so
this module (and MODEL_REGISTRY) can be imported without TF installed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


def build_basic_cnn(input_shape: Tuple[int, int, int], weights: "str | None" = None) -> Any:
    """5-block Conv2D/MaxPool stack, no regularization. Baseline model."""

    del weights  # unused; kept for a uniform builder signature
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(name="basic_cnn")
    model.add(layers.Input(shape=input_shape))
    filters = (32, 64, 64, 128, 256)
    for f in filters:
        model.add(layers.Conv2D(f, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D((2, 2), padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def build_improved_cnn(input_shape: Tuple[int, int, int], weights: "str | None" = None) -> Any:
    """Same 5 Conv2D blocks as the basic CNN, plus BatchNorm after each
    block and Dropout after 3 of them. Matches the project report's
    architecture param-for-param (1,246,977 total).
    """

    del weights
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(name="improved_cnn")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding="same"))

    model.add(layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding="same"))

    model.add(layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding="same"))

    model.add(layers.Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding="same"))

    model.add(layers.Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding="same"))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def _build_transfer_model(
    base_model_fn: Callable,
    input_shape: Tuple[int, int, int],
    weights: "str | None",
    name: str,
    fine_tune: bool = False,
) -> Any:
    from tensorflow import keras
    from tensorflow.keras import layers

    base_model = base_model_fn(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = fine_tune

    inputs = base_model.input
    x = layers.Flatten()(base_model.output)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=name)


def build_vgg16(
    input_shape: Tuple[int, int, int], weights: "str | None" = "imagenet", fine_tune: bool = False
) -> Any:
    from tensorflow.keras.applications import VGG16

    return _build_transfer_model(VGG16, input_shape, weights, "vgg16_transfer", fine_tune)


def build_vgg19(
    input_shape: Tuple[int, int, int], weights: "str | None" = "imagenet", fine_tune: bool = False
) -> Any:
    from tensorflow.keras.applications import VGG19

    return _build_transfer_model(VGG19, input_shape, weights, "vgg19_transfer", fine_tune)


def build_resnet50(
    input_shape: Tuple[int, int, int], weights: "str | None" = "imagenet", fine_tune: bool = False
) -> Any:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import ResNet50

    base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = fine_tune

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(base_model.input, outputs, name="resnet50_transfer")


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "basic": build_basic_cnn,
    "improved": build_improved_cnn,
    "vgg16": build_vgg16,
    "vgg19": build_vgg19,
    "resnet50": build_resnet50,
}


def build_model(name: str, input_shape: Tuple[int, int, int], **kwargs) -> Any:
    """Build a model by its registry name. Raises KeyError if unknown."""

    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Valid options: {valid}") from exc
    return builder(input_shape, **kwargs)
