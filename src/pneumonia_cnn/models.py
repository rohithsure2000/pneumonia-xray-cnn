"""CNN architectures compared in this project.

Five architectures are implemented, matching the project report's own
comparison: a basic CNN, an "Improved CNN" (BatchNorm + Dropout), and
three transfer-learning models (VGG-16, VGG-19, ResNet-50).

Three correctness fixes relative to the original exploratory notebook are
worth calling out explicitly, since they change what "Improved CNN" and
"VGG-19" actually mean here:

1. **"Improved CNN" was rebuilt to match the project report, not an
   unrelated experiment.** An earlier version of this module implemented a
   deeper SeparableConv2D architecture under this name. That network does
   exist in the original notebook, but as an unlabeled second experiment
   (never part of the report's five compared architectures) -- using it
   here was a mismatch with the source material. ``build_improved_cnn``
   now matches the report's own model-summary table param-for-param.
2. **VGG-19 was accidentally built on the VGG-16 backbone.** The original
   notebook instantiated a ``VGG19`` base model but then built the
   classification head on top of the *previous* ``VGG16`` variable by
   mistake, so the two "different" transfer-learning models were actually
   identical networks with independently-initialized dense heads. This
   module builds each backbone from its own base model.
3. **Partial layer-freezing didn't scale across backbones.** The notebook
   froze "the first 20 layers" for every transfer-learning model. VGG-16
   only has ~19 layers, so that happened to freeze the whole backbone --
   but ResNet-50 has ~175 layers, so freezing just 20 left the vast
   majority of it trainable, which is closer to full fine-tuning than
   feature extraction. Here every backbone freezes consistently
   (``base_model.trainable = False``) unless the caller asks to fine-tune.

All builder functions share the signature
``build_x(input_shape, weights="imagenet") -> keras.Model`` so they can be
swapped in and out from a single registry (see ``MODEL_REGISTRY`` below).
Passing ``weights=None`` skips the ImageNet download entirely, which is
what the unit tests use to check output shapes without any network access.

TensorFlow is imported lazily inside each function (rather than at module
level) so that ``import pneumonia_cnn`` and ``MODEL_REGISTRY`` lookups work
even in environments that only need the CLI's ``--help`` output or the
config/metrics helpers, without requiring TensorFlow to be installed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


def build_basic_cnn(input_shape: Tuple[int, int, int], weights: "str | None" = None) -> Any:
    """A plain 5-block Conv2D/MaxPool stack with no regularization.

    This is the "Basic CNN" baseline: it trains fast and establishes a
    lower bound for what the more sophisticated architectures need to beat.
    """

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
    """The "Improved CNN" as actually defined and reported in the project.

    Five Conv2D blocks (32/64/64/128/256 filters), each followed by
    BatchNormalization, with Dropout inserted after the 2nd, 4th, and 5th
    blocks and once more after the dense layer -- 5 BatchNormalization and
    4 Dropout layers total, matching the project report's own description
    ("we added 5 Batch Normalization and 4 Dropout layers") and its
    model-summary table param-for-param (1,246,977 total parameters).

    This replaces an earlier version of this function that implemented a
    different, deeper SeparableConv2D architecture. That architecture does
    exist in the original exploratory notebook, but as an unlabeled second
    experiment -- it was never one of the report's five compared
    architectures, so using it here under the name "Improved CNN" was a
    mismatch with the source material worth correcting.
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
    """Look up and construct a model by its registry name.

    Raises:
        KeyError: if ``name`` isn't one of ``MODEL_REGISTRY``.
    """

    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Valid options: {valid}") from exc
    return builder(input_shape, **kwargs)
