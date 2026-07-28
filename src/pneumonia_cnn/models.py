"""CNN architectures compared in this project.

Five architectures are implemented, matching the comparison described in
the project write-up: a basic CNN, a more heavily regularized "improved"
CNN, and three transfer-learning models (VGG-16, VGG-19, ResNet-50).

Two correctness fixes relative to the original exploratory notebook are
worth calling out explicitly, since they change what "VGG-19" and
"Improved CNN" actually mean:

1. **VGG-19 was accidentally built on the VGG-16 backbone.** The original
   notebook instantiated a ``VGG19`` base model but then built the
   classification head on top of the *previous* ``VGG16`` variable by
   mistake, so the two "different" transfer-learning models were actually
   identical networks with independently-initialized dense heads. This
   module builds each backbone from its own base model.
2. **Partial layer-freezing didn't scale across backbones.** The notebook
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
    """A deeper, regularized CNN using separable convolutions.

    Batch normalization and progressively increasing dropout are used to
    fight overfitting, which the basic CNN is prone to given the modest
    training set size and class imbalance.
    """

    del weights
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2, 2))(x)

    for filters in (32, 64, 128, 256):
        x = layers.SeparableConv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.SeparableConv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D((2, 2))(x)
        if filters >= 128:
            x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="improved_cnn")


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
