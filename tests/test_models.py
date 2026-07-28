"""Shape/sanity tests for every architecture in the model registry.

These tests deliberately pass ``weights=None`` (or omit it, for the two
from-scratch CNNs) so they build architectures without downloading
ImageNet weights -- that keeps CI fast and network-independent while still
catching shape bugs, typos in layer configs, and the VGG16/VGG19 mix-up
that existed in the original notebook.
"""

from __future__ import annotations

import pytest

from pneumonia_cnn.models import MODEL_REGISTRY, build_model

INPUT_SHAPE = (150, 150, 3)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_build_model_returns_binary_classifier(name):
    kwargs = {"weights": None} if name in {"vgg16", "vgg19", "resnet50"} else {}
    model = build_model(name, INPUT_SHAPE, **kwargs)

    assert model.input_shape[1:] == INPUT_SHAPE
    assert model.output_shape[1:] == (1,)
    assert model.layers[-1].activation.__name__ == "sigmoid"


def test_vgg16_and_vgg19_have_different_architectures():
    """Regression test for the original notebook's copy-paste bug, where
    the VGG-19 model was accidentally built on the VGG-16 base model.
    """

    vgg16 = build_model("vgg16", INPUT_SHAPE, weights=None)
    vgg19 = build_model("vgg19", INPUT_SHAPE, weights=None)

    assert vgg16.count_params() != vgg19.count_params()
    assert len(vgg16.layers) != len(vgg19.layers)


def test_transfer_models_freeze_backbone_by_default():
    vgg16 = build_model("vgg16", INPUT_SHAPE, weights=None, fine_tune=False)
    # Every layer except the final Dense head should be non-trainable.
    trainable_layers = [layer for layer in vgg16.layers if layer.trainable and layer.weights]
    assert len(trainable_layers) == 1


def test_unknown_model_name_raises():
    with pytest.raises(KeyError):
        build_model("not-a-real-model", INPUT_SHAPE)
