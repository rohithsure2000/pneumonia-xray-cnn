"""Shape/sanity tests for every architecture in the registry. weights=None
skips the ImageNet download so these run fast with no network access.
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
    # Regression test: VGG-19 used to get built on the VGG-16 base model
    # by mistake (copy-paste bug in the original notebook).
    vgg16 = build_model("vgg16", INPUT_SHAPE, weights=None)
    vgg19 = build_model("vgg19", INPUT_SHAPE, weights=None)

    assert vgg16.count_params() != vgg19.count_params()
    assert len(vgg16.layers) != len(vgg19.layers)


def test_improved_cnn_matches_reported_architecture():
    # Ties this to the project report's model-summary table (1,246,977
    # params) so it can't silently drift to a different architecture.
    model = build_model("improved", INPUT_SHAPE)

    assert model.count_params() == 1_246_977
    batch_norm_layers = [layer for layer in model.layers if "batch_normalization" in layer.name]
    dropout_layers = [layer for layer in model.layers if "dropout" in layer.name]
    assert len(batch_norm_layers) == 5
    assert len(dropout_layers) == 4


def test_transfer_models_freeze_backbone_by_default():
    vgg16 = build_model("vgg16", INPUT_SHAPE, weights=None, fine_tune=False)
    # Every layer except the final Dense head should be non-trainable.
    trainable_layers = [layer for layer in vgg16.layers if layer.trainable and layer.weights]
    assert len(trainable_layers) == 1


def test_unknown_model_name_raises():
    with pytest.raises(KeyError):
        build_model("not-a-real-model", INPUT_SHAPE)
