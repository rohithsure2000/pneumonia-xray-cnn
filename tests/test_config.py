from pathlib import Path

from pneumonia_cnn.config import TrainingConfig


def test_derived_paths_and_shape():
    config = TrainingConfig(data_dir="data/chest_xray", image_size=(150, 150))

    assert config.train_dir == Path("data/chest_xray/train")
    assert config.val_dir == Path("data/chest_xray/val")
    assert config.test_dir == Path("data/chest_xray/test")
    assert config.input_shape == (150, 150, 3)


def test_string_paths_are_coerced_to_path_objects():
    config = TrainingConfig(data_dir="some/string/path", output_dir="another/string")

    assert isinstance(config.data_dir, Path)
    assert isinstance(config.output_dir, Path)
