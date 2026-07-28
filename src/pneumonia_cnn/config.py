"""Central configuration for the pneumonia CNN project.

Keeping every tunable value in one dataclass makes it trivial to reproduce
a run (just print the config) and to override values from the command
line without threading a dozen function arguments through the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class TrainingConfig:
    """Hyperparameters and paths shared by every training/eval script.

    Attributes:
        data_dir: Root of the extracted ``chest_xray`` Kaggle dataset. Must
            contain ``train/``, ``val/`` and ``test/`` sub-directories, each
            with ``NORMAL/`` and ``PNEUMONIA/`` class folders.
        image_size: Height/width the images are resized to before being fed
            to a model. 150x150 matches the original experiments; 224x224
            is a common choice if you swap in a different backbone.
        batch_size: Mini-batch size used for both training and evaluation.
        epochs: Number of passes over the training set.
        learning_rate: Initial learning rate for the Adam optimizer.
        seed: Random seed applied to numpy/tensorflow for reproducibility.
            The original exploratory notebook did not fix a seed, which is
            part of why re-running it produces slightly different numbers
            each time -- fixing it here is a deliberate improvement.
        output_dir: Where checkpoints, logs, and evaluation reports land.
    """

    data_dir: Path = Path("data/chest_xray")
    image_size: Tuple[int, int] = (150, 150)
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    output_dir: Path = Path("artifacts")

    # Data augmentation, applied to the training split only.
    rotation_range: int = 30
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = True

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (*self.image_size, 3)

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.data_dir / "val"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
