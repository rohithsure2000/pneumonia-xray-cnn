"""Central config for the project. One dataclass instead of threading a
dozen args through every script, and makes a run trivial to reproduce
(just print the config).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class TrainingConfig:
    """Hyperparameters and paths shared by the training/eval scripts.

    data_dir needs train/, val/, test/ subfolders, each with NORMAL/ and
    PNEUMONIA/ class folders (val/ is unused -- see validation_split).

    validation_split carves validation out of the *training* folder instead
    of using the dataset's own val/, which only has 16 images -- too few
    for EarlyStopping to get a stable signal from. test/ is untouched and
    is what final accuracy/precision/recall/F1 are based on.

    seed fixes numpy/tensorflow randomness for reproducibility.
    """

    data_dir: Path = Path("data/chest_xray")
    image_size: Tuple[int, int] = (150, 150)
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    output_dir: Path = Path("artifacts")
    validation_split: float = 0.15

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
