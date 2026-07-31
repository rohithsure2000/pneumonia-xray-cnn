# Pneumonia Detection from Pediatric Chest X-Rays with CNNs

[![CI](https://github.com/rohithsure2000/pneumonia-xray-cnn/actions/workflows/ci.yml/badge.svg)](https://github.com/rohithsure2000/pneumonia-xray-cnn/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohithsure2000/pneumonia-xray-cnn/blob/main/notebooks/run_in_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](requirements.txt)

A comparison of five CNN architectures -- a from-scratch basic CNN, a
regularized "improved" CNN, and three ImageNet transfer-learning backbones
(VGG-16, VGG-19, ResNet-50) -- for classifying pediatric chest X-rays as
`NORMAL` or `PNEUMONIA`. This repository turns the original exploratory
notebook into a modular, tested, and reproducible Python package with a
CLI for training, evaluation, and single-image inference.

## Project origin

This project began as a team assignment for **BIA 678 (Big Data
Technologies)** at Stevens Institute of Technology, built collaboratively
with a team of five as coursework. This repository is my individual
follow-up: I took our team's exploratory Colab notebook and rebuilt it as
a proper package -- modular architecture definitions, a typed config
object, a real test suite, Docker support, and CI -- while fixing a couple
of correctness bugs I found in the original notebook along the way (see
[`docs/results.md`](docs/results.md) for details). The reported results
table in that doc is our team's original result; this codebase is a
distinct, corrected reimplementation of the same experiment, not a replay
of it.

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Results](#results)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting the data](#getting-the-data)
- [Usage](#usage)
- [Testing](#testing)
- [Docker](#docker)
- [Limitations & future work](#limitations--future-work)
- [License](#license)

## Overview

Chest X-ray interpretation for pediatric pneumonia normally requires an
expert radiologist. This project explores how far a CNN trained on a
public, physician-labeled dataset can go toward automating that first
read, and compares a small custom architecture against transfer learning
from three well-known ImageNet backbones.

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
-- 5,863 anterior-posterior X-rays from pediatric patients (ages 1-5) at
Guangzhou Women and Children's Medical Center, each graded by two expert
physicians (with a third expert re-checking the evaluation set), split
into `train`/`val`/`test` folders of `NORMAL`/`PNEUMONIA` images.

**Approach:**
1. Address the training set's class imbalance (1,342 `NORMAL` vs. 3,876
   `PNEUMONIA`) with on-the-fly augmentation: random rotation, zoom, and
   horizontal/vertical flips.
2. Train and evaluate five architectures under a shared pipeline so their
   results are directly comparable.
3. Evaluate every model on a held-out test set with accuracy, precision,
   recall, F1, and a confusion matrix.

## Repository structure

```
pneumonia-xray-cnn/
├── src/pneumonia_cnn/
│   ├── config.py       # TrainingConfig dataclass -- one place for every hyperparameter
│   ├── data.py         # Data generators, augmentation, test-set loading
│   ├── models.py        # All 5 architectures behind a MODEL_REGISTRY
│   ├── train.py         # CLI: train a model end-to-end
│   ├── evaluate.py      # CLI: evaluate a saved model on the test set
│   ├── predict.py       # CLI: run inference on a single image
│   └── utils.py          # Metrics, seeding, plotting helpers
├── tests/                 # pytest unit tests (no dataset/network required)
├── notebooks/run_in_colab.ipynb  # Clone -> install -> train -> visualize, on a free GPU
├── scripts/
│   ├── download_data.sh
│   └── visualize_predictions.py  # Builds the example-predictions grid for the README
├── docs/
│   ├── results.md          # Full results table + methodology notes
│   └── assets/               # Training curves / example predictions (generated)
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt / requirements-dev.txt
└── pyproject.toml
```

## Results

| Model | Test Accuracy |
| --- | --- |
| **Improved CNN** | **91.28%** |
| VGG-19 | 90.95% |
| VGG-16 | 90.62% |
| Basic CNN | 90.13% |
| ResNet-50 | 89.14% |

*Table above: the original team's reported results (see [Project origin](#project-origin)).
This repo's own architecture implementations are verified to match the
report's model-summary tables (see `docs/results.md` for details and for a
fresh reproduction once available).*

See [`docs/results.md`](docs/results.md) for the full breakdown (precision,
recall, F1, confusion matrices) and for the notes on bugs found and fixed
while refactoring the original notebook into this package.

## Example predictions

<!--
Generated by scripts/visualize_predictions.py -- run notebooks/run_in_colab.ipynb
(or the script directly after training a model) and commit the output here.
-->
![Example test-set predictions](docs/assets/example_predictions.png)

![Training curves](docs/assets/training_curves.png)

## Reproduce the results

The fastest way to run this end-to-end, including on a free GPU, is the
included Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohithsure2000/pneumonia-xray-cnn/blob/main/notebooks/run_in_colab.ipynb)

It clones this repo, installs dependencies, downloads the dataset from
Kaggle, trains a model, and generates the plots above. See
[Installation](#installation) below to run the same steps locally instead.

## Prerequisites

- Python 3.10+
- A [Kaggle](https://www.kaggle.com/) account (to download the dataset)
- Optional: a GPU for reasonable training times -- the transfer-learning
  models in particular are slow on CPU

## Installation

```bash
git clone https://github.com/rohithsure2000/pneumonia-xray-cnn.git
cd pneumonia-xray-cnn
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Getting the data

**Option A -- Kaggle API (used by the Colab notebook):**

1. Create a Kaggle API token: Kaggle account settings → "Create New
   Token" → downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json`
   on Windows) and restrict its permissions: `chmod 600 ~/.kaggle/kaggle.json`.
3. Run the helper script:

   ```bash
   ./scripts/download_data.sh data
   ```

**Option B -- download it by hand (simpler for local/CPU training):**

1. Open the [dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   (sign-in required) and click **Download** (~1.2 GB).
2. Extract the zip into your repo's `data/` folder.
3. ⚠️ **Check the folder depth.** The archive contains an extra nested
   `chest_xray` folder, so a manual extract often lands as
   `data/chest_xray/chest_xray/train` instead of `data/chest_xray/train`.
   If so, move everything out of the inner `chest_xray` folder up one
   level (or pass `--data-dir data/chest_xray/chest_xray` to the CLI
   scripts instead). Either way, you want `NORMAL/` and `PNEUMONIA/`
   folders to end up directly inside `train/`, `val/`, and `test/`.

Both options leave you with:

```
data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

If the folders don't line up, `train.py` will raise a `FileNotFoundError`
telling you exactly which path it expected and, if it can detect the
nested-folder case above, exactly how to fix it.

**A note on speed:** training locally without a GPU works but is slow,
especially for the VGG/ResNet transfer-learning models. If your machine
doesn't have one, either use the Colab notebook (free GPU) or start with
`--model basic --epochs 3` locally just to confirm everything runs before
committing to a longer run.

## Usage

**Train a model:**

```bash
python -m pneumonia_cnn.train --model improved \
    --data-dir data/chest_xray \
    --output-dir artifacts \
    --epochs 15
```

`--model` accepts `basic`, `improved`, `vgg16`, `vgg19`, or `resnet50`.
Transfer-learning models also accept `--fine-tune` to unfreeze the
backbone instead of training only the classification head. Run
`python -m pneumonia_cnn.train --help` for the full option list.

Training writes a checkpoint, a training-curve plot, a `test_metrics.json`
report, and the final saved model to `<output-dir>/<model>/`.

**Evaluate a saved model:**

```bash
python -m pneumonia_cnn.evaluate \
    --model-path artifacts/improved/model.keras \
    --data-dir data/chest_xray
```

**Run inference on a single image:**

```bash
python -m pneumonia_cnn.predict \
    --model-path artifacts/improved/model.keras \
    --image path/to/xray.jpeg
```

**Generate the example-predictions grid (used in the README above):**

```bash
python scripts/visualize_predictions.py \
    --model-path artifacts/improved/model.keras \
    --data-dir data/chest_xray \
    --output docs/assets/example_predictions.png
```

## Testing

The test suite covers the model registry, config, and metrics logic and
does **not** require the dataset or a GPU -- it builds every architecture
with `weights=None` to skip the ImageNet download entirely, so it runs
quickly in CI.

```bash
pip install -r requirements-dev.txt
pytest
```

## Docker

```bash
docker build -t pneumonia-xray-cnn .
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/artifacts:/app/artifacts \
    pneumonia-xray-cnn --model basic --epochs 5
```

## Limitations & future work

- Results reflect a single training run per architecture at 10-15 epochs;
  a k-fold cross-validation setup would give more reliable comparisons.
- The transfer-learning models freeze their entire backbone by default
  (`--fine-tune` opts into full fine-tuning) -- a partial, gradual unfreeze
  schedule would likely close some of the gap with the custom CNN.
- No Grad-CAM or other interpretability output yet, which matters for a
  medical-imaging use case where a clinician would want to see *why* a
  model flagged an image.
- No inference API/web demo -- the CLI is script-only for now.

## License

[MIT](LICENSE)

## Author

**Rohith Sure** -- [surerh2000@gmail.com](mailto:surerh2000@gmail.com)
