# Results & Methodology

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(Kermany et al., via Paul Mooney on Kaggle): 5,863 anterior-posterior chest
X-rays from pediatric patients aged 1-5 at Guangzhou Women and Children's
Medical Center, labeled `NORMAL` or `PNEUMONIA` and pre-split into
`train`/`val`/`test`. Images were screened for quality and graded by two
expert physicians, with a third expert checking the evaluation set.

The training split is imbalanced: 1,342 `NORMAL` vs. 3,876 `PNEUMONIA`
images. Training-time augmentation (30° rotation, 20% zoom, horizontal and
vertical flips) is used to reduce overfitting to the majority class.

## Architectures compared

| Model | Description |
| --- | --- |
| Basic CNN | 5-block Conv2D + MaxPool stack, no regularization |
| Improved CNN | Deeper separable-convolution network with batch norm and dropout |
| VGG-16 | ImageNet-pretrained backbone, frozen, custom classification head |
| VGG-19 | ImageNet-pretrained backbone, frozen, custom classification head |
| ResNet-50 | ImageNet-pretrained backbone, frozen, custom classification head |

## Reported results

This project began as a team assignment for **BIA 678 (Big Data
Technologies)** at Stevens Institute of Technology. The table below is the
result set from the team's original project report, based on a single
10-epoch training run per model:

| Model | Test Accuracy |
| --- | --- |
| **Improved CNN** | **91.28%** |
| VGG-19 | 90.95% |
| VGG-16 | 90.62% |
| Basic CNN | 90.13% |
| ResNet-50 | 89.14% |

Precision, recall, and F1 were also tracked per-model via a confusion
matrix on the held-out test set (see `test_metrics.json` produced by
`train.py` for a reproducible, per-run version of this report).

## Notes from productionizing the original notebook

Turning the original exploratory Colab notebook into this package surfaced
a couple of correctness issues worth documenting rather than quietly
papering over:

- **VGG-19 was accidentally built on the VGG-16 backbone.** The notebook
  instantiated a `VGG19` base model but then built the classification head
  on top of the previously-defined `VGG16` variable by mistake, so the two
  "different" transfer-learning models in that run were actually the same
  backbone with independently-initialized heads. `models.py` builds each
  backbone from its own base model.
- **A few evaluation cells reused a stale `acc` variable** when printing
  results for later models in the notebook, so some of the printed
  "Accuracy" lines didn't match the confusion matrix directly above them
  (precision/recall, which were recomputed fresh each cell, were correct).
  `utils.compute_metrics` derives every metric from the same confusion
  matrix in one place to avoid that class of bug.
- **Partial layer-freezing didn't scale across backbones.** Freezing "the
  first 20 layers" happened to freeze all of VGG-16/19 (~19 layers) but
  left most of ResNet-50 (~175 layers) trainable. This repo freezes each
  backbone consistently via `base_model.trainable = False`, with a
  `--fine-tune` flag to opt into full fine-tuning instead.
- **No random seed was fixed**, so re-running the original notebook
  produced slightly different numbers each time. `utils.set_seed` is
  called at the start of `train.py` for reproducibility.

Because of these fixes, a fresh run of `train.py` in this repository is
not expected to reproduce the exact percentages above -- it's a distinct,
corrected implementation of the same experiment. The table is kept here as
the historical result the team reported.
