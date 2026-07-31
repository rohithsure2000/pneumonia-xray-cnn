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
| Improved CNN | Same 5-block Conv2D stack + BatchNorm after every block and Dropout after 3 of them |
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

- **"Improved CNN" initially implemented the wrong architecture.** The
  project's PDF report (with full model-summary tables) wasn't available
  when this repo's `improved_cnn` was first written, so it was built to
  match a different, deeper SeparableConv2D network that also appears in
  the exploratory notebook (as an unlabeled second experiment, `model2`)
  but was never one of the report's five official architectures. Once the
  actual report was available, `build_improved_cnn` was rewritten to match
  it exactly -- verified param-for-param (1,246,977 total parameters)
  against the report's own model-summary table, and confirmed again by the
  actual Colab training run in the "Reproduced results" section below.
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
- **The dataset's official `val/` folder contains only 16 images.** This
  is a well-documented quirk of this specific Kaggle dataset, and it bit
  a real training run of this repo's `improved` model: with such a tiny,
  noisy validation signal, `EarlyStopping`/`ReduceLROnPlateau` restored a
  checkpoint that had collapsed into predicting `NORMAL` for every image
  (37.5% test accuracy, 0% precision/recall on the `PNEUMONIA` class --
  which is exactly the fraction of `NORMAL` images in the test set).
  `build_generators` now carves a validation split out of the *training*
  folder instead (`TrainingConfig.validation_split`, default 15%), giving
  a validation set roughly 50x larger. The real `test/` folder is
  untouched by this change and remains the only thing used for the
  reported metrics above.

Because of these fixes, a fresh run of `train.py` in this repository is
not expected to reproduce the exact percentages above -- it's a distinct,
corrected implementation of the same experiment. The table is kept here as
the historical result the team reported.

## Reproduced results (this repo)

Actual output of `train.py --model improved --epochs 15` in this repo, run
against the corrected architecture above (see `test_metrics.json`) -- the
model summary for this run confirmed **1,246,977 total parameters**,
matching the report's table exactly:

| Metric | Value |
| --- | --- |
| Accuracy | 88.30% |
| Precision | 90.96% |
| Recall | 90.26% |
| F1 | 90.60% |

Confusion matrix: 199 true negatives, 35 false positives, 38 false
negatives, 352 true positives (624 test images total).

This lines up well with the historical 91.28% figure above -- close enough
to be the same experiment, with the remaining gap plausibly explained by
this being a single run at 12 actual epochs (EarlyStopping triggered)
rather than a full 15, plus the validation-split and random-seed changes
described above. Unlike the earlier (wrong-architecture) run, precision
and recall are now well balanced rather than one dominating the other,
which is a healthier sign for a model meant to be used as a screening aid
rather than tuned to just maximize one metric.

One implementation quirk observed during this run, worth noting rather
than hiding: roughly half the epochs finished in ~6 seconds instead of the
expected ~70, each accompanied by a `"Your input ran out of data;
interrupting training"` warning from Keras. This is a known limitation of
using the legacy `ImageDataGenerator` API (`data.py`) across multiple
`.fit()` epochs on newer TensorFlow/Keras versions -- it doesn't invalidate
the result above (EarlyStopping still selected its checkpoint off real
validation numbers), but it means the model saw somewhat less real
training data than 12 full epochs would suggest, and the training curve
looks choppier as a result. Switching `data.py` to
`tf.keras.utils.image_dataset_from_directory` would resolve this and is a
reasonable next improvement.
