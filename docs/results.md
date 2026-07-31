# Results & Methodology

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(Kermany et al., via Paul Mooney on Kaggle): 5,863 anterior-posterior
chest X-rays from pediatric patients aged 1-5 at Guangzhou Women and
Children's Medical Center, labeled `NORMAL` or `PNEUMONIA`, split into
`train`/`val`/`test`. Graded by two expert physicians, with a third
checking the evaluation set.

Training split is imbalanced: 1,342 `NORMAL` vs. 3,876 `PNEUMONIA`.
Countered with on-the-fly augmentation (30° rotation, 20% zoom,
horizontal/vertical flip).

## Architectures compared

| Model | Description |
| --- | --- |
| Basic CNN | 5-block Conv2D + MaxPool stack, no regularization |
| Improved CNN | Same 5-block stack + BatchNorm after every block, Dropout after 3 of them |
| VGG-16 | ImageNet-pretrained backbone, frozen, custom classification head |
| VGG-19 | ImageNet-pretrained backbone, frozen, custom classification head |
| ResNet-50 | ImageNet-pretrained backbone, frozen, custom classification head |

## Reported results

Team assignment for BIA 678 (Big Data Technologies) at Stevens Institute
of Technology. Table below is the team's original project report, a
single 10-epoch run per model:

| Model | Test Accuracy |
| --- | --- |
| **Improved CNN** | **91.28%** |
| VGG-19 | 90.95% |
| VGG-16 | 90.62% |
| Basic CNN | 90.13% |
| ResNet-50 | 89.14% |

Precision, recall, and F1 were tracked per model too (see
`test_metrics.json` from `train.py` for a reproducible per-run version).

## Notes from productionizing

A few correctness issues came up while rebuilding the original notebook
into this package:

- **Improved CNN was rebuilt to match the report.** It initially used a
  different, deeper SeparableConv2D network that exists in the notebook
  as an unlabeled second experiment (`model2`), never one of the report's
  five architectures. `build_improved_cnn` now matches the report's
  model-summary table param-for-param (1,246,977 total), confirmed again
  by the training run below.
- **VGG-19 was accidentally built on the VGG-16 backbone.** The notebook
  instantiated a `VGG19` base model but built the classification head on
  the previous `VGG16` variable by mistake, so the two "different"
  transfer-learning models were actually the same backbone with separate
  heads. Fixed by building each backbone independently.
- **A few evaluation cells reused a stale `acc` variable**, so some
  printed accuracy lines didn't match the confusion matrix above them
  (precision/recall were recomputed fresh each cell and were correct).
  `compute_metrics` now derives every metric from one confusion matrix.
- **Partial layer-freezing didn't scale across backbones.** Freezing
  "the first 20 layers" froze all of VGG-16/19 (~19 layers) but left most
  of ResNet-50 (~175 layers) trainable. Backbones now freeze consistently
  via `base_model.trainable = False`, with `--fine-tune` to opt into
  full fine-tuning.
- **No random seed was fixed**, so re-runs produced different numbers
  each time. `set_seed()` runs at the start of `train.py`.
- **The official `val/` folder has only 16 images** -- too small for a
  stable EarlyStopping/ReduceLROnPlateau signal. It once caused a
  checkpoint that collapsed into predicting `NORMAL` for every image
  (37.5% accuracy, 0% precision/recall on `PNEUMONIA`). `build_generators`
  now carves validation out of the training folder instead
  (`TrainingConfig.validation_split`, default 15%), ~50x more data.
  `test/` is untouched and still the only thing the reported metrics use.

A fresh run of `train.py` won't reproduce the exact percentages above --
this is a corrected reimplementation, not a replay of the original.

## Reproduced results (this repo)

`train.py --model improved --epochs 15`, corrected architecture (see
`test_metrics.json`). Model summary confirmed 1,246,977 params, matching
the report exactly:

| Metric | Value |
| --- | --- |
| Accuracy | 88.30% |
| Precision | 90.96% |
| Recall | 90.26% |
| F1 | 90.60% |

Confusion matrix: 199 true negatives, 35 false positives, 38 false
negatives, 352 true positives (624 total).

Close to the historical 91.28% -- the remaining gap is plausibly this
being a single run at 12 actual epochs (EarlyStopping triggered) rather
than 15, plus the validation-split and seed changes above. Precision and
recall are well balanced here, unlike the earlier wrong-architecture run.

About half the epochs in this run finished in ~6s instead of ~70s, each
with a "ran out of data" warning from Keras -- a known issue with
`ImageDataGenerator` across multiple `.fit()` epochs on newer TF/Keras.
The result above still holds (EarlyStopping picked its checkpoint off
real validation numbers), but the model saw less real training data than
12 full epochs implies. Switching to
`tf.keras.utils.image_dataset_from_directory` would fix this.
