# df-augment — Decisions & Open Questions

Answers are written here with the date decided. Open items are marked TBD — bring to Jacob.

---

## Decided

| Date | Decision |
|------|----------|
| 2026-10-13 | Repo name: `df-augment`. PyPI name: `df-augment`. Import name: `df_augment`. CLI command: `df-augment`. |
| 2026-10-13 | Licence: TBD — MIT or BSD-3, to be confirmed with Jacob at kickoff. |
| 2026-10-13 | Repo is public from day one (JOSS requires 6+ months of public history). |
| 2026-10-13 | Branch protection: PRs required to merge to `main`, 1 required approval (Jacob reviews). |
| 2026-10-13 | `aug_source_group` feeds `--grouper` in both df-analyze and DeepTune, which passes it to scikit-learn `StratifiedGroupKFold`. All copies of one patient share the same group ID and can never split across train/test. |
| 2026-10-13 | SMOTE rows have two parents — both listed in `aug_parent_ids`, both merged into one `aug_source_group`. |
| 2026-10-13 | MixUp rows have two parents and produce soft labels stored in `aug_label_soft`. |
| 2026-10-13 | SMOTE cannot interpolate categorical columns — constrained SMOTE will raise a clear error on datasets with categoricals until SMOTE-NC arrives in Phase 2. |
| 2026-10-13 | Image format for df-embed: two-column parquet with `image` (PIL-readable bytes) and `label` (int 0..c-1). |
| 2026-10-13 | DeepTune split files: `data_splits_*/train_split.parquet` and `test_split.parquet`. |
| 2026-10-13 | df-analyze output splits: `results/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`. |

---

## Open — Bring to Jacob at Kickoff

| # | Question |
|---|----------|
| 1 | **Licence**: MIT or BSD-3? |
| 2 | **Parquet input for df-analyze**: `--df` help text lists `.xlsx`, `.json`, `.csv`, `.npy` only. Does the current version accept `.parquet`? If not, does the package need a CSV export path, or should we patch df-analyze? |
| 3 | **`--grouper` type constraints**: Does `--grouper` in both tools accept any string column, or are there type/cardinality constraints? |
| 4 | **test/validation pools**: Should they hold original rows only (`aug_is_original == True`)? This needs a change in both tools — decide early. |
| 5 | **Default `--multiplier`**: 2x or 5x? |
| 6 | **`--balance` target ratio**: does it take a target ratio (e.g. 1.0 = fully balanced)? |
| 7 | **SMOTE pairing strategy**: greedy nearest-neighbour (simple, fast) or optimal matching? |
| 8 | **Datasets for leakage demo and benefit study**: which ones? |
| 9 | **3D MRI/CT path**: needed in year one? |
| 10 | **MixUp/CutMix soft labels in DeepTune**: does DeepTune's loss function accept soft labels? (Required for MixUp to work end-to-end.) |
| 11 | **PR reviewers and meeting cadence**: who reviews PRs, and how often do we meet? |
| 12 | **Compute Canada / DRAC access**: needed for later experiments — arrange early. |

---

## Reading Log (Phase 0a Step 1)

| Item | Status | Key notes |
|------|--------|-----------|
| Project plan (df-augment) | Done | Read twice. Open questions logged above. |
| CSCI 555 crash course slides | Done | Covers ML workflow, df-analyze, DeepTune, supervised/unsupervised learning, model validation, KNN, SVM, trees, forests, boosting, stacking. |
| df-analyze README + argument list | Done | `--grouper` -> StratifiedGroupKFold. `--drops` removes columns entirely. Parquet input TBD (Q2 above). |
| DeepTune README | Done | `--grouper` supported. Splits saved as parquet. Supports images (ResNet/DenseNet/Swin/ViT), NLP (BERT/GPT-2), tabular (GANDALF/TabPFN), time series (DeepAR). |
| SMOTE — Chawla et al. 2002 | Done | Interpolates between k nearest neighbours in minority class. Two parents per row. Cannot handle categoricals. |
| MixUp — Zhang et al. 2018 | Done | x̃ = λxᵢ + (1-λ)xⱼ, ỹ = λyᵢ + (1-λ)yⱼ, λ ~ Beta(α,α). Produces soft labels. Two parents per row. |
| Image augmentation survey — Shorten & Khoshgoftaar 2019 | Done | Key categories: geometric (flip, rotate, crop), photometric (brightness, contrast, hue), erase (CutOut), mix (MixUp, CutMix), learned (AutoAugment). |
| scikit-learn GroupKFold / StratifiedGroupKFold | Done | GroupKFold: group separation only. StratifiedGroupKFold: group separation + class balance. df-analyze uses the latter via `--grouper`. |
