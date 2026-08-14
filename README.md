# Multimodal Product Price Prediction

An independent research/engineering project studying product price prediction with the Amazon ML Challenge 2025 dataset as a realistic benchmark.

> This is an independent project using the Amazon ML Challenge 2025 dataset as a real-world benchmark. It is not a reproduction of the competition-winning solution and does not claim participation or ranking in the original competition.

## Research question

How much incremental predictive value does visual information provide beyond textual product information for product price estimation, and can adaptive multimodal fusion learn when to trust each modality?

Milestone 1 establishes a reproducible, CPU-friendly foundation: schema validation, leakage-aware validation splitting, EDA, a median benchmark, and structured-text regression. It does **not** include pretrained encoders, image embedding extraction, multimodal fusion, or gated fusion.

## Dataset and safety

The dataset is local-only and must be supplied through `DATA_ROOT`; it is never copied, modified, uploaded, or committed. Only the supplied dataset is used for price supervision. No external pricing data or product scraping is used.

Set the root to the directory containing `train.csv` and `test.csv`:

```powershell
$env:DATA_ROOT = 'C:\path\to\dataset'
```

## Evaluation

The primary metric is SMAPE:

`mean(|prediction - actual| / ((|actual| + |prediction|) / 2)) * 100`

Zero denominators are assigned a zero contribution. MAE and RMSE are also reported on the original price scale.

## Validation methodology

The split is deterministic from `configs/baseline.json`. Connected groups that share an exact `catalog_content` or exact `image_link` are kept together, preventing direct duplicate leakage across train and validation. This is deliberately conservative but does not claim to prevent semantic near-duplicate leakage.

## Structured features

`src/features/text.py` extracts counts for characters, words, digits, uppercase letters, numeric tokens, punctuation, whitespace, average word length, and line-level `Item Name`/`Product Description` lengths. Item pack quantity is only extracted for explicit `Pack of N` or `N per case` patterns; all other cases are left as zero rather than inferred.

## Installation and usage

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python scripts/run_baselines.py --data-root $env:DATA_ROOT
```

Open `notebooks/01_eda.ipynb` for exploratory figures and `notebooks/02_baseline.ipynb` to inspect recorded results. Figures and experiment logs are intentionally gitignored because they are generated only from actual local runs.

## Repository layout

```
configs/                 reproducible parameters
data/                    empty, git-kept output roots
notebooks/               thin EDA and baseline notebooks
src/data/                loading, validation, splitting, image-job interface
src/features/            structured text features
src/training/            CPU baseline runner
src/evaluation/          SMAPE and regression metrics
tests/                   unit tests
experiments/results/     local experiment CSVs (ignored)
reports/figures/         local EDA figures (ignored)
```

## Current experiments

The following results were produced locally on the deterministic 80/20 duplicate-grouped validation split (`seed=42`) and are logged in the local, gitignored `experiments/results/baselines.csv`.

| Experiment | SMAPE | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Training median | 72.463 | 16.626 | 40.745 |
| LightGBM structured features, direct price | 70.113 | 16.068 | 37.187 |
| LightGBM structured features, log1p(price) | **64.277** | **14.593** | 38.321 |

The log-target model is the strongest Milestone 1 baseline by SMAPE; no multimodal improvement is claimed.

## Verified dataset snapshot

The configured local dataset contains 75,000 train rows and 75,000 test rows. Train columns are `sample_id` (int64), `catalog_content` (object), `image_link` (object), and `price` (float64); test omits `price`. Neither split has missing values, duplicate sample IDs, or duplicate full rows. Train has 100 duplicate catalog-content rows and 2,712 duplicate image-link rows; test has 119 and 2,778 respectively. Train price has mean 23.648, median 14.000, standard deviation 33.377, range 0.13–2796.00, and 1%/99% quantiles 1.32/145.25.

The local train image directory contains 72,287 files (16.42 GiB) and the test directory 72,552 files (16.60 GiB). URL-basename matching finds a local image for 74,999 of 75,000 rows in each split; image pixels were not opened or processed.

## Hardware and future work

Development is designed for CPU-only use. Future work will first cache batched, resumable embeddings from small image samples, then compare text-only, image-only, concatenation, late fusion, and adaptive gated fusion. Those methods will be treated as hypotheses and evaluated against strong baselines, including missing-modality and error analyses.

## Milestone 2: vision-embedding pilot

The first vision experiment is deliberately bounded to 500 unique locally available training images. It uses torchvision ResNet-18 ImageNet-1K V1 as a frozen feature extractor, removes its classification head, and caches 512-dimensional `float32` embeddings. The model has 11.7M parameters, needs 1.81 GFLOPs per image, and downloads 44.7 MB of weights; torchvision supplies the official resize/crop/normalization transform. The torchvision repository is BSD-3-Clause licensed. The PyTorch CPU runtime is materially larger than the weights, so it is installed only when this pilot is desired. [Model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE)

No raw image or embedding is tracked by Git. Validate the deterministic selection without loading a model:

```powershell
python scripts/extract_image_embeddings.py --data-root $env:DATA_ROOT --dry-run
```

After installing the updated requirements, run the 500-image CPU pilot:

```powershell
pip install -r requirements.txt
python scripts/extract_image_embeddings.py --data-root $env:DATA_ROOT
```

The command is resumable: it records `manifest.csv`, writes failed image reads to `failed_images.csv`, avoids duplicate image links during selection, and skips completed sample IDs on rerun. The cache lives below `data/processed/image_embeddings/` and remains ignored.

### Verified pilot run

On the local CPU environment, the deterministic `seed=42` pilot completed 500/500 selected unique train images with zero failed reads. It produced a finite `(500, 512)` `float32` embedding matrix (1,024,128 bytes on disk). A second run skipped all 500 completed sample IDs, confirming resumability. These embeddings are infrastructure validation only; no price model has been trained on them yet.

### Next: leakage-aware image-only baseline

`scripts/run_image_baseline_pilot.py` creates the same duplicate-grouped 80/20 split before sampling 800 training and 200 validation images. It then uses frozen 512-dimensional ResNet-18 embeddings with regularized Ridge regression, evaluating direct-price and `log1p(price)` targets only on the 200-image validation sample. This is a small-sample feasibility experiment, not a comparison to the full text baseline.

```powershell
python scripts/run_image_baseline_pilot.py --data-root $env:DATA_ROOT
```

The matched-sample median control is recorded with the image-only models, so the pilot measures whether frozen visual features add value over a trivial baseline on exactly the same 800/200 labels.

| Image-pilot experiment (800 train / 200 validation) | SMAPE | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Matched training median | 71.936 | 16.685 | 29.773 |
| ResNet-18 + Ridge, direct price | 87.040 | 19.602 | **27.519** |
| ResNet-18 + Ridge, `log1p(price)` | **70.940** | **16.619** | 28.309 |

On this small, leakage-aware image-only pilot, log-target frozen visual features improve SMAPE by 0.996 points over the matched median. This is evidence of a weak visual signal, not evidence that vision outperforms text: the pilot uses a much smaller validation sample than the full text baseline, and no cross-modal comparison has been performed yet.

### Matched text comparison

The next experiment uses the **exact same** 800 training and 200 validation sample IDs from the cached image pilot, with the same duplicate-grouped split. It therefore supports a direct text-versus-image comparison without changing the labelled products:

```powershell
python scripts/run_matched_text_baselines.py --data-root $env:DATA_ROOT
```

| Matched 800/200 experiment | SMAPE | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Median control | 71.936 | 16.685 | 29.773 |
| ResNet-18 image embeddings + log-Ridge | **70.940** | **16.619** | 28.309 |
| Structured text LightGBM, direct price | 86.482 | 21.041 | 30.384 |
| Structured text LightGBM, `log1p(price)` | 78.125 | 18.048 | 29.675 |

On this small matched pilot, frozen image embeddings are stronger than the deliberately lightweight structured-text features. That should not be overinterpreted as a general modality ranking: 800 training products is too few for a stable conclusion, these text features are non-semantic, and the full-data text baseline remains substantially stronger than either tiny-pilot result. The appropriate next comparison is a simple late-fusion model on this exact same sample.

### Fixed-weight late fusion

The first fusion experiment is deliberately transparent: independently train the log-target image Ridge and structured-text LightGBM models on the same 800 products, then average their original-price predictions with a fixed 50/50 weight. The validation set is not used to pick the fusion weight.

```powershell
python scripts/run_late_fusion_pilot.py --data-root $env:DATA_ROOT
```

| Matched 800/200 log-target model | SMAPE | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Image-only Ridge | 70.940 | 16.619 | 28.309 |
| Text-only LightGBM | 78.125 | 18.048 | 29.675 |
| Fixed 50/50 late fusion | **69.922** | **16.179** | **27.395** |

The fixed blend improves this pilot’s SMAPE by 1.018 points relative to its image component. This is a small, single-split result; it is evidence to test the fusion hypothesis at a larger predeclared scale, not confirmation that fixed fusion generalizes.

### Training-only selected late fusion

To test learned fusion without choosing a weight on the outer holdout, this experiment makes five GroupKFold splits of the 800 training products (grouped by exact catalog text), generates out-of-fold predictions from both component models, and selects a convex blend weight from 0.00 to 1.00 in 0.05 increments by OOF SMAPE. It refits the components on all 800 training products before evaluating the 200-product outer holdout.

```powershell
python scripts/run_oof_late_fusion_pilot.py --data-root $env:DATA_ROOT
```

The five-fold training-only OOF procedure selected a 60% text / 40% image blend (inner OOF SMAPE 67.738). Its outer-holdout score was SMAPE 70.472, MAE 16.324, and RMSE 27.605. It improves on image-only but does not beat the fixed 50/50 reference at this scale, so the project retains equal-weight fusion as the clearer pilot baseline.

### Predeclared scale-5,000 confirmation study

The next matched experiment is locked to `seed=2026`, 4,000 training products, and 1,000 validation products. It uses a separate cache and separate ignored result logs, avoiding any mix with the 800/200 pilot. It still uses no test data.

The complete research decisions, active experiment design, execution plan, and interpretation limits are maintained in [PROJECT_LOG.md](PROJECT_LOG.md).

## Limitations

Structured text features are not semantic text understanding. Image files are inspected only as metadata in Milestone 1; their pixels are not processed. The validation design blocks exact duplicate leakage but not all related-product leakage.

## Semantic fusion at scale 5,000

The next fair multimodal comparison reuses the cached 4,000/1,000 matched sample (seed=2026) but replaces the non-semantic structured text features with the already-cached MiniLM-L6-v2 semantic embeddings. Both fixed 50/50 and OOF-selected weights are evaluated on the same outer holdout under the same experimental protocol as the prior late-fusion study.

```powershell
python scripts/run_semantic_late_fusion_scale.py --data-root $env:DATA_ROOT
```

Results are written to `experiments/results/semantic_fusion_scale_5000.csv`. The script does not re-extract any embeddings; it requires that both the image cache (`data/processed/image_embeddings/resnet18_matched_scale_5000`) and the text cache (`data/processed/text_embeddings/minilm_l6_v2_matched_scale_5000`) already exist.

## Concatenation baseline at scale 5,000

Early/concatenation fusion: the 384-D semantic text embedding and the 512-D image embedding are horizontally stacked into a single 896-D joint feature vector, which is passed to a single LightGBM regressor. This tests whether a unified model extracting cross-modal interactions outperforms independent component models combined in a late-fusion step.

```powershell
python scripts/run_concatenation_baseline_scale.py --data-root $env:DATA_ROOT
```

Results are written to `experiments/results/concatenation_baseline_scale_5000.csv`.

## Adaptive (gated) fusion at scale 5,000

A shallow MLP gating network learns per-sample modality weights from the concatenated [text; image] embedding. For each sample it produces a scalar gate `g ∈ (0, 1)` such that the fused log-price prediction is `g * log_text_pred + (1 - g) * log_image_pred`. The gate is a 2-layer network (896 → 128 ReLU → 1 Sigmoid) trained with Adam and early stopping on a 15% inner training hold-out. It is implemented in pure NumPy so no additional torch dependency is introduced. The outer validation holdout plays no role in gate training.

```powershell
python scripts/run_adaptive_fusion_scale.py --data-root $env:DATA_ROOT
```

Results are written to `experiments/results/adaptive_fusion_scale_5000.csv`. Optional arguments `--epochs`, `--hidden`, and `--batch-size` allow lightweight ablation.

## Robustness and error analysis

Once any fusion experiment has been run, the error analysis script segments validation residuals by three factors:

- **Price decile** — reveals whether the model is systematically worse for cheap or expensive products.
- **Catalog content text length quartile** — identifies whether thin or verbose product descriptions affect accuracy.
- **Modality disagreement quartile** — measures whether large text/image prediction gaps correlate with higher error, which would motivate adaptive fusion.

```powershell
python scripts/run_error_analysis.py --data-root $env:DATA_ROOT [--figures]
```

Results are written to `experiments/results/error_analysis_scale_5000.csv`. Pass `--figures` to also save matplotlib bar charts under `reports/figures/`.

### Verified scale-5,000 comparison

All models below use the same predeclared 4,000/1,000 matched sample (`seed=2026`) and cached MiniLM-L6-v2 text plus ResNet-18 image embeddings where applicable.

| Model | SMAPE | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Semantic text LightGBM | 66.641 | 14.735 | 30.231 |
| Image Ridge | 69.881 | 16.009 | 32.290 |
| Fixed 50/50 late fusion | 65.881 | 14.763 | 30.591 |
| OOF-selected late fusion | 65.472 | — | — |
| Adaptive OOF-supervised MLP gate | 65.480 | 14.586 | 30.333 |
| Concatenation LightGBM, direct price | 72.278 | 16.192 | 27.845 |
| **Concatenation LightGBM, log target** | **63.477** | **14.095** | **29.316** |

Concatenation is the strongest result on this single held-out study. The adaptive gate varied its text weight across validation examples, but did not outperform the simpler methods; the project therefore does not claim that adaptive fusion is beneficial yet.
