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

## Limitations

Structured text features are not semantic text understanding. Image files are inspected only as metadata in Milestone 1; their pixels are not processed. The validation design blocks exact duplicate leakage but not all related-product leakage.
