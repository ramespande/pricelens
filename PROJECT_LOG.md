# Project Log and Execution Plan

Last updated: 2026-08-14

## Project identity

**Multimodal Product Price Prediction** is an independent research/engineering project using the Amazon ML Challenge 2025 data as a real-world benchmark. It is not a competition-solution reproduction and makes no participation or ranking claim.

Research question: *How much incremental predictive value does visual information provide beyond textual product information for product price estimation, and can adaptive multimodal fusion learn when to trust each modality?*

## Research safeguards

- Price supervision uses only the supplied local dataset.
- No external product or pricing data is used.
- Raw dataset, images, embedding caches, model artifacts, figures, and experiment CSVs are ignored by Git.
- The data split groups exact duplicate catalog text and image links to prevent direct cross-partition leakage.
- Test CSV is not used for model fitting or model selection.

## Completed work

| Milestone | Status | Verified outcome |
| --- | --- | --- |
| Data audit and EDA | Complete | 75k train / 75k test; no missing data; direct duplicate checks documented. |
| Text baseline | Complete | Full train split LightGBM structured-text log target: SMAPE 64.277. |
| Vision infrastructure | Complete | Frozen ResNet-18 pilot is batched, resumable, cached, and failure logged. |
| Image-only pilot | Complete | 800/200 matched subset; image log-Ridge SMAPE 70.940. |
| Matched text pilot | Complete | Same subset; text log-LightGBM SMAPE 78.125. |
| Simple late fusion | Complete | Fixed 50/50 late fusion: SMAPE 69.922 on the matched 800/200 pilot. |
| OOF-selected late fusion | Complete | OOF selected 60% text / 40% image; outer SMAPE 70.472. |
| Scale-5,000 matched study | Complete | Predeclared seed 2026; 4,000 training / 1,000 validation products. |
| Semantic text baseline | Complete | Full split MiniLM-L6-v2 log-LightGBM SMAPE 62.013; matched scale SMAPE 66.641. |
| Semantic late fusion (scale 5K) | Complete | Fixed: SMAPE 65.881; OOF-selected: 65.472. MiniLM-L6-v2 + ResNet-18. |
| Concatenation baseline (scale 5K) | Complete | 896-D [text; image] LightGBM: SMAPE 63.477 (best scale-5K result). |
| Adaptive gated fusion (scale 5K) | Complete | OOF-supervised MLP gate: SMAPE 65.480; did not beat simpler fusion. |
| Error analysis | Complete | Segmented fixed semantic-fusion residuals by price, text length, and disagreement; three local figures saved. |

## Current experiment design

- Outer split: deterministic 80/20 split with `seed=42`; exact duplicate text/image connected groups remain together.
- Pilot subset: 800 products from the outer training partition and 200 from the outer validation partition, selected deterministically from distinct available image links.
- Image predictor: frozen ImageNet-1K ResNet-18, 512-dimensional embedding, standardized Ridge regression on `log1p(price)`.
- Text predictor: lightweight structured catalog-content features, LightGBM on `log1p(price)`.
- Late fusion: arithmetic mean of the two independently trained predictions on the original price scale (`0.5 * text + 0.5 * image`). The 0.5 weight is fixed before evaluation; it is not selected on validation.

## Decision record

1. **Use ResNet-18 for the pilot.** Its 44.7 MB weights and 512-D output make a practical CPU test. Larger encoders are deferred until this baseline is understood.
2. **Use cached embeddings.** Raw image inference is not repeated in each pricing experiment.
3. **Use a small pilot first.** 1,000 images validate data paths, caching, and methodology before any larger extraction.
4. **Compare modalities on identical labels.** The matched 800/200 experiment avoids confusing a modality comparison with a changed validation set.
5. **Use fixed-weight late fusion before adaptive fusion.** It supplies a transparent reference point. Learned/gated fusion is deferred until its comparison protocol is designed.
6. **Use all-MiniLM-L6-v2 for the first semantic text baseline.** It has 22M parameters, ~80 MB weights, 384-dimensional output, Apache-2.0 license, and runs on CPU via `sentence-transformers`. [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
7. **Run semantic fusion at scale 5K using the same cached embeddings.** Both the image cache (`resnet18_matched_scale_5000`) and the text cache (`minilm_l6_v2_matched_scale_5000`) are reused without re-extraction. The fusion weight is pre-specified (50/50) or selected from training-only OOF predictions, as before.
8. **Use a single LightGBM for the concatenation baseline.** Horizontally stacking text (384-D) and image (512-D) into a 896-D joint vector and fitting one LightGBM is the simplest early-fusion approach. No weight tuning is needed and it is directly comparable to the late-fusion baselines.
9. **Implement the adaptive gate in pure NumPy (no torch).** The gating MLP is a shallow 2-layer network (d_in → 128 → 1, ReLU + Sigmoid) trained with mini-batch Adam (lr=1e-3, weight_decay=1e-4) and early stopping (patience=20, 15% inner split). It learns from five-fold OOF component predictions, not in-sample predictions. Avoiding a new torch dependency keeps the gate CPU-friendly and within the existing environment.
10. **Error analysis segments the fixed semantic fusion model.** The fixed 50/50 semantic fusion is the most transparent model and serves as the reference for decomposing errors by price range, text length, and modality disagreement. This produces actionable signals for future feature engineering or encoder upgrades.

## Active and next plan

1. ~~Compare concatenation, training-only-selected late fusion, and adaptive/gated fusion fairly, using semantic text embeddings where appropriate.~~ *(Implemented — see scripts below)*
2. ~~Run robustness and error analyses by price range, text length, image availability, and modality disagreement.~~ *(Implemented — `scripts/run_error_analysis.py`)*
3. ~~Run the implemented scripts against the local dataset and record results.~~ *(Complete; results below.)*
4. **Evaluate cross-validation stability** of concatenation fusion using multiple predeclared seeds. *(Active next step.)*
5. **Evaluate a stronger image encoder** (e.g., ResNet-50 or EfficientNet-B0) only if multi-seed results identify image representation as a bottleneck.
6. Compare missing-modality robustness and investigate category-like error segments only when a reliable derivation is defined.

## Interpretation limits

The 800/200 pilot is too small to establish a stable modality ranking. Its structured-text features are non-semantic, and its frozen ResNet-18 embeddings are generic rather than product-specific. Results from this pilot should guide the next experiment design, not serve as final claims.

## Latest result

The first semantic text baseline uses frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings (22M parameters, ~80 MB, Apache-2.0, 384-D) with LightGBM on the deterministic 80/20 duplicate-grouped split (`seed=42`). Log-target semantic text reached SMAPE **62.013** (MAE 14.282; RMSE 37.781), improving on the structured-text log baseline (SMAPE **64.277**) on the same full split.

On the predeclared 4,000/1,000 matched scale study (`seed=2026`), semantic log-LightGBM SMAPE was **66.641**, essentially matching structured text (SMAPE **66.665**) and still beating image log-Ridge (SMAPE **69.881**). Fixed 50/50 late fusion remains SMAPE **65.661** and OOF-selected fusion SMAPE **65.358** on that matched sample using structured text features.

Semantic embeddings strengthen the full-data text reference, but the matched multimodal study still used structured text. The next fair comparison should rerun fusion with cached semantic text on the same 5,000 products.

### Scale-5,000 semantic fusion results

All models below use the same predeclared `seed=2026` sample: 4,000 training and 1,000 outer-validation products, cached 384-D MiniLM text embeddings, and cached 512-D ResNet-18 image embeddings.

| Model | Validation SMAPE | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Semantic text LightGBM | 66.641 | 14.735 | 30.231 |
| Image Ridge | 69.881 | 16.009 | 32.290 |
| Fixed 50/50 late fusion | 65.881 | 14.763 | 30.591 |
| OOF-selected late fusion (70% text) | 65.472 | — | — |
| Adaptive OOF-supervised gate | 65.480 | 14.586 | 30.333 |
| Concatenation LightGBM, direct price | 72.278 | 16.192 | 27.845 |
| **Concatenation LightGBM, log target** | **63.477** | **14.095** | **29.316** |

Concatenation is the strongest model on this one split, improving SMAPE by 3.164 points over semantic text alone and 2.404 points over fixed late fusion. The gate has meaningful variation on validation (mean text weight 0.636, standard deviation 0.278) but does not beat OOF late fusion, so this experiment does not support its additional complexity yet.

### Scale-5,000 error analysis

For fixed semantic late fusion, error is highest at the price extremes: SMAPE is 127.120 in the cheapest decile (0.4-3.5) and 109.080 in the most expensive decile (50.0-333.3), versus 27.671-52.068 across the middle deciles. Longer catalog content is associated with lower SMAPE (74.478 for the shortest quartile versus 57.402 for the longest). Disagreement is not monotonic: lowest-disagreement SMAPE is 65.129, while highest-disagreement SMAPE is 68.313. This supports further price-aware error work, but not a simple rule that high modality disagreement implies failure.

### Prior 800/200 pilot (reference)

On the matched 800/200 pilot, the pre-specified fixed 50/50 blend achieved SMAPE **69.922**, MAE **16.179**, and RMSE **27.395**. It improves on the image component (SMAPE 70.940) and text component (SMAPE 78.125). The training-only OOF protocol selected 60% text / 40% image (inner OOF SMAPE 67.738) with outer-holdout SMAPE **70.472** — worse than the fixed blend at that scale.

## Predeclared scale study

The next confirmation study uses the new `configs/image_baseline_scale_5000.json` configuration: seed 2026, 4,000 training products, 1,000 validation products, and the same frozen ResNet-18/structured-text model families. Results will be written to separate ignored CSV files. The sample size and seed are set before executing this study; no test data is involved.

### Execution incident

During the first scale-5,000 extraction, a time-limited process was interrupted while rewriting `embeddings.npy`, leaving a generated cache with a mismatched header and payload. No original data or prior cache was changed. The cache writer now saves to a temporary file and atomically replaces the completed cache. The scale-specific cache was subsequently regenerated to 5,000 finite embeddings (4,000 train / 1,000 validation) and used for the completed study. Local results are in `experiments/results/*_scale_5000.csv`.
