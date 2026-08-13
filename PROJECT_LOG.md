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
| Scale-5,000 matched study | In progress | Predeclared seed 2026; 4,000 training / 1,000 validation products. |

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

## Active and next plan

1. Complete the predeclared 4,000/1,000 matched scale study. *(Active)*
2. Establish a stronger semantic text-embedding baseline, after documenting model size, license, and CPU/cloud requirements.
3. Compare concatenation, training-only-selected late fusion, and adaptive/gated fusion fairly.
4. Run robustness and error analyses by price range, text length, image availability, and modality disagreement.

## Interpretation limits

The 800/200 pilot is too small to establish a stable modality ranking. Its structured-text features are non-semantic, and its frozen ResNet-18 embeddings are generic rather than product-specific. Results from this pilot should guide the next experiment design, not serve as final claims.

## Latest result

On the matched 800/200 pilot, the pre-specified fixed 50/50 blend achieved SMAPE **69.922**, MAE **16.179**, and RMSE **27.395**. It improves on the image component (SMAPE 70.940) and text component (SMAPE 78.125). Because the blend weight was fixed before this evaluation, the result does not tune a fusion parameter on validation; nevertheless, this is a single small pilot and requires confirmation at a larger, predeclared scale.

The training-only five-fold OOF protocol selected a 60% text / 40% image convex blend, with inner OOF SMAPE 67.738. Its outer-holdout SMAPE was **70.472** (MAE 16.324; RMSE 27.605): better than image-only but worse than the fixed blend. At this small scale, training-only blend selection is noisy, so it does not justify replacing the transparent equal-weight reference.

## Predeclared scale study

The next confirmation study uses the new `configs/image_baseline_scale_5000.json` configuration: seed 2026, 4,000 training products, 1,000 validation products, and the same frozen ResNet-18/structured-text model families. Results will be written to separate ignored CSV files. The sample size and seed are set before executing this study; no test data is involved.

### Execution incident

During the first scale-5,000 extraction, a time-limited process was interrupted while rewriting `embeddings.npy`, leaving a generated cache with a mismatched header and payload. No original data or prior cache was changed. The cache writer now saves to a temporary file and atomically replaces the completed cache; the corrupted scale-specific cache will be regenerated using the unchanged predeclared configuration.
