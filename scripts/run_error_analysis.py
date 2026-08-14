"""Robustness and error analysis across experiment segments.

Loads the best available experiment results and segments validation residuals
by four factors:
  1. Price range (deciles of true validation price)
  2. Catalog content text length (quartiles of character count)
  3. Modality disagreement (quartiles of |text_pred - image_pred| / mean_pred)

For each segmentation variable the script reports per-segment:
  - N (sample count)
  - SMAPE
  - MAE
  - median absolute error

An optional matplotlib figure is produced for each factor if --figures is set.

Usage:
    python scripts/run_error_analysis.py --data-root <DATA_ROOT> [--figures]

Options:
    --data-root    Path to the directory containing train.csv and test.csv.
    --config       Config JSON path (default: configs/semantic_fusion_scale_5000.json).
    --results      Path to write the segmented error CSV
                   (default: experiments/results/error_analysis_scale_5000.csv).
    --figures      If set, also save matplotlib bar charts to reports/figures/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import regression_metrics, smape
from src.training.matched_text_baselines import _matched_rows
from src.training.semantic_late_fusion import _load_embeddings, _fit_image_model, _fit_text_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def _price_decile_label(prices: np.ndarray) -> np.ndarray:
    """Return 1-indexed decile labels (1=cheapest 10 %, 10=most expensive 10 %)."""
    quantiles = np.nanpercentile(prices, np.arange(0, 101, 10))
    labels = np.searchsorted(quantiles[1:-1], prices, side="right") + 1
    return np.clip(labels, 1, 10)


def _quartile_label(values: np.ndarray) -> np.ndarray:
    """Return 1-indexed quartile labels."""
    q = np.nanpercentile(values, [25, 50, 75])
    labels = np.searchsorted(q, values, side="right") + 1
    return np.clip(labels, 1, 4)


def _segment_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Compute per-segment SMAPE, MAE, and median-AE."""
    rows = []
    for lbl in sorted(np.unique(labels)):
        mask = labels == lbl
        y_seg, p_seg = y_true[mask], pred[mask]
        rows.append({
            "segment": label_names[lbl] if label_names else str(lbl),
            "n": int(mask.sum()),
            "smape": smape(y_seg, p_seg),
            "mae": float(np.mean(np.abs(p_seg - y_seg))),
            "median_ae": float(np.median(np.abs(p_seg - y_seg))),
        })
    return pd.DataFrame(rows)


def _record(path: Path, rows: pd.DataFrame, segment_name: str) -> None:
    out = rows.copy()
    out.insert(0, "segmentation", segment_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    out.to_csv(path, mode="a", index=False, header=header)


def _save_figure(df: pd.DataFrame, title: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available; skipping figure %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["segment"].astype(str), df["smape"], color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Segment")
    ax.set_ylabel("SMAPE (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Figure saved: %s", out_path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_error_analysis(
    data_root: str | Path,
    config_path: str | Path = "configs/semantic_fusion_scale_5000.json",
    results_path: str | Path = "experiments/results/error_analysis_scale_5000.csv",
    save_figures: bool = False,
) -> pd.DataFrame:
    """Segment validation residuals for the best fusion model (fixed 50/50 semantic).

    Returns a DataFrame with all segmented error rows.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    image_dir = Path(config["image_output_dir"])
    text_dir = Path(config["text_output_dir"])
    seed = int(config["random_seed"])

    image_config = dict(config)
    image_config["output_dir"] = str(image_dir)
    train, validation = _matched_rows(Path(data_root), image_config)

    x_image_train, y_train = _load_embeddings(image_dir, train)
    x_image_val, y_val = _load_embeddings(image_dir, validation)
    x_text_train, _ = _load_embeddings(text_dir, train)
    x_text_val, _ = _load_embeddings(text_dir, validation)

    image_model = _fit_image_model(x_image_train, y_train)
    text_model = _fit_text_model(x_text_train, y_train, seed)

    text_pred = np.expm1(text_model.predict(x_text_val))
    image_pred = np.expm1(image_model.predict(x_image_val))
    fusion_pred = np.clip((np.clip(text_pred, 0, None) + np.clip(image_pred, 0, None)) / 2, 0, None)

    all_frames: list[pd.DataFrame] = []
    results_path = Path(results_path)

    # --- Segmentation 1: Price decile ---
    LOGGER.info("Segmenting by price decile …")
    decile_labels = _price_decile_label(y_val)
    q_vals = np.nanpercentile(y_val, np.arange(0, 101, 10))
    decile_names = {
        i + 1: f"D{i + 1} [{q_vals[i]:.1f}-{q_vals[i + 1]:.1f}]"
        for i in range(10)
    }
    seg_price = _segment_metrics(y_val, fusion_pred, decile_labels, decile_names)
    _record(results_path, seg_price, "price_decile")
    all_frames.append(seg_price.assign(segmentation="price_decile"))
    if save_figures:
        _save_figure(seg_price, "SMAPE by price decile (semantic fixed fusion)", Path("reports/figures/error_price_decile.png"))

    # --- Segmentation 2: Text length quartile ---
    LOGGER.info("Segmenting by catalog content length …")
    text_lengths = validation.catalog_content.astype(str).str.len().to_numpy()
    q_text = np.nanpercentile(text_lengths, [25, 50, 75])
    length_labels = _quartile_label(text_lengths)
    length_names = {
        1: f"Q1 [0-{q_text[0]:.0f} chars]",
        2: f"Q2 [{q_text[0]:.0f}-{q_text[1]:.0f} chars]",
        3: f"Q3 [{q_text[1]:.0f}-{q_text[2]:.0f} chars]",
        4: f"Q4 [{q_text[2]:.0f}+ chars]",
    }
    seg_text = _segment_metrics(y_val, fusion_pred, length_labels, length_names)
    _record(results_path, seg_text, "text_length_quartile")
    all_frames.append(seg_text.assign(segmentation="text_length_quartile"))
    if save_figures:
        _save_figure(seg_text, "SMAPE by catalog text length (semantic fixed fusion)", Path("reports/figures/error_text_length.png"))

    # --- Segmentation 3: Modality disagreement quartile ---
    LOGGER.info("Segmenting by modality disagreement …")
    mean_pred = (np.clip(text_pred, 1e-3, None) + np.clip(image_pred, 1e-3, None)) / 2
    disagreement = np.abs(text_pred - image_pred) / mean_pred
    disagree_labels = _quartile_label(disagreement)
    q_dis = np.nanpercentile(disagreement, [25, 50, 75])
    disagree_names = {
        1: f"Q1 [low <= {q_dis[0]:.2f}]",
        2: f"Q2 [{q_dis[0]:.2f}-{q_dis[1]:.2f}]",
        3: f"Q3 [{q_dis[1]:.2f}-{q_dis[2]:.2f}]",
        4: f"Q4 [high >{q_dis[2]:.2f}]",
    }
    seg_disagree = _segment_metrics(y_val, fusion_pred, disagree_labels, disagree_names)
    _record(results_path, seg_disagree, "modality_disagreement_quartile")
    all_frames.append(seg_disagree.assign(segmentation="modality_disagreement_quartile"))
    if save_figures:
        _save_figure(seg_disagree, "SMAPE by modality disagreement (semantic fixed fusion)", Path("reports/figures/error_disagreement.png"))

    combined = pd.concat(all_frames, ignore_index=True)
    LOGGER.info("Error analysis written to %s", results_path)
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/semantic_fusion_scale_5000.json")
    parser.add_argument("--results", default="experiments/results/error_analysis_scale_5000.csv")
    parser.add_argument("--figures", action="store_true", help="Save matplotlib figures to reports/figures/")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    df = run_error_analysis(args.data_root, args.config, args.results, save_figures=args.figures)
    for seg, group in df.groupby("segmentation", sort=False):
        LOGGER.info("\n--- %s ---", seg)
        LOGGER.info(group[["segment", "n", "smape", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
