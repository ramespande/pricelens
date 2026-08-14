"""Run the concatenation (early) fusion baseline at scale 5000.

Horizontally stacks cached 384-D semantic text embeddings (MiniLM-L6-v2) and
512-D image embeddings (ResNet-18) into a 896-D joint vector and fits a single
LightGBM regressor on the matched 4,000/1,000 sample (seed=2026).

Both embedding caches must already exist; this script will not re-extract them.

Usage:
    python scripts/run_concatenation_baseline_scale.py --data-root <DATA_ROOT>

Options:
    --data-root   Path to the directory containing train.csv and test.csv.
    --config      Config JSON path (default: configs/concatenation_baseline_scale_5000.json).
    --results     Output CSV path (default: experiments/results/concatenation_baseline_scale_5000.csv).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Path to dataset root containing train.csv")
    parser.add_argument("--config", default="configs/concatenation_baseline_scale_5000.json")
    parser.add_argument("--results", default="experiments/results/concatenation_baseline_scale_5000.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    from src.training.concatenation_baseline import run_concatenation_baseline

    logging.info("=== Concatenation Baseline (scale 5000) ===")
    rows = run_concatenation_baseline(args.data_root, args.config, args.results)
    for row in rows:
        if row["model"] == "median":
            logging.info("  %-35s  SMAPE=%.3f", row["experiment_name"], row["validation_smape"])
        else:
            logging.info(
                "  %-35s  SMAPE=%.3f  MAE=%.3f  RMSE=%.3f  (d_in=%d)",
                row["experiment_name"],
                row["validation_smape"],
                row["validation_mae"],
                row["validation_rmse"],
                row["input_dimension"],
            )

    logging.info("Results written to %s", args.results)


if __name__ == "__main__":
    main()
