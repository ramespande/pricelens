"""Run semantic late fusion (fixed 50/50 and OOF-selected) at scale 5000.

Uses cached semantic text embeddings (MiniLM-L6-v2) and image embeddings
(ResNet-18) for the pre-declared 4,000/1,000 matched sample (seed=2026).
Both embedding caches must already exist; this script will not re-extract them.

Usage:
    python scripts/run_semantic_late_fusion_scale.py --data-root <DATA_ROOT>

Options:
    --data-root   Path to the directory containing train.csv and test.csv.
    --config      Config JSON path (default: configs/semantic_fusion_scale_5000.json).
    --results     Output CSV path (default: experiments/results/semantic_fusion_scale_5000.csv).
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
    parser.add_argument("--config", default="configs/semantic_fusion_scale_5000.json")
    parser.add_argument("--results", default="experiments/results/semantic_fusion_scale_5000.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    from src.training.semantic_late_fusion import run_semantic_fixed_late_fusion, run_semantic_oof_late_fusion

    logging.info("=== Semantic Fixed 50/50 Late Fusion (scale 5000) ===")
    fixed_rows = run_semantic_fixed_late_fusion(args.data_root, args.config, args.results)
    for row in fixed_rows:
        logging.info(
            "  %-40s  SMAPE=%.3f  MAE=%.3f  RMSE=%.3f",
            row["experiment_name"],
            row["validation_smape"],
            row["validation_mae"],
            row["validation_rmse"],
        )

    logging.info("=== Semantic OOF-Selected Late Fusion (scale 5000) ===")
    oof_rows = run_semantic_oof_late_fusion(args.data_root, args.config, args.results)
    for row in oof_rows:
        logging.info(
            "  %-40s  SMAPE=%.3f  (text_w=%.2f  oof_smape=%.3f)",
            row["experiment_name"],
            row["validation_smape"],
            row["fusion_weight_text"],
            row["inner_oof_smape"],
        )

    logging.info("Results written to %s", args.results)


if __name__ == "__main__":
    main()
