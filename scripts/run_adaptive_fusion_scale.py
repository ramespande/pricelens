"""Run adaptive (gated) MLP fusion at scale 5000.

Trains a shallow MLP gating network on the concatenated [text; image] embedding
to learn per-sample modality weights. The gate is trained entirely within the
training partition (with a 15% inner hold-out for early stopping); the outer
validation holdout is not used during gate fitting.

Both embedding caches must already exist; this script will not re-extract them.

Usage:
    python scripts/run_adaptive_fusion_scale.py --data-root <DATA_ROOT>

Options:
    --data-root    Path to the directory containing train.csv and test.csv.
    --config       Config JSON path (default: configs/semantic_fusion_scale_5000.json).
    --results      Output CSV path (default: experiments/results/adaptive_fusion_scale_5000.csv).
    --epochs       Max training epochs for the gate (default: 200).
    --hidden       Hidden units in the gate MLP (default: 128).
    --batch-size   Mini-batch size for Adam (default: 256).
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
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/semantic_fusion_scale_5000.json")
    parser.add_argument("--results", default="experiments/results/adaptive_fusion_scale_5000.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    from src.training.adaptive_fusion import run_adaptive_fusion

    logging.info("=== Adaptive MLP Gate Fusion (scale 5000) ===")
    logging.info("  epochs=%d  hidden=%d  batch_size=%d", args.epochs, args.hidden, args.batch_size)
    rows = run_adaptive_fusion(
        args.data_root,
        config_path=args.config,
        results_path=args.results,
        epochs=args.epochs,
        hidden=args.hidden,
        batch_size=args.batch_size,
    )
    for row in rows:
        logging.info(
            "  %-35s  SMAPE=%.3f  MAE=%.3f  RMSE=%.3f",
            row["experiment_name"],
            row["validation_smape"],
            row["validation_mae"],
            row["validation_rmse"],
        )
        logging.info(
            "  gate_mean_val=%.3f  gate_std_val=%.3f  stopped_epoch=%d  inner_val_mse=%.5f",
            row["gate_mean_val"],
            row["gate_std_val"],
            row["gate_stopped_epoch"],
            row["gate_best_inner_val_mse"],
        )

    logging.info("Results written to %s", args.results)


if __name__ == "__main__":
    main()
