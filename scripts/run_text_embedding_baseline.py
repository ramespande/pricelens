"""Run semantic text-embedding baselines on cached sentence-transformer vectors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.text_embedding_baselines import run_text_embedding_baseline

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", required=True)
parser.add_argument("--config", default="configs/text_embedding_baseline.json")
parser.add_argument("--results-path", default="experiments/results/text_embedding_baseline.csv")
args = parser.parse_args()
for result in run_text_embedding_baseline(args.data_root, args.config, args.results_path):
    print(result)
