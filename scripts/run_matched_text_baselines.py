"""Run matched structured-text baselines against the cached image pilot."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.matched_text_baselines import run_matched_text_baselines

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", required=True)
parser.add_argument("--config", default="configs/image_baseline_pilot.json")
parser.add_argument("--results-path", default="experiments/results/matched_text_baselines.csv")
args = parser.parse_args()
for result in run_matched_text_baselines(args.data_root, args.config, args.results_path):
    print(result)
