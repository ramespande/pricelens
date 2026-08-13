"""Run training-only OOF-selected late fusion on the matched pilot."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.validated_late_fusion import run_oof_selected_late_fusion

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", required=True)
parser.add_argument("--config", default="configs/image_baseline_pilot.json")
parser.add_argument("--results-path", default="experiments/results/oof_late_fusion_pilot.csv")
args = parser.parse_args()
for result in run_oof_selected_late_fusion(args.data_root, args.config, args.results_path):
    print(result)
