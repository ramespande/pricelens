"""Run the leakage-aware, image-only 1,000-image pilot."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.image_baselines import run_image_only_pilot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", required=True)
parser.add_argument("--config", default="configs/image_baseline_pilot.json")
args = parser.parse_args()
for result in run_image_only_pilot(args.data_root, args.config):
    print(result)
