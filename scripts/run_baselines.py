"""Run CPU baselines; requires DATA_ROOT or --data-root."""
from __future__ import annotations
import argparse, os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.baselines import run_baselines

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
parser.add_argument("--config", default="configs/baseline.json")
args = parser.parse_args()
if not args.data_root: parser.error("Provide --data-root or set DATA_ROOT")
for row in run_baselines(args.data_root, args.config): print(row)
