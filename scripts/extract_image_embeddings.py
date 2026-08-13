"""Extract a bounded, resumable vision-embedding pilot."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loading import load_split, resolve_data_root
from src.features.image_embeddings import VisionPilotConfig, extract_embeddings, select_pilot_rows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", required=True)
parser.add_argument("--config", default="configs/vision_pilot.json")
parser.add_argument("--dry-run", action="store_true", help="Validate selection and report resources without loading model weights.")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
raw["data_root"] = resolve_data_root(args.data_root)
raw["output_dir"] = Path(raw["output_dir"])
config = VisionPilotConfig(**raw)
rows = select_pilot_rows(load_split(config.data_root, config.split), config)
if args.dry_run:
    print({"selected_unique_images": len(rows), "embedding_dimension": 512, "estimated_embedding_bytes": len(rows) * 512 * 4, "output_dir": str(config.output_dir)})
else:
    print(extract_embeddings(rows, config))
