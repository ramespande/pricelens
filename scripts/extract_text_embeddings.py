"""Extract a bounded or full resumable semantic text-embedding cache."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loading import load_train, resolve_data_root
from src.features.text_embeddings import TextEmbeddingConfig, extract_text_embeddings, select_embedding_rows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", required=True)
parser.add_argument("--config", default="configs/text_embedding_baseline.json")
parser.add_argument("--dry-run", action="store_true", help="Validate row selection without loading model weights.")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
raw["data_root"] = resolve_data_root(args.data_root)
raw["output_dir"] = Path(raw["output_dir"])
if raw.get("manifest_source"):
    raw["manifest_source"] = Path(raw["manifest_source"])
config = TextEmbeddingConfig(**raw)
rows = select_embedding_rows(load_train(config.data_root), config)
if args.dry_run:
    print(
        {
            "selected_rows": len(rows),
            "embedding_dimension": 384,
            "estimated_embedding_bytes": len(rows) * 384 * 4,
            "output_dir": str(config.output_dir),
            "encoder": config.encoder,
        }
    )
else:
    print(extract_text_embeddings(rows, config))
