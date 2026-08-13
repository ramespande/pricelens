"""Resumable, local semantic text-embedding extraction for text baselines.

The implementation intentionally imports sentence-transformers only while an extraction
command runs. Embeddings are cached as one NumPy file plus a sample manifest.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.image_embeddings import _atomic_save_embeddings, _atomic_write_manifest

LOGGER = logging.getLogger(__name__)

SUPPORTED_ENCODERS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


@dataclass(frozen=True)
class TextEmbeddingConfig:
    data_root: Path
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str = "cpu"
    output_dir: Path = Path("data/processed/text_embeddings/minilm_l6_v2_train_full")
    manifest_source: Path | None = None

    def __post_init__(self) -> None:
        if self.encoder not in SUPPORTED_ENCODERS:
            raise ValueError(f"Unsupported encoder: {self.encoder}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


def embedding_dimension(encoder: str) -> int:
    """Return the output dimension for a supported encoder."""
    return SUPPORTED_ENCODERS[encoder]


def select_embedding_rows(frame: pd.DataFrame, config: TextEmbeddingConfig) -> pd.DataFrame:
    """Select train rows to embed, optionally restricted to a cached manifest."""
    rows = frame[["sample_id", "catalog_content"]].copy()
    if config.manifest_source is not None:
        manifest = pd.read_csv(config.manifest_source, usecols=["sample_id"])
        rows = rows.merge(manifest, on="sample_id", how="inner", validate="one_to_one")
    return rows.sort_values("sample_id").reset_index(drop=True)


def _config_fingerprint(config: TextEmbeddingConfig) -> str:
    payload = asdict(config)
    payload["data_root"] = str(config.data_root.resolve())
    payload["output_dir"] = str(config.output_dir)
    payload["manifest_source"] = str(config.manifest_source) if config.manifest_source else None
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _existing_manifest(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "manifest.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["sample_id"])


def _load_model(encoder: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install sentence-transformers before extracting text embeddings.") from error
    return SentenceTransformer(encoder, device=device)


def extract_text_embeddings(rows: pd.DataFrame, config: TextEmbeddingConfig) -> dict[str, int]:
    """Extract a resumable text-embedding cache for the selected rows."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = config.output_dir / "metadata.json"
    fingerprint = _config_fingerprint(config)
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8"))["config_fingerprint"] != fingerprint:
        raise ValueError("Output directory belongs to a different text-embedding configuration")
    existing = _existing_manifest(config.output_dir)
    remaining = rows[~rows.sample_id.isin(existing.sample_id)].copy()
    metadata_path.write_text(
        json.dumps(
            {
                "config_fingerprint": fingerprint,
                "config": {
                    **asdict(config),
                    "data_root": str(config.data_root),
                    "output_dir": str(config.output_dir),
                    "manifest_source": str(config.manifest_source) if config.manifest_source else None,
                },
                "embedding_dimension": embedding_dimension(config.encoder),
                "encoder_documentation": {
                    "parameters": "22M",
                    "weights_megabytes": 80,
                    "license": "Apache-2.0",
                    "documentation": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if remaining.empty:
        return {"requested": len(rows), "completed": len(existing), "failed": 0, "skipped": len(rows)}
    model = _load_model(config.encoder, config.device)
    all_embeddings: list[np.ndarray] = []
    all_rows: list[dict[str, object]] = []
    if not existing.empty:
        all_embeddings.extend(np.load(config.output_dir / "embeddings.npy"))
        all_rows.extend({"sample_id": row.sample_id} for row in existing.itertuples(index=False))
    for start in range(0, len(remaining), config.batch_size):
        batch = remaining.iloc[start : start + config.batch_size]
        texts = batch.catalog_content.fillna("").astype(str).tolist()
        vectors = model.encode(texts, batch_size=len(texts), show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        all_embeddings.extend(vectors)
        all_rows.extend({"sample_id": row.sample_id} for row in batch.itertuples(index=False))
        _atomic_save_embeddings(config.output_dir / "embeddings.npy", np.asarray(all_embeddings, dtype=np.float32))
        _atomic_write_manifest(config.output_dir / "manifest.csv", all_rows)
        LOGGER.info("Processed %s/%s rows", min(start + config.batch_size, len(remaining)), len(remaining))
    return {"requested": len(rows), "completed": len(all_rows), "failed": 0, "skipped": len(existing)}
