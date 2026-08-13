"""Interfaces for future cached image embeddings; no image processing occurs here."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ImageEmbeddingJob:
    image_root: Path
    output_dir: Path
    batch_size: int = 32
    resume: bool = True
    failed_log_name: str = "failed_images.csv"
