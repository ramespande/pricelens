"""Resumable, local image-embedding extraction for bounded vision pilots.

The implementation intentionally imports PyTorch only while an extraction command runs.
It uses URL basenames to resolve the provided local JPEG files and never downloads or
modifies source images. Embeddings are cached as one NumPy file plus a sample manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionPilotConfig:
    data_root: Path
    split: str = "train"
    sample_size: int = 500
    random_seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    device: str = "cpu"
    encoder: str = "resnet18_imagenet1k_v1"
    output_dir: Path = Path("data/processed/image_embeddings/resnet18_train_pilot")

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if self.sample_size < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("sample_size and batch_size must be positive; num_workers cannot be negative")
        if self.encoder != "resnet18_imagenet1k_v1":
            raise ValueError("Milestone 2 pilot currently supports only resnet18_imagenet1k_v1")


def image_directory(config: VisionPilotConfig) -> Path:
    """Resolve the known local image layout without traversing it."""
    return config.data_root / ("train/images" if config.split == "train" else "test/test")


def image_filename(image_link: str) -> str:
    """Return the image filename portion of a dataset image link."""
    name = Path(urlparse(image_link).path).name
    if not name:
        raise ValueError(f"No filename in image link: {image_link!r}")
    return name


def select_pilot_rows(frame: pd.DataFrame, config: VisionPilotConfig, sample_size: int | None = None) -> pd.DataFrame:
    """Deterministically select unique, locally available image rows for a pilot."""
    root = image_directory(config)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    selected = frame[["sample_id", "image_link"]].drop_duplicates("image_link").copy()
    selected["image_path"] = selected["image_link"].map(lambda link: root / image_filename(link))
    selected = selected[selected.image_path.map(Path.is_file)]
    requested = sample_size or config.sample_size
    if len(selected) < requested:
        raise ValueError(f"Only {len(selected)} locally available unique images; need {requested}")
    return selected.sample(n=requested, random_state=config.random_seed).sort_values("sample_id").reset_index(drop=True)


def _config_fingerprint(config: VisionPilotConfig) -> str:
    payload = asdict(config)
    payload["data_root"] = str(config.data_root.resolve())
    payload["output_dir"] = str(config.output_dir)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _existing_manifest(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "manifest.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["sample_id", "image_link", "image_path"])


def _write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    if not failures:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "image_path", "error"])
        if not exists:
            writer.writeheader()
        writer.writerows(failures)


def _atomic_save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    """Write a complete NPY file before atomically replacing the previous cache."""
    temporary = path.with_name(f"{path.stem}.tmp.npy")
    np.save(temporary, embeddings)
    os.replace(temporary, path)


def _atomic_write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a complete manifest before atomically replacing the previous version."""
    temporary = path.with_name(f"{path.stem}.tmp.csv")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_model(device: str):
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise RuntimeError("Install torch, torchvision, and Pillow before extracting embeddings.") from error
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return torch, model, weights.transforms()


def extract_embeddings(rows: pd.DataFrame, config: VisionPilotConfig) -> dict[str, int]:
    """Extract a resumable embedding cache for the selected rows.

    Re-running with the same config skips rows found in the manifest. The cache is
    atomically rewritten after each completed batch so a later run can resume safely.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = config.output_dir / "metadata.json"
    fingerprint = _config_fingerprint(config)
    if metadata_path.exists() and json.loads(metadata_path.read_text())["config_fingerprint"] != fingerprint:
        raise ValueError("Output directory belongs to a different vision-pilot configuration")
    existing = _existing_manifest(config.output_dir)
    remaining = rows[~rows.sample_id.isin(existing.sample_id)].copy()
    metadata_path.write_text(json.dumps({"config_fingerprint": fingerprint, "config": {**asdict(config), "data_root": str(config.data_root), "output_dir": str(config.output_dir)}, "embedding_dimension": 512}, indent=2), encoding="utf-8")
    if remaining.empty:
        return {"requested": len(rows), "completed": len(existing), "failed": 0, "skipped": len(rows)}
    torch, model, transform = _load_model(config.device)
    from PIL import Image
    all_embeddings: list[np.ndarray] = []
    all_rows: list[dict[str, object]] = []
    if not existing.empty:
        all_embeddings.extend(np.load(config.output_dir / "embeddings.npy"))
        all_rows.extend(existing.to_dict("records"))
    failures: list[dict[str, str]] = []
    for start in range(0, len(remaining), config.batch_size):
        batch = remaining.iloc[start : start + config.batch_size]
        tensors, valid = [], []
        for row in batch.itertuples(index=False):
            try:
                with Image.open(row.image_path) as image:
                    tensors.append(transform(image.convert("RGB")))
                valid.append(row)
            except (OSError, ValueError) as error:
                failures.append({"sample_id": str(row.sample_id), "image_path": str(row.image_path), "error": str(error)})
        if tensors:
            with torch.inference_mode():
                output = model(torch.stack(tensors).to(config.device)).cpu().numpy().astype(np.float32)
            all_embeddings.extend(output)
            all_rows.extend({"sample_id": row.sample_id, "image_link": row.image_link, "image_path": str(row.image_path)} for row in valid)
            _atomic_save_embeddings(config.output_dir / "embeddings.npy", np.asarray(all_embeddings, dtype=np.float32))
            _atomic_write_manifest(config.output_dir / "manifest.csv", all_rows)
        LOGGER.info("Processed %s/%s rows", min(start + config.batch_size, len(remaining)), len(remaining))
    _write_failures(config.output_dir / "failed_images.csv", failures)
    return {"requested": len(rows), "completed": len(all_rows), "failed": len(failures), "skipped": len(existing)}
