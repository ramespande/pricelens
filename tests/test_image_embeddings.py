from pathlib import Path
import pandas as pd
import pytest
import numpy as np
from src.features.image_embeddings import VisionPilotConfig, _atomic_save_embeddings, image_filename, select_pilot_rows

def test_image_filename_extracts_url_basename():
    assert image_filename("https://example.test/a/b/name.jpg?x=1") == "name.jpg"

def test_select_pilot_rows_is_deterministic_and_deduplicates_images(tmp_path: Path):
    images = tmp_path / "train" / "images"; images.mkdir(parents=True)
    for name in ("one.jpg", "two.jpg"): (images / name).touch()
    frame = pd.DataFrame({"sample_id":[3,1,2], "image_link":["https://x/one.jpg", "https://x/one.jpg", "https://x/two.jpg"]})
    config = VisionPilotConfig(data_root=tmp_path, sample_size=2)
    first, second = select_pilot_rows(frame, config), select_pilot_rows(frame, config)
    assert first.sample_id.tolist() == second.sample_id.tolist()
    assert len(first) == 2 and first.image_link.nunique() == 2

def test_select_pilot_rows_honours_explicit_sample_size(tmp_path: Path):
    images = tmp_path / "train" / "images"; images.mkdir(parents=True)
    for name in ("one.jpg", "two.jpg"): (images / name).touch()
    frame = pd.DataFrame({"sample_id":[1,2], "image_link":["https://x/one.jpg", "https://x/two.jpg"]})
    assert len(select_pilot_rows(frame, VisionPilotConfig(data_root=tmp_path, sample_size=2), sample_size=1)) == 1

def test_select_pilot_rows_requires_existing_images(tmp_path: Path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    frame = pd.DataFrame({"sample_id":[1], "image_link":["https://x/nope.jpg"]})
    with pytest.raises(ValueError): select_pilot_rows(frame, VisionPilotConfig(data_root=tmp_path, sample_size=1))

def test_atomic_embedding_save_replaces_complete_array(tmp_path: Path):
    path = tmp_path / "embeddings.npy"
    _atomic_save_embeddings(path, np.ones((2, 3), dtype=np.float32))
    _atomic_save_embeddings(path, np.zeros((3, 3), dtype=np.float32))
    assert np.load(path).shape == (3, 3)
