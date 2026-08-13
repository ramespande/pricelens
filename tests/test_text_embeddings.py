from pathlib import Path

import pandas as pd
import pytest

from src.features.text_embeddings import TextEmbeddingConfig, embedding_dimension, select_embedding_rows


def test_embedding_dimension_for_supported_encoder():
    assert embedding_dimension("sentence-transformers/all-MiniLM-L6-v2") == 384


def test_select_embedding_rows_can_restrict_to_manifest(tmp_path: Path):
    frame = pd.DataFrame({"sample_id": [1, 2, 3], "catalog_content": ["a", "b", "c"]})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"sample_id": [2, 3]}).to_csv(manifest, index=False)
    config = TextEmbeddingConfig(data_root=tmp_path, manifest_source=manifest)
    selected = select_embedding_rows(frame, config)
    assert selected.sample_id.tolist() == [2, 3]


def test_text_embedding_config_rejects_unknown_encoder(tmp_path: Path):
    with pytest.raises(ValueError):
        TextEmbeddingConfig(data_root=tmp_path, encoder="unknown/model")
