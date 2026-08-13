from pathlib import Path
import json
import pandas as pd
import pytest
from src.training.matched_text_baselines import _matched_rows

def test_matched_rows_rejects_incomplete_manifest(tmp_path: Path):
    (tmp_path / "cache").mkdir()
    pd.DataFrame({"sample_id":[1]}).to_csv(tmp_path / "cache" / "manifest.csv", index=False)
    config = {"output_dir":str(tmp_path / "cache"),"validation_fraction":.5,"random_seed":42,"train_sample_size":1,"validation_sample_size":1}
    # Dataset files are absent, so the requested data-root error occurs before any result is fabricated.
    with pytest.raises(FileNotFoundError): _matched_rows(tmp_path, config)
