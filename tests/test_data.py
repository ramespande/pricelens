import pandas as pd
import pytest
from src.data.loading import validate_schema
from src.data.quality import data_quality_report, leakage_aware_split

def test_schema_validation():
    validate_schema(pd.DataFrame({"sample_id":[1],"catalog_content":["x"],"image_link":["y"],"price":[1.0]}), split="train")
    with pytest.raises(ValueError): validate_schema(pd.DataFrame({"sample_id":[1]}), split="test")
def test_quality_report_counts_duplicates():
    frame=pd.DataFrame({"sample_id":[1,1],"catalog_content":["x","x"],"image_link":["a","a"],"price":[1.,1.]})
    assert data_quality_report(frame)["duplicate_rows"] == 1
def test_split_prevents_exact_content_and_image_leakage():
    frame=pd.DataFrame({"sample_id":range(6),"catalog_content":["a","a","b","c","d","e"],"image_link":["1","2","2","3","4","5"],"price":range(6)})
    split=leakage_aware_split(frame, validation_fraction=.5, random_seed=7)
    assert not (set(split.train.catalog_content)&set(split.validation.catalog_content))
    assert not (set(split.train.image_link)&set(split.validation.image_link))
