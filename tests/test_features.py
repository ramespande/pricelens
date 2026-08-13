import pandas as pd
from src.features.text import extract_text_features

def test_text_features_counts_and_explicit_pack_quantity():
    value="Item Name: Sample (Pack of 6)\nProduct Description: Details 12.5"
    features=extract_text_features(pd.Series([value]))
    assert features.loc[0,"item_pack_quantity"] == 6
    assert features.loc[0,"title_length"] == len("Sample (Pack of 6)")
    assert features.loc[0,"numeric_token_count"] == 2
