"""Deterministic, CPU-friendly structured features from catalog_content."""
from __future__ import annotations
import re
import pandas as pd

NUMERIC_TOKEN = re.compile(r"\b\d+(?:\.\d+)?\b")
PACK_QUANTITY = re.compile(r"\b[Pp]ack\s+[Oo]f\s+(\d+)\b|\b(\d+)\s+[Pp]er\s+[Cc]ase\b")

def _field(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.*)$", text, flags=re.MULTILINE)
    return match.group(1) if match else ""

def extract_text_features(content: pd.Series) -> pd.DataFrame:
    """Extract documented counts and conservative fields from catalog content.

    ``title_length`` uses the value after an ``Item Name:`` line. ``description_length``
    uses ``Product Description:`` when present. ``item_pack_quantity`` is populated only
    for explicit 'Pack of N' or 'N per case' patterns; absent/ambiguous values are 0.
    """
    text = content.fillna("").astype(str)
    rows = []
    for value in text:
        words = re.findall(r"\b\w+\b", value)
        numeric_tokens = NUMERIC_TOKEN.findall(value)
        pack = PACK_QUANTITY.search(value)
        pack_quantity = int(next((item for item in pack.groups() if item), 0)) if pack else 0
        rows.append({
            "char_count": len(value), "word_count": len(words), "digit_count": sum(ch.isdigit() for ch in value),
            "uppercase_count": sum(ch.isupper() for ch in value), "numeric_token_count": len(numeric_tokens),
            "punctuation_count": sum(not ch.isalnum() and not ch.isspace() for ch in value), "whitespace_count": sum(ch.isspace() for ch in value),
            "average_word_length": (sum(map(len, words)) / len(words)) if words else 0.0,
            "title_length": len(_field(value, "Item Name")), "description_length": len(_field(value, "Product Description")),
            "item_pack_quantity": pack_quantity,
        })
    return pd.DataFrame(rows, index=content.index)
