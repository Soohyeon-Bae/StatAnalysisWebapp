from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


MISSING_STRINGS = {
    "",
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    ".",
    "-999",
    "missing",
}


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str
    non_null_count: int
    unique_count: int
    missing_count: int
    sample_values: List[str]


TYPE_OPTIONS = ["continuous", "categorical", "binary", "datetime", "text", "id", "exclude"]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = []
    seen: Dict[str, int] = {}
    for c in df.columns:
        col = str(c).strip() if c is not None else ""
        col = col or "unnamed"
        if col in seen:
            seen[col] += 1
            col = f"{col}__{seen[col]}"
        else:
            seen[col] = 0
        cols.append(col)
    df.columns = cols
    return df


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].apply(_normalize_cell)
    return out


def _normalize_cell(value):
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    if s.lower() in MISSING_STRINGS:
        return pd.NA
    return value


def infer_type(series: pd.Series) -> str:
    s = series.dropna()
    n = len(s)
    if n == 0:
        return "exclude"

    unique_count = s.nunique(dropna=True)
    if unique_count == 1:
        return "exclude"

    if _looks_like_datetime(series):
        return "datetime"

    # numeric handling
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean() if n > 0 else 0
    if numeric_ratio >= 0.9:
        unique_numeric = numeric.dropna().nunique()
        if unique_numeric == 2:
            return "binary"
        # treat low-cardinality integer codes as categorical by default
        if unique_numeric <= 10:
            integer_like = ((numeric.dropna() % 1) == 0).mean() >= 0.95
            if integer_like:
                return "categorical"
        return "continuous"

    # string-ish handling
    if unique_count == 2:
        return "binary"

    uniqueness_ratio = unique_count / max(n, 1)
    avg_len = s.astype(str).str.len().mean()

    if uniqueness_ratio > 0.9 and n > 20:
        return "id"
    if unique_count <= 20:
        return "categorical"
    if avg_len > 30:
        return "text"
    return "categorical"


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    s = series.dropna()
    if len(s) == 0:
        return False
    converted = pd.to_datetime(s, errors="coerce")
    return converted.notna().mean() >= 0.8


def build_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "inferred_type": infer_type(s),
                "non_null_count": int(s.notna().sum()),
                "unique_count": int(s.nunique(dropna=True)),
                "missing_count": int(s.isna().sum()),
                "sample_values": ", ".join(map(str, s.dropna().astype(str).head(5).tolist())),
            }
        )
    return pd.DataFrame(rows)
