from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import pandas as pd

from .type_inference import clean_column_names, normalize_missing_values


ALLOWED_SUFFIXES = (".csv", ".xlsx")


def get_excel_sheet_names(uploaded_file) -> List[str]:
    data = uploaded_file.getvalue()
    xls = pd.ExcelFile(BytesIO(data))
    return xls.sheet_names


def load_uploaded_file(uploaded_file, sheet_name: str | None = None) -> Tuple[pd.DataFrame, str]:
    filename = uploaded_file.name.lower()
    if not filename.endswith(ALLOWED_SUFFIXES):
        raise ValueError("지원하지 않는 파일 형식입니다. CSV 또는 XLSX만 업로드해주세요.")

    raw = uploaded_file.getvalue()
    if filename.endswith(".csv"):
        df = _read_csv_fallback(raw)
        active_sheet = "csv"
    else:
        # 자동 datetime 추론을 줄이기 위해 object 우선
        df = pd.read_excel(BytesIO(raw), sheet_name=sheet_name or 0, dtype=object)
        active_sheet = str(sheet_name or 0)

    if df is None or df.empty:
        raise ValueError("파일을 읽었지만 데이터가 비어 있습니다.")

    df = clean_column_names(df)
    df = normalize_missing_values(df)

    # 숫자처럼 보이는 열만 숫자로 재변환
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            converted = pd.to_numeric(s, errors="coerce")
            ratio = converted.notna().mean() if len(s) > 0 else 0
            if ratio >= 0.9:
                df[col] = converted

    return df, active_sheet


def _read_csv_fallback(raw: bytes) -> pd.DataFrame:
    tried = []
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(BytesIO(raw), encoding=enc, dtype=object)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{enc}: {e}")
    raise ValueError("CSV 인코딩을 해석하지 못했습니다. 시도한 인코딩: " + " | ".join(tried))
