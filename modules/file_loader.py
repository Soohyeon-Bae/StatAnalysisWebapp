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
        # 핵심: 엑셀 자동 datetime 추론을 막기 위해 우선 object로 읽기
        df = pd.read_excel(BytesIO(raw), sheet_name=sheet_name or 0, dtype=object)
        active_sheet = str(sheet_name or 0)

    if df is None or df.empty:
        raise ValueError("파일을 읽었지만 데이터가 비어 있습니다.")

    df = clean_column_names(df)
    df = normalize_missing_values(df)

    # 핵심: 숫자처럼 생긴 열만 숫자로 변환
    # 함수 밖이 아니라 반드시 여기, df가 만들어진 뒤에 실행해야 함
    for col in df.columns:
        s = df[col]

        # 문자열/object 계열만 숫자 변환 시도
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            converted = pd.to_numeric(s, errors="coerce")

            # 숫자로 변환 가능한 비율이 충분히 높을 때만 실제 반영
            # 너무 공격적으로 바꾸면 ID/코드 열까지 숫자로 바뀔 수 있어서 0.9 기준 사용
            ratio = converted.notna().mean() if len(s) > 0 else 0
            if ratio >= 0.9:
                df[col] = converted

    return df, active_sheet


def _read_csv_fallback(raw: bytes) -> pd.DataFrame:
    tried = []
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]:
        try:
            # CSV도 동일하게 우선 object로 읽기
            return pd.read_csv(BytesIO(raw), encoding=enc, dtype=object)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{enc}: {e}")
    raise ValueError("CSV 인코딩을 해석하지 못했습니다. 시도한 인코딩: " + " | ".join(tried))
