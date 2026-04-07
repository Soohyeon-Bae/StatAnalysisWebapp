from __future__ import annotations

from io import BytesIO
from typing import Dict

import pandas as pd



def build_excel_report(frames: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in frames.items():
            safe_name = str(sheet_name)[:31] or "Sheet1"
            df.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.read()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
