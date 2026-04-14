from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import zipfile

import pandas as pd


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def build_excel_report(sheet_map: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheet_map.items():
            safe_name = str(name)[:31]
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.getvalue()


def build_full_zip(
    result_tables: dict[str, pd.DataFrame],
    interpretation_texts: dict[str, str],
    figures: dict[str, object],
    zip_filename: str = "analysis_results.zip",
) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        excel_path = tmp_path / "results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for name, df in result_tables.items():
                if isinstance(df, pd.DataFrame):
                    df.to_excel(writer, sheet_name=str(name)[:31], index=False)

        interp_path = tmp_path / "interpretation.txt"
        with open(interp_path, "w", encoding="utf-8") as f:
            for k, v in interpretation_texts.items():
                f.write(f"[{k}]\n")
                f.write(v)
                f.write("\n\n")

        plot_dir = tmp_path / "plots"
        plot_dir.mkdir(exist_ok=True)
        for name, fig in figures.items():
            try:
                fig.write_image(str(plot_dir / f"{name}.png"))
            except Exception:
                fig.write_html(str(plot_dir / f"{name}.html"), include_plotlyjs="cdn")

        mem = BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in tmp_path.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(tmp_path))
        mem.seek(0)
        return mem.getvalue()
