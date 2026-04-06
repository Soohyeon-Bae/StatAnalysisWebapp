from __future__ import annotations

import pandas as pd
import plotly.express as px



def scatter_plot(df: pd.DataFrame, x: str, y: str):
    clean = df[[x, y]].dropna()
    return px.scatter(clean, x=x, y=y, trendline="ols", title=f"{x} vs {y}")



def box_plot(df: pd.DataFrame, x: str, y: str):
    clean = df[[x, y]].dropna()
    return px.box(clean, x=x, y=y, points="all", title=f"{x} by {y}")



def violin_plot(df: pd.DataFrame, x: str, y: str):
    clean = df[[x, y]].dropna()
    return px.violin(clean, x=x, y=y, box=True, points="all", title=f"{x} by {y}")



def stacked_bar_plot(table: pd.DataFrame):
    long_df = table.reset_index().melt(id_vars=table.index.name or "index", var_name="column", value_name="count")
    id_col = table.index.name or "index"
    return px.bar(
        long_df,
        x=id_col,
        y="count",
        color="column",
        barmode="stack",
        title="Categorical Distribution",
    )



def heatmap_plot(table: pd.DataFrame):
    return px.imshow(table, text_auto=True, aspect="auto", title="Contingency Table Heatmap")
