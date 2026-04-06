from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scatter_plot(df: pd.DataFrame, x: str, y: str):
    clean = df[[x, y]].dropna()
    return px.scatter(clean, x=x, y=y, trendline="ols", title=f"{x} vs {y}")


def box_plot(df: pd.DataFrame, x: str, y: str):
    clean = df[[x, y]].dropna()
    return px.box(clean, x=x, y=y, points="all", title=f"{y} by {x}")


def violin_plot(df: pd.DataFrame, x: str, y: str):
    clean = df[[x, y]].dropna()
    return px.violin(clean, x=x, y=y, box=True, points="all", title=f"{y} by {x}")


def stacked_bar_plot(table: pd.DataFrame):
    long_df = table.reset_index().melt(id_vars=table.index.name or "index", var_name="column", value_name="count")
    id_col = table.index.name or "index"
    return px.bar(long_df, x=id_col, y="count", color="column", barmode="stack", title="Categorical Distribution")


def heatmap_plot(table: pd.DataFrame):
    return px.imshow(table, text_auto=True, aspect="auto", title="Contingency Table Heatmap")


def coefficient_bar_plot(coef_df: pd.DataFrame, coef_col: str = "Coef."):
    if coef_col not in coef_df.columns:
        return go.Figure()
    plot_df = coef_df[coef_df["term"] != "Intercept"].copy()
    fig = px.bar(plot_df, x="term", y=coef_col, title="Coefficients")
    fig.update_layout(xaxis_title="term", yaxis_title="coefficient")
    return fig


def did_trend_plot(means_df: pd.DataFrame, outcome: str):
    plot_df = means_df.copy()
    plot_df["treated_label"] = plot_df["treated"].map({0.0: "Control", 1.0: "Treated"}).fillna(plot_df["treated"].astype(str))
    return px.line(plot_df, x="post", y="mean", color="treated_label", markers=True, title=f"Pre/Post means of {outcome}")


def event_study_plot(event_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=event_df["period"], y=event_df["coef"], mode="lines+markers", name="estimate"))
    if {"ci_low", "ci_high"}.issubset(event_df.columns):
        fig.add_trace(go.Scatter(x=event_df["period"], y=event_df["ci_high"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=event_df["period"], y=event_df["ci_low"], mode="lines", line=dict(width=0), fill="tonexty", name="95% CI"))
    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=-1, line_dash="dot")
    fig.update_layout(title="Event Study Coefficients", xaxis_title="Relative period", yaxis_title="Estimate")
    return fig


def kaplan_meier_plot(km_df: pd.DataFrame):
    return px.line(km_df, x="time", y="survival_probability", color="group", title="Kaplan-Meier Survival Curves")
