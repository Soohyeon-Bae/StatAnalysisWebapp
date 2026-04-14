from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scatter_plot(df: pd.DataFrame, x: str, y: str):
    return px.scatter(df, x=x, y=y, trendline="ols")


def box_plot(df: pd.DataFrame, x: str, y: str):
    return px.box(df, x=x, y=y, points="outliers")


def violin_plot(df: pd.DataFrame, x: str, y: str):
    return px.violin(df, x=x, y=y, box=True, points="outliers")


def stacked_bar_plot(table: pd.DataFrame):
    long = table.reset_index().melt(id_vars=table.index.name or "index")
    return px.bar(long, x=long.columns[0], y="value", color="variable", barmode="stack")


def heatmap_plot(table: pd.DataFrame):
    return px.imshow(table, text_auto=True, aspect="auto")


def coefficient_bar_plot(coef_table: pd.DataFrame):
    col = "Coef." if "Coef." in coef_table.columns else coef_table.columns[1]
    plot_df = coef_table[coef_table["term"] != "Intercept"].copy().head(20)
    return px.bar(plot_df, x="term", y=col)


def did_trend_plot(means: pd.DataFrame, outcome: str):
    plot_df = means.copy()
    plot_df["treated_label"] = plot_df["treated"].map({0.0: "Control", 1.0: "Treated"}).fillna(plot_df["treated"].astype(str))
    plot_df["post_label"] = plot_df["post"].map({0.0: "Pre", 1.0: "Post"}).fillna(plot_df["post"].astype(str))
    return px.line(plot_df, x="post_label", y="mean", color="treated_label", markers=True, title=f"{outcome}: group means by pre/post")


def event_study_plot(event_table: pd.DataFrame):
    plot_df = event_table.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["period"], y=plot_df["coef"], mode="lines+markers", name="Estimate",
        error_y=dict(type="data", symmetric=False,
                     array=(plot_df["ci_high"] - plot_df["coef"]).fillna(0),
                     arrayminus=(plot_df["coef"] - plot_df["ci_low"]).fillna(0))
    ))
    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=-1, line_dash="dot")
    fig.update_layout(xaxis_title="Relative period", yaxis_title="Coefficient")
    return fig


def kaplan_meier_plot(km_table: pd.DataFrame):
    return px.line(km_table, x="time", y="survival_probability", color="group", markers=False)


def propensity_score_plot(ps_df: pd.DataFrame, matched: bool = False):
    title = "Matched propensity score distribution" if matched else "Propensity score distribution"
    return px.histogram(ps_df, x="propensity_score", color="treated_label", barmode="overlay", opacity=0.6, nbins=30, title=title)


def love_plot(balance_df: pd.DataFrame):
    plot_df = balance_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["smd_before"], y=plot_df["covariate"], mode="markers", name="Before"))
    fig.add_trace(go.Scatter(x=plot_df["smd_after"], y=plot_df["covariate"], mode="markers", name="After"))
    fig.add_vline(x=0.1, line_dash="dash")
    fig.add_vline(x=-0.1, line_dash="dash")
    fig.update_layout(xaxis_title="Standardized mean difference", yaxis_title="Covariate")
    return fig
