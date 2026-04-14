from __future__ import annotations

import math
import numpy as np
import pandas as pd


def cohens_d(x, y) -> float:
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return math.nan
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0 or pd.isna(pooled):
        return math.nan
    return float((x.mean() - y.mean()) / math.sqrt(pooled))


def cramers_v(table: pd.DataFrame) -> float:
    n = table.to_numpy().sum()
    if n == 0:
        return math.nan
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(table)[0]
    r, k = table.shape
    denom = n * max(min(r - 1, k - 1), 1)
    return float(math.sqrt(chi2 / denom))


def eta_squared_from_anova(f_value: float, df_between: int, df_within: int) -> float:
    denom = f_value * df_between + df_within
    if denom == 0:
        return math.nan
    return float((f_value * df_between) / denom)
