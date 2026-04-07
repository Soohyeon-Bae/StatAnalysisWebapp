from __future__ import annotations

import math

import numpy as np


def cohens_d(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def cramers_v(table) -> float:
    import scipy.stats as stats

    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, k = table.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return float("nan")
    return float(math.sqrt((chi2 / n) / denom))


def eta_squared_from_anova(f_stat: float, df_between: int, df_within: int) -> float:
    denom = (f_stat * df_between) + df_within
    if denom == 0:
        return float("nan")
    return float((f_stat * df_between) / denom)
