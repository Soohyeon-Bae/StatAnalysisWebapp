from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from .effect_size import cohens_d, cramers_v, eta_squared_from_anova


@dataclass
class AnalysisResult:
    summary: Dict[str, Any]
    stats_table: pd.DataFrame
    detail_tables: Dict[str, pd.DataFrame]
    narrative: str
    recommended_test: str


class AnalysisError(ValueError):
    pass


CONTINUOUS = {"continuous"}
CATEGORICAL = {"categorical", "binary"}



def recommend_analysis(x_type: str, y_type: str, unique_y: int | None = None, unique_x: int | None = None) -> str:
    if x_type in CONTINUOUS and y_type in CONTINUOUS:
        return "Correlation"
    if x_type in CATEGORICAL and y_type in CONTINUOUS:
        return "Group Difference"
    if x_type in CONTINUOUS and y_type in CATEGORICAL:
        return "Group Difference"
    if x_type in CATEGORICAL and y_type in CATEGORICAL:
        return "Categorical Association"
    return "Unsupported"



def run_bivariate_analysis(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_type: str,
    y_type: str,
    alpha: float = 0.05,
    missing: str = "drop_rows",
    method_override: Optional[str] = None,
) -> AnalysisResult:
    work = df[[x, y]].copy()
    work = _prepare_series(work, x, x_type)
    work = _prepare_series(work, y, y_type)
    work = work.dropna()
    if len(work) < 3:
        raise AnalysisError("유효 표본 수가 너무 적어서 분석할 수 없습니다.")

    recommendation = recommend_analysis(x_type, y_type, work[y].nunique(), work[x].nunique())
    if recommendation == "Correlation":
        return _run_correlation(work, x, y, alpha, method_override)
    if recommendation == "Group Difference":
        if x_type in CONTINUOUS and y_type in CATEGORICAL:
            return _run_group_difference(work, group=y, value=x, alpha=alpha, method_override=method_override)
        return _run_group_difference(work, group=x, value=y, alpha=alpha, method_override=method_override)
    if recommendation == "Categorical Association":
        return _run_categorical_association(work, x, y, alpha, method_override)
    raise AnalysisError("선택한 변수 조합은 현재 지원되지 않습니다.")



def run_regression(
    df: pd.DataFrame,
    dependent: str,
    independents: list[str],
    type_map: dict[str, str],
    family: str,
) -> AnalysisResult:
    if not independents:
        raise AnalysisError("설명변수를 하나 이상 선택해주세요.")

    cols = [dependent] + independents
    work = df[cols].copy()
    for col in cols:
        work = _prepare_series(work, col, type_map[col])
    work = work.dropna()
    if len(work) < max(10, len(independents) + 3):
        raise AnalysisError("회귀분석을 수행하기에 표본 수가 부족합니다.")

    formula_terms = []
    for col in independents:
        if type_map[col] in CATEGORICAL:
            formula_terms.append(f"C(Q('{col}'))")
        else:
            formula_terms.append(f"Q('{col}')")
    formula = f"Q('{dependent}') ~ " + " + ".join(formula_terms)

    if family == "linear":
        model = smf.ols(formula=formula, data=work).fit()
        coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        summary = {
            "analysis": "Linear Regression",
            "n": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        }
        narrative = (
            f"선형회귀 결과, 표본 {int(model.nobs)}건 기준 모형의 R²는 {model.rsquared:.3f}입니다. "
            "회귀계수표에서 각 변수의 방향과 유의성을 함께 확인하세요."
        )
    elif family == "logistic":
        model = smf.logit(formula=formula, data=work).fit(disp=False)
        coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        coef["odds_ratio"] = np.exp(coef["Coef."])
        summary = {
            "analysis": "Logistic Regression",
            "n": int(model.nobs),
            "pseudo_r_squared": float(model.prsquared),
            "llr_pvalue": float(model.llr_pvalue),
        }
        narrative = (
            f"로지스틱 회귀 결과, 표본 {int(model.nobs)}건 기준 McFadden pseudo R²는 {model.prsquared:.3f}입니다. "
            "odds ratio가 1보다 크면 종속변수=1의 가능성이 증가하는 방향입니다."
        )
    else:
        raise AnalysisError("지원하지 않는 회귀 유형입니다.")

    return AnalysisResult(
        summary=summary,
        stats_table=coef,
        detail_tables={"model_data_preview": work.head(20)},
        narrative=narrative,
        recommended_test=summary["analysis"],
    )



def run_bulk_pairwise(df: pd.DataFrame, selected_columns: list[str], type_map: dict[str, str], correction: str) -> AnalysisResult:
    rows = []
    detail_tables: Dict[str, pd.DataFrame] = {}
    for i in range(len(selected_columns)):
        for j in range(i + 1, len(selected_columns)):
            x, y = selected_columns[i], selected_columns[j]
            x_type, y_type = type_map[x], type_map[y]
            try:
                result = run_bivariate_analysis(df, x, y, x_type, y_type)
                rows.append(
                    {
                        "x": x,
                        "y": y,
                        "test": result.summary.get("test"),
                        "statistic": result.summary.get("statistic"),
                        "p_value": result.summary.get("p_value"),
                        "effect_size": result.summary.get("effect_size"),
                        "n": result.summary.get("n"),
                    }
                )
            except Exception:
                rows.append({"x": x, "y": y, "test": "Unsupported/Failed", "statistic": np.nan, "p_value": np.nan, "effect_size": np.nan, "n": np.nan})
    table = pd.DataFrame(rows)
    valid = table["p_value"].notna()
    if valid.any() and correction != "none":
        _, qvals, _, _ = multipletests(table.loc[valid, "p_value"].values, method=correction)
        table.loc[valid, "adjusted_p_value"] = qvals
    else:
        table["adjusted_p_value"] = table["p_value"]

    narrative = (
        f"총 {len(table)}개의 변수 쌍을 탐색했습니다. 다중비교 보정 방법은 {correction}입니다. "
        "adjusted_p_value를 기준으로 해석하는 것이 안전합니다."
    )
    return AnalysisResult(
        summary={"analysis": "Bulk Pairwise Screening", "pairs_tested": len(table), "correction": correction},
        stats_table=table,
        detail_tables=detail_tables,
        narrative=narrative,
        recommended_test="Bulk Pairwise Screening",
    )



def _prepare_series(df: pd.DataFrame, col: str, col_type: str) -> pd.DataFrame:
    if col_type == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].map(lambda v: v.toordinal() if pd.notna(v) else np.nan)
    elif col_type in CONTINUOUS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif col_type in CATEGORICAL:
        df[col] = df[col].astype("string")
    return df



def _run_correlation(work: pd.DataFrame, x: str, y: str, alpha: float, method_override: Optional[str]) -> AnalysisResult:
    clean = work[[x, y]].dropna()
    if clean[x].nunique() < 2 or clean[y].nunique() < 2:
        raise AnalysisError("상관분석을 하려면 두 변수 모두 두 개 이상의 값이 필요합니다.")
    method = method_override or "Pearson"
    if method == "Pearson":
        stat, p = stats.pearsonr(clean[x], clean[y])
    else:
        stat, p = stats.spearmanr(clean[x], clean[y], nan_policy="omit")
    stats_table = pd.DataFrame(
        [{"x": x, "y": y, "correlation": stat, "p_value": p, "n": len(clean), "method": method}]
    )
    narrative = (
        f"{method} 상관분석 결과, {x}와 {y}의 상관계수는 {stat:.3f}, p-value는 {p:.4g}입니다. "
        + ("통계적으로 유의한 관계가 관찰됩니다." if p < alpha else "통계적으로 유의하다고 보기는 어렵습니다.")
    )
    return AnalysisResult(
        summary={"test": f"{method} correlation", "statistic": float(stat), "p_value": float(p), "effect_size": float(stat), "n": len(clean)},
        stats_table=stats_table,
        detail_tables={"descriptive_stats": clean.describe().reset_index()},
        narrative=narrative,
        recommended_test=f"{method} correlation",
    )



def _run_group_difference(work: pd.DataFrame, group: str, value: str, alpha: float, method_override: Optional[str]) -> AnalysisResult:
    clean = work[[group, value]].dropna()
    groups = list(clean[group].astype(str).unique())
    if len(groups) < 2:
        raise AnalysisError("그룹 비교를 하려면 최소 두 개 집단이 필요합니다.")
    if len(groups) > 10:
        raise AnalysisError("집단 수가 너무 많아 현재 화면에서는 비교하기 어렵습니다. 범주 수를 줄여주세요.")

    samples = [pd.to_numeric(clean.loc[clean[group].astype(str) == g, value], errors="coerce").dropna().values for g in groups]
    desc = clean.groupby(group)[value].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()

    if len(groups) == 2:
        method = method_override or "t-test"
        if method == "Mann-Whitney U":
            stat, p = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided", method="auto")
            eff = cohens_d(samples[0], samples[1])
        else:
            levene_p = stats.levene(samples[0], samples[1]).pvalue if len(samples[0]) >= 2 and len(samples[1]) >= 2 else np.nan
            equal_var = False if pd.notna(levene_p) and levene_p < 0.05 else True
            stat, p = stats.ttest_ind(samples[0], samples[1], equal_var=equal_var, nan_policy="omit")
            eff = cohens_d(samples[0], samples[1])
            method = "Welch t-test" if not equal_var else "Student t-test"
        narrative = (
            f"두 집단 비교 결과 {method}의 p-value는 {p:.4g}입니다. "
            f"효과크기(Cohen's d)는 {eff:.3f}입니다."
        )
        stats_table = pd.DataFrame([
            {"group_variable": group, "value_variable": value, "test": method, "statistic": stat, "p_value": p, "effect_size": eff, "n": len(clean)}
        ])
        summary = {"test": method, "statistic": float(stat), "p_value": float(p), "effect_size": float(eff), "n": len(clean)}
    else:
        method = method_override or "ANOVA"
        if method == "Kruskal-Wallis":
            stat, p = stats.kruskal(*samples, nan_policy="omit")
            eff = np.nan
        else:
            stat, p = stats.f_oneway(*samples)
            df_between = len(groups) - 1
            df_within = len(clean) - len(groups)
            eff = eta_squared_from_anova(stat, df_between, df_within)
            method = "One-way ANOVA"
        narrative = (
            f"{len(groups)}개 집단 비교 결과 {method}의 p-value는 {p:.4g}입니다. "
            + (f"효과크기(eta²)는 {eff:.3f}입니다." if pd.notna(eff) else "비모수 검정에서는 효과크기를 별도 계산하지 않았습니다.")
        )
        stats_table = pd.DataFrame([
            {"group_variable": group, "value_variable": value, "test": method, "statistic": stat, "p_value": p, "effect_size": eff, "n": len(clean)}
        ])
        summary = {"test": method, "statistic": float(stat), "p_value": float(p), "effect_size": float(eff) if pd.notna(eff) else np.nan, "n": len(clean)}

    return AnalysisResult(
        summary=summary,
        stats_table=stats_table,
        detail_tables={"group_descriptives": desc},
        narrative=narrative,
        recommended_test=summary["test"],
    )



def _run_categorical_association(work: pd.DataFrame, x: str, y: str, alpha: float, method_override: Optional[str]) -> AnalysisResult:
    clean = work[[x, y]].dropna().astype(str)
    table = pd.crosstab(clean[x], clean[y])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise AnalysisError("범주형 연관 분석을 위해서는 최소 2x2 교차표가 필요합니다.")

    if method_override == "Fisher exact" or (table.shape == (2, 2) and (table < 5).any().any()):
        if table.shape != (2, 2):
            raise AnalysisError("Fisher exact test는 현재 2x2 표만 지원합니다.")
        stat, p = stats.fisher_exact(table.to_numpy())
        test = "Fisher exact"
    else:
        stat, p, dof, expected = stats.chi2_contingency(table)
        test = "Chi-square"
    eff = cramers_v(table)
    row_pct = table.div(table.sum(axis=1), axis=0).reset_index()
    stats_table = pd.DataFrame([
        {"x": x, "y": y, "test": test, "statistic": stat, "p_value": p, "effect_size": eff, "n": int(table.to_numpy().sum())}
    ])
    narrative = (
        f"{test} 결과 p-value는 {p:.4g}이며, Cramér's V는 {eff:.3f}입니다. "
        + ("범주 간 분포 차이가 통계적으로 유의합니다." if p < alpha else "뚜렷한 분포 차이가 있다고 보기는 어렵습니다.")
    )
    return AnalysisResult(
        summary={"test": test, "statistic": float(stat), "p_value": float(p), "effect_size": float(eff), "n": int(table.to_numpy().sum())},
        stats_table=stats_table,
        detail_tables={"contingency_table": table.reset_index(), "row_percentages": row_pct},
        narrative=narrative,
        recommended_test=test,
    )
