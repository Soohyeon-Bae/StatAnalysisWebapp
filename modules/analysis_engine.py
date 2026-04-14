from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
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


def _prepare_series(df: pd.DataFrame, col: str, col_type: str) -> pd.DataFrame:
    if col_type in CONTINUOUS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif col_type == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce")
    elif col_type in CATEGORICAL:
        df[col] = df[col].astype("string").str.strip()
    return df


def _encode_binary_for_regression(series: pd.Series) -> tuple[pd.Series, dict[Any, int] | None]:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int), {False: 0, True: 1}
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        uniq = sorted(pd.Series(numeric.dropna().unique()).tolist())
        if len(uniq) == 2 and set(uniq).issubset({0, 1}):
            return numeric.astype(float), {uniq[0]: int(uniq[0]), uniq[1]: int(uniq[1])}
    s = series.astype("string").str.strip()
    uniq = sorted([u for u in s.dropna().unique().tolist()])
    if len(uniq) != 2:
        raise AnalysisError("로지스틱 회귀의 종속변수는 정확히 두 범주만 가져야 합니다.")
    lower_map = {str(u).lower(): u for u in uniq}
    if set(lower_map.keys()) == {"no", "yes"}:
        mapping = {lower_map["no"]: 0, lower_map["yes"]: 1}
    elif set(lower_map.keys()) == {"false", "true"}:
        mapping = {lower_map["false"]: 0, lower_map["true"]: 1}
    else:
        mapping = {uniq[0]: 0, uniq[1]: 1}
    return s.map(mapping).astype(float), mapping


def run_bivariate_analysis(df: pd.DataFrame, x: str, y: str, x_type: str, y_type: str, alpha: float = 0.05, missing: str = "drop_rows", method_override: Optional[str] = None) -> AnalysisResult:
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


def run_regression(df: pd.DataFrame, dependent: str, independents: list[str], type_map: dict[str, str], family: str) -> AnalysisResult:
    if not independents:
        raise AnalysisError("설명변수를 하나 이상 선택해주세요.")
    cols = [dependent] + independents
    work = df[cols].copy()
    for col in cols:
        work = _prepare_series(work, col, type_map[col])

    encoding_info = None
    if family == "logistic":
        work[dependent], encoding_info = _encode_binary_for_regression(work[dependent])

    work = work.dropna()
    if len(work) < max(20, len(independents) + 5):
        raise AnalysisError("회귀분석을 수행하기에 표본 수가 부족합니다.")

    formula_terms = []
    for col in independents:
        if type_map[col] in CATEGORICAL:
            formula_terms.append(f"C(Q('{col}'))")
        else:
            formula_terms.append(f"Q('{col}')")
    formula = f"Q('{dependent}') ~ " + " + ".join(formula_terms)

    if family == "linear":
        model = smf.ols(formula=formula, data=work).fit(cov_type="HC1")
        coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        summary = {"analysis": "Linear Regression", "n": int(model.nobs), "r_squared": float(model.rsquared), "adj_r_squared": float(model.rsquared_adj), "model_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan}
        narrative = f"선형회귀 결과, 표본 {int(model.nobs)}건 기준 모형의 R²는 {model.rsquared:.3f}입니다."
    elif family == "logistic":
        model = smf.logit(formula=formula, data=work).fit(disp=False)
        coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        coef["odds_ratio"] = np.exp(coef["Coef."])
        summary = {"analysis": "Logistic Regression", "n": int(model.nobs), "pseudo_r_squared": float(model.prsquared), "model_pvalue": float(model.llr_pvalue), "encoding_info": str(encoding_info) if encoding_info else None}
        narrative = f"로지스틱 회귀 결과, 표본 {int(model.nobs)}건 기준 McFadden pseudo R²는 {model.prsquared:.3f}입니다."
    else:
        raise AnalysisError("지원하지 않는 회귀 유형입니다.")

    detail_tables = {"model_data_preview": work.head(20)}
    if encoding_info is not None:
        detail_tables["dependent_encoding"] = pd.DataFrame([{"original_value": k, "encoded_value": v} for k, v in encoding_info.items()])

    return AnalysisResult(summary=summary, stats_table=coef, detail_tables=detail_tables, narrative=narrative, recommended_test=summary["analysis"])


def run_bulk_pairwise(df: pd.DataFrame, selected_columns: list[str], type_map: dict[str, str], correction: str) -> AnalysisResult:
    rows = []
    for i in range(len(selected_columns)):
        for j in range(i + 1, len(selected_columns)):
            x, y = selected_columns[i], selected_columns[j]
            try:
                result = run_bivariate_analysis(df, x, y, type_map[x], type_map[y])
                rows.append({"x": x, "y": y, "test": result.summary.get("test"), "statistic": result.summary.get("statistic"), "p_value": result.summary.get("p_value"), "effect_size": result.summary.get("effect_size"), "n": result.summary.get("n")})
            except Exception:
                rows.append({"x": x, "y": y, "test": "Unsupported/Failed", "statistic": np.nan, "p_value": np.nan, "effect_size": np.nan, "n": np.nan})
    table = pd.DataFrame(rows)
    valid = table["p_value"].notna()
    if valid.any() and correction != "none":
        _, qvals, _, _ = multipletests(table.loc[valid, "p_value"].values, method=correction)
        table.loc[valid, "adjusted_p_value"] = qvals
    else:
        table["adjusted_p_value"] = table["p_value"]
    return AnalysisResult(summary={"analysis": "Bulk Pairwise Screening", "pairs_tested": len(table), "correction": correction}, stats_table=table, detail_tables={}, narrative=f"총 {len(table)}개의 변수 쌍을 탐색했습니다.", recommended_test="Bulk Pairwise Screening")


def _ordered_time_codes(series: pd.Series) -> tuple[pd.Series, dict[Any, int]]:
    s = series.copy()
    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = pd.to_datetime(s, errors="coerce")
        uniques = sorted(parsed.dropna().unique())
        mapping = {u: i for i, u in enumerate(uniques)}
        return parsed.map(mapping).astype(float), mapping

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        uniques = sorted(pd.Series(numeric.dropna().unique()).tolist())
        mapping = {u: i for i, u in enumerate(uniques)}
        return numeric.map(mapping).astype(float), mapping

    col_name = str(series.name).lower() if series.name is not None else ""
    hints = ["date", "time", "day", "month", "year", "일자", "날짜", "시점", "연월"]
    if any(k in col_name for k in hints):
        parsed_dt = pd.to_datetime(s, errors="coerce")
        if parsed_dt.notna().mean() >= 0.8:
            uniques = sorted(parsed_dt.dropna().unique())
            mapping = {u: i for i, u in enumerate(uniques)}
            return parsed_dt.map(mapping).astype(float), mapping

    string = s.astype("string").str.strip()
    uniques = sorted(string.dropna().unique().tolist())
    mapping = {u: i for i, u in enumerate(uniques)}
    codes = string.map(mapping)
    return pd.to_numeric(codes, errors="coerce"), mapping


def _find_intervention_idx(mapping: dict[Any, int], intervention_value: Any) -> int:
    target = str(intervention_value).strip()
    for k, v in mapping.items():
        if str(k).strip() == target:
            return int(v)
        try:
            if float(k) == float(intervention_value):
                return int(v)
        except Exception:
            pass
        try:
            if pd.to_datetime(k, errors="coerce") == pd.to_datetime(intervention_value, errors="coerce"):
                return int(v)
        except Exception:
            pass
    raise AnalysisError("개입 시점을 시간변수 값에서 찾지 못했습니다.")


def _parallel_trend_test(event_table: pd.DataFrame) -> dict[str, float]:
    pre = event_table[(event_table["period"] < 0) & (event_table["period"] != -1)].copy()
    if pre.empty or pre["p_value"].isna().all():
        return {"pretrend_joint_p_value": np.nan, "num_significant_pretrends": np.nan}
    valid = pre.dropna(subset=["coef", "std_err"])
    if valid.empty:
        return {"pretrend_joint_p_value": np.nan, "num_significant_pretrends": np.nan}
    wald = np.sum((valid["coef"] / valid["std_err"]) ** 2)
    joint_p = 1 - chi2.cdf(wald, len(valid))
    return {"pretrend_joint_p_value": float(joint_p), "num_significant_pretrends": int((pre["p_value"] < 0.05).fillna(False).sum())}


def run_did_analysis(df: pd.DataFrame, outcome: str, treatment: str, time_var: str, type_map: dict[str, str], treated_value: Any, intervention_value: Any, controls: list[str] | None = None, unit_var: str | None = None, add_unit_fe: bool = False, add_time_fe: bool = False) -> AnalysisResult:
    controls = controls or []
    cols = [outcome, treatment, time_var] + controls + ([unit_var] if unit_var else [])
    work = df[cols].copy()
    for col in [outcome] + controls:
        work = _prepare_series(work, col, type_map[col])
    work = _prepare_series(work, treatment, type_map[treatment])
    time_idx, time_map = _ordered_time_codes(df[time_var])
    work["_time_idx"] = time_idx
    if work["_time_idx"].isna().all():
        raise AnalysisError("시간변수를 해석하지 못했습니다.")
    intervention_idx = _find_intervention_idx(time_map, intervention_value)
    work["treated"] = (df[treatment].astype("string").str.strip() == str(treated_value).strip()).astype(float)
    work["post"] = (work["_time_idx"] >= intervention_idx).astype(float)
    if unit_var:
        work[unit_var] = df[unit_var].astype("string")
    work = work.dropna(subset=[outcome, "treated", "post", "_time_idx"] + controls + ([unit_var] if unit_var else []))
    if len(work) < max(30, len(controls) + 10):
        raise AnalysisError("DiD를 수행하기에 유효 표본 수가 부족합니다.")
    if work["treated"].nunique() < 2 or work["post"].nunique() < 2:
        raise AnalysisError("처리집단/통제집단 또는 사전/사후 구분이 충분하지 않습니다.")

    formula_parts = ["treated", "post", "treated:post"]
    for col in controls:
        formula_parts.append(_formula_term(col, type_map[col]))
    if add_time_fe:
        formula_parts = [p for p in formula_parts if p != "post"]
        formula_parts.append("C(_time_idx)")
    if add_unit_fe and unit_var:
        if work[unit_var].nunique(dropna=True) > 300:
            raise AnalysisError("unit FE는 현재 300개 이하 단위에서만 권장합니다.")
        formula_parts.append(f"C(Q('{unit_var}'))")

    formula = f"Q('{outcome}') ~ " + " + ".join(formula_parts)
    model = smf.ols(formula=formula, data=work).fit(cov_type="HC1")
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    interaction_row = coef[coef["term"].str.contains("treated:post|post:treated", regex=True, na=False)]
    if interaction_row.empty:
        raise AnalysisError("DiD 상호작용항을 추출하지 못했습니다.")
    did_est = float(interaction_row.iloc[0]["Coef."])
    p_col = "P>|t|" if "P>|t|" in coef.columns else ("P>|z|" if "P>|z|" in coef.columns else None)
    did_p = float(interaction_row.iloc[0][p_col]) if p_col else np.nan

    means = work.groupby(["treated", "post"])[outcome].agg(["count", "mean", "std"]).reset_index()
    pre_post = means.pivot(index="treated", columns="post", values="mean").reset_index()
    pre_post.columns = ["treated", "pre_mean", "post_mean"]

    summary = {"analysis": "Difference-in-Differences", "n": int(model.nobs), "did_estimate": did_est, "p_value": did_p, "r_squared": float(model.rsquared)}
    detail_tables = {"group_time_means": means, "did_pre_post_summary": pre_post, "coefficient_table": coef, "time_mapping": pd.DataFrame({"original_time": list(time_map.keys()), "time_index": list(time_map.values())})}
    return AnalysisResult(summary=summary, stats_table=coef, detail_tables=detail_tables, narrative="DiD를 적합했습니다.", recommended_test="Difference-in-Differences")


def run_event_study_analysis(df: pd.DataFrame, outcome: str, treatment: str, time_var: str, type_map: dict[str, str], treated_value: Any, intervention_value: Any, controls: list[str] | None = None, unit_var: str | None = None, add_unit_fe: bool = False, window_pre: int = 4, window_post: int = 4) -> AnalysisResult:
    controls = controls or []
    cols = [outcome, treatment, time_var] + controls + ([unit_var] if unit_var else [])
    work = df[cols].copy()
    for col in [outcome] + controls:
        work = _prepare_series(work, col, type_map[col])
    work = _prepare_series(work, treatment, type_map[treatment])

    time_idx, time_map = _ordered_time_codes(df[time_var])
    work["_time_idx"] = time_idx
    intervention_idx = _find_intervention_idx(time_map, intervention_value)
    work["treated"] = (df[treatment].astype("string").str.strip() == str(treated_value).strip()).astype(float)
    work["rel_period"] = (work["_time_idx"] - intervention_idx).clip(lower=-window_pre, upper=window_post)

    if unit_var:
        work[unit_var] = df[unit_var].astype("string")
    work = work.dropna(subset=[outcome, "treated", "rel_period", "_time_idx"] + controls + ([unit_var] if unit_var else []))
    if len(work) < max(40, len(controls) + 12):
        raise AnalysisError("Event study를 수행하기에 표본 수가 부족합니다.")

    formula_parts = ["treated", "C(_time_idx)", "C(rel_period, Treatment(reference=-1)):treated"]
    for col in controls:
        formula_parts.append(_formula_term(col, type_map[col]))
    if add_unit_fe and unit_var:
        if work[unit_var].nunique(dropna=True) > 300:
            raise AnalysisError("unit FE는 현재 300개 이하 단위에서만 권장합니다.")
        formula_parts.append(f"C(Q('{unit_var}'))")

    formula = f"Q('{outcome}') ~ " + " + ".join(formula_parts)
    model = smf.ols(formula=formula, data=work).fit(cov_type="HC1")
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})

    rows = []
    for period in range(-window_pre, window_post + 1):
        if period == -1:
            rows.append({"period": period, "coef": 0.0, "std_err": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "is_reference": True})
            continue
        matching = coef[coef["term"].str.contains(fr"{period}", regex=False, na=False) & coef["term"].str.contains("treated", regex=False, na=False)]
        if matching.empty:
            rows.append({"period": period, "coef": np.nan, "std_err": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "is_reference": False})
            continue
        r = matching.iloc[0]
        se_col = "Std.Err." if "Std.Err." in matching.columns else "Std.Err"
        p_col = "P>|t|" if "P>|t|" in matching.columns else ("P>|z|" if "P>|z|" in matching.columns else None)
        se = float(r.get(se_col, np.nan))
        coef_val = float(r["Coef."])
        rows.append({"period": period, "coef": coef_val, "std_err": se, "p_value": float(r.get(p_col, np.nan)) if p_col else np.nan, "ci_low": coef_val - 1.96 * se if pd.notna(se) else np.nan, "ci_high": coef_val + 1.96 * se if pd.notna(se) else np.nan, "is_reference": False})

    event_table = pd.DataFrame(rows).sort_values("period")
    parallel_info = _parallel_trend_test(event_table)
    detail_tables = {"event_study_effects": event_table, "coefficient_table": coef, "time_mapping": pd.DataFrame({"original_time": list(time_map.keys()), "time_index": list(time_map.values())}), "parallel_trend_test": pd.DataFrame([parallel_info])}
    summary = {"analysis": "Event Study", "n": int(model.nobs), "r_squared": float(model.rsquared), **parallel_info}
    return AnalysisResult(summary=summary, stats_table=event_table, detail_tables=detail_tables, narrative="Event study를 적합했습니다.", recommended_test="Event Study")


def _formula_term(col: str, col_type: str) -> str:
    return f"C(Q('{col}'))" if col_type in CATEGORICAL else f"Q('{col}')"


def _run_correlation(work: pd.DataFrame, x: str, y: str, alpha: float, method_override: Optional[str]) -> AnalysisResult:
    clean = work[[x, y]].dropna()
    method = method_override or "Pearson"
    if method == "Pearson":
        stat, p = stats.pearsonr(clean[x], clean[y])
    else:
        stat, p = stats.spearmanr(clean[x], clean[y], nan_policy="omit")
    stats_table = pd.DataFrame([{"x": x, "y": y, "correlation": stat, "p_value": p, "n": len(clean), "method": method}])
    return AnalysisResult(summary={"test": f"{method} correlation", "statistic": float(stat), "p_value": float(p), "effect_size": float(stat), "n": len(clean)}, stats_table=stats_table, detail_tables={"descriptive_stats": clean.describe().reset_index()}, narrative="상관분석 결과입니다.", recommended_test=f"{method} correlation")


def _run_group_difference(work: pd.DataFrame, group: str, value: str, alpha: float, method_override: Optional[str]) -> AnalysisResult:
    clean = work[[group, value]].dropna()
    groups = list(clean[group].astype(str).unique())
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
    else:
        method = method_override or "ANOVA"
        if method == "Kruskal-Wallis":
            stat, p = stats.kruskal(*samples, nan_policy="omit")
            eff = np.nan
        else:
            stat, p = stats.f_oneway(*samples)
            eff = eta_squared_from_anova(stat, len(groups)-1, len(clean)-len(groups))
    stats_table = pd.DataFrame([{"group_variable": group, "value_variable": value, "test": method, "statistic": stat, "p_value": p, "effect_size": eff, "n": len(clean)}])
    return AnalysisResult(summary={"test": method, "statistic": float(stat), "p_value": float(p), "effect_size": float(eff) if pd.notna(eff) else np.nan, "n": len(clean)}, stats_table=stats_table, detail_tables={"group_descriptives": desc}, narrative="집단 차이 분석 결과입니다.", recommended_test=method)


def _run_categorical_association(work: pd.DataFrame, x: str, y: str, alpha: float, method_override: Optional[str]) -> AnalysisResult:
    clean = work[[x, y]].dropna().astype(str)
    table = pd.crosstab(clean[x], clean[y])
    if method_override == "Fisher exact" or (table.shape == (2,2) and (table < 5).any().any()):
        if table.shape != (2,2):
            raise AnalysisError("Fisher exact test는 2x2 표만 지원합니다.")
        stat, p = stats.fisher_exact(table.to_numpy())
        test = "Fisher exact"
    else:
        stat, p, _, _ = stats.chi2_contingency(table)
        test = "Chi-square"
    eff = cramers_v(table)
    return AnalysisResult(summary={"test": test, "statistic": float(stat), "p_value": float(p), "effect_size": float(eff), "n": int(table.to_numpy().sum())}, stats_table=pd.DataFrame([{"x": x, "y": y, "test": test, "statistic": stat, "p_value": p, "effect_size": eff, "n": int(table.to_numpy().sum())}]), detail_tables={"contingency_table": table.reset_index()}, narrative="범주형 연관성 분석 결과입니다.", recommended_test=test)


# ---------- PSM / research mode ----------

def _one_hot_for_model(df: pd.DataFrame, covariates: list[str], type_map: dict[str, str]) -> pd.DataFrame:
    X = df[covariates].copy()
    cat_cols = [c for c in covariates if type_map.get(c) in CATEGORICAL]
    for c in covariates:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            X[c] = X[c].astype("string").str.strip()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X


def _standardized_mean_diff(x_t, x_c) -> float:
    x_t = pd.Series(x_t).dropna().astype(float)
    x_c = pd.Series(x_c).dropna().astype(float)
    if len(x_t) == 0 or len(x_c) == 0:
        return np.nan
    m1, m0 = x_t.mean(), x_c.mean()
    v1, v0 = x_t.var(ddof=1), x_c.var(ddof=1)
    pooled = np.sqrt((v1 + v0) / 2) if pd.notna(v1) and pd.notna(v0) else np.nan
    if pooled == 0 or pd.isna(pooled):
        return np.nan
    return float((m1 - m0) / pooled)


def _expand_balance_df(df: pd.DataFrame, covariates: list[str], treatment_col: str, type_map: dict[str, str]) -> pd.DataFrame:
    temp = df[covariates + [treatment_col]].copy()
    cat_cols = [c for c in covariates if type_map.get(c) in CATEGORICAL]
    temp = pd.get_dummies(temp, columns=cat_cols, drop_first=False)
    rows = []
    for col in [c for c in temp.columns if c != treatment_col]:
        smd = _standardized_mean_diff(temp.loc[temp[treatment_col] == 1, col], temp.loc[temp[treatment_col] == 0, col])
        rows.append({"covariate": col, "smd": smd})
    return pd.DataFrame(rows)


def _match_nearest(ps_df: pd.DataFrame, caliper: float | None = None, with_replacement: bool = True) -> pd.DataFrame:
    treated = ps_df[ps_df["treated"] == 1].copy().sort_values("propensity_score")
    controls = ps_df[ps_df["treated"] == 0].copy().sort_values("propensity_score")
    available = controls.index.tolist()
    matches = []

    for ti, tr in treated.iterrows():
        if not available:
            break
        candidate_idx = available if not with_replacement else controls.index.tolist()
        cand = controls.loc[candidate_idx].copy()
        cand["distance"] = (cand["propensity_score"] - tr["propensity_score"]).abs()
        cand = cand.sort_values("distance")
        if caliper is not None:
            cand = cand[cand["distance"] <= caliper]
        if cand.empty:
            continue
        chosen = cand.iloc[0]
        matches.append({
            "treated_row_id": tr["row_id"],
            "control_row_id": chosen["row_id"],
            "treated_unit_id": tr.get("unit_id", tr["row_id"]),
            "control_unit_id": chosen.get("unit_id", chosen["row_id"]),
            "treated_ps": tr["propensity_score"],
            "control_ps": chosen["propensity_score"],
            "distance": chosen["distance"],
        })
        if not with_replacement:
            available.remove(chosen.name)
    return pd.DataFrame(matches)


def run_psm_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    type_map: dict[str, str],
    treated_value: Any,
    time_var: str | None = None,
    intervention_value: Any | None = None,
    unit_var: str | None = None,
    match_method: str = "nearest",
    caliper: float | None = None,
    with_replacement: bool = True,
):
    if not covariates:
        raise AnalysisError("PSM에는 공변량이 하나 이상 필요합니다.")

    # baseline sample
    base = df.copy()
    base["__row_id__"] = np.arange(len(base))
    base[treatment] = (base[treatment].astype("string").str.strip() == str(treated_value).strip()).astype(int)

    if unit_var and time_var and intervention_value is not None:
        time_idx, time_map = _ordered_time_codes(base[time_var])
        base["_time_idx"] = time_idx
        intervention_idx = _find_intervention_idx(time_map, intervention_value)
        pre = base[base["_time_idx"] < intervention_idx].copy()
        if pre.empty:
            raise AnalysisError("개입 이전 표본이 없어 baseline matching을 수행할 수 없습니다.")
        pre = pre.sort_values("_time_idx").groupby(unit_var, as_index=False).tail(1).copy()
        pre["unit_id"] = pre[unit_var].astype("string")
    elif unit_var:
        pre = base.groupby(unit_var, as_index=False).head(1).copy()
        pre["unit_id"] = pre[unit_var].astype("string")
    else:
        pre = base.copy()
        pre["unit_id"] = pre["__row_id__"].astype(str)

    needed = covariates + [treatment]
    pre = pre.dropna(subset=needed)
    if pre[treatment].nunique() < 2:
        raise AnalysisError("처리집단과 통제집단이 모두 존재해야 합니다.")
    if len(pre) < 30:
        raise AnalysisError("PSM을 수행하기에 baseline 표본 수가 너무 적습니다.")

    X = _one_hot_for_model(pre, covariates, type_map)
    X = X.replace([np.inf, -np.inf], np.nan)
    good = X.notna().all(axis=1)
    X = X.loc[good]
    y = pre.loc[good, treatment].astype(int)
    pre = pre.loc[good].copy()

    X_const = sm.add_constant(X, has_constant="add")
    try:
        model = sm.Logit(y, X_const).fit(disp=False)
    except Exception:
        model = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()
    pre["propensity_score"] = model.predict(X_const)
    pre["treated"] = y.values
    pre["row_id"] = pre["__row_id__"]

    matches = _match_nearest(pre[["row_id", "unit_id", "treated", "propensity_score"]], caliper=caliper, with_replacement=with_replacement)
    if matches.empty:
        raise AnalysisError("매칭 결과가 비었습니다. caliper를 완화하거나 공변량을 조정해보세요.")

    # matched units / rows
    matched_t_units = matches["treated_unit_id"].astype(str).unique().tolist()
    matched_c_units = matches["control_unit_id"].astype(str).unique().tolist()
    matched_units = set(matched_t_units + matched_c_units)

    if unit_var:
        matched_df = df[df[unit_var].astype("string").isin(list(matched_units))].copy()
    else:
        matched_rows = set(matches["treated_row_id"].tolist() + matches["control_row_id"].tolist())
        matched_df = base[base["__row_id__"].isin(list(matched_rows))].copy()

    # balance tables
    before_balance = _expand_balance_df(pre.assign(__treat__=pre["treated"]), covariates, "__treat__", type_map).rename(columns={"smd": "smd_before"})
    if unit_var:
        matched_pre = pre[pre["unit_id"].astype(str).isin(list(matched_units))].copy()
    else:
        matched_pre = pre[pre["row_id"].isin(list(matches["treated_row_id"]) + list(matches["control_row_id"]))].copy()
    after_balance = _expand_balance_df(matched_pre.assign(__treat__=matched_pre["treated"]), covariates, "__treat__", type_map).rename(columns={"smd": "smd_after"})
    balance = before_balance.merge(after_balance, on="covariate", how="outer")

    # ATT on baseline/current outcome
    att = np.nan
    att_p = np.nan
    if outcome in matched_pre.columns:
        treated_vals = pd.to_numeric(matched_pre.loc[matched_pre["treated"] == 1, outcome], errors="coerce").dropna()
        control_vals = pd.to_numeric(matched_pre.loc[matched_pre["treated"] == 0, outcome], errors="coerce").dropna()
        if len(treated_vals) > 1 and len(control_vals) > 1:
            att = float(treated_vals.mean() - control_vals.mean())
            _, att_p = stats.ttest_ind(treated_vals, control_vals, equal_var=False, nan_policy="omit")

    ps_table = pre[["row_id", "unit_id", "treated", "propensity_score"]].copy()
    ps_table["treated_label"] = ps_table["treated"].map({1: "Treated", 0: "Control"})
    matched_ps = ps_table[ps_table["unit_id"].astype(str).isin(list(matched_units))].copy()

    result = {
        "n_before": int(len(pre)),
        "n_after": int(len(matched_pre)),
        "att": att,
        "att_p_value": att_p,
        "propensity_table": ps_table,
        "matched_propensity_table": matched_ps,
        "match_pairs": matches,
        "balance_table": balance.sort_values("covariate"),
        "matched_df": matched_df,
        "matched_baseline_df": matched_pre,
        "model_coef": pd.DataFrame({"term": X_const.columns, "coef": model.params}),
    }
    return result


def run_survival_analysis(df: pd.DataFrame, duration_var: str, event_var: str, type_map: dict[str, str], event_value: Any, group_var: str | None = None, group_values: list[Any] | None = None, covariates: list[str] | None = None) -> AnalysisResult:
    covariates = covariates or []
    cols = [duration_var, event_var] + ([group_var] if group_var else []) + covariates
    work = df[cols].copy()
    work[duration_var] = pd.to_numeric(work[duration_var], errors="coerce")
    work["event_observed"] = (df[event_var].astype("string").str.strip() == str(event_value).strip()).astype(int)
    if group_var:
        work[group_var] = df[group_var].astype("string").str.strip()
    for col in covariates:
        work = _prepare_series(work, col, type_map[col])
    work = work.dropna(subset=[duration_var, "event_observed"] + ([group_var] if group_var else []) + covariates)
    work = work[work[duration_var] > 0]
    if len(work) < 20:
        raise AnalysisError("생존분석을 수행하기에 유효 표본 수가 부족합니다.")

    km_rows = []
    group_summary_rows = []
    if group_var:
        if group_values:
            work = work[work[group_var].isin([str(v).strip() for v in group_values])]
        groups = list(work[group_var].dropna().unique())
        if len(groups) < 1:
            raise AnalysisError("생존곡선을 그릴 집단이 없습니다.")
        for g in groups:
            gdf = work[work[group_var] == g]
            sf = SurvfuncRight(gdf[duration_var].to_numpy(), gdf["event_observed"].to_numpy())
            curve = pd.DataFrame({"time": sf.surv_times, "survival_probability": sf.surv_prob})
            curve["group"] = str(g)
            km_rows.append(curve)
            group_summary_rows.append({"group": str(g), "n": int(len(gdf)), "events": int(gdf["event_observed"].sum())})
        if len(groups) == 2:
            lr_stat, lr_p = survdiff(work[duration_var].to_numpy(), work["event_observed"].to_numpy(), work[group_var].to_numpy())
            lr_p = float(lr_p)
        else:
            lr_p = np.nan
    else:
        sf = SurvfuncRight(work[duration_var].to_numpy(), work["event_observed"].to_numpy())
        curve = pd.DataFrame({"time": sf.surv_times, "survival_probability": sf.surv_prob})
        curve["group"] = "All"
        km_rows.append(curve)
        group_summary_rows.append({"group": "All", "n": int(len(work)), "events": int(work["event_observed"].sum())})
        lr_p = np.nan

    km_table = pd.concat(km_rows, ignore_index=True)
    group_summary = pd.DataFrame(group_summary_rows)
    detail_tables = {"kaplan_meier_curve": km_table, "group_summary": group_summary}
    summary = {"analysis": "Survival Analysis", "n": int(len(work)), "events": int(work["event_observed"].sum()), "logrank_p": lr_p}
    narrative = "Kaplan-Meier 생존곡선을 추정했습니다."
    stats_table = group_summary
    return AnalysisResult(summary=summary, stats_table=stats_table, detail_tables=detail_tables, narrative=narrative, recommended_test="Kaplan-Meier")
