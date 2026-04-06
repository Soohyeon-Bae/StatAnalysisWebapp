from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
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
        summary = {
            "analysis": "Linear Regression",
            "n": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "model_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        }
        narrative = (
            f"선형회귀 결과, 표본 {int(model.nobs)}건 기준 모형의 R²는 {model.rsquared:.3f}입니다. "
            "HC1 robust standard errors를 사용했습니다. 회귀계수표에서 각 변수의 방향과 유의성을 함께 확인하세요."
        )
    elif family == "logistic":
        model = smf.logit(formula=formula, data=work).fit(disp=False)
        coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        coef["odds_ratio"] = np.exp(coef["Coef."])
        summary = {
            "analysis": "Logistic Regression",
            "n": int(model.nobs),
            "pseudo_r_squared": float(model.prsquared),
            "model_pvalue": float(model.llr_pvalue),
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
        detail_tables={},
        narrative=narrative,
        recommended_test="Bulk Pairwise Screening",
    )


def run_did_analysis(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    time_var: str,
    type_map: dict[str, str],
    treated_value: Any,
    intervention_value: Any,
    controls: list[str] | None = None,
    unit_var: str | None = None,
    add_unit_fe: bool = False,
    add_time_fe: bool = False,
) -> AnalysisResult:
    controls = controls or []
    cols = [outcome, treatment, time_var] + controls + ([unit_var] if unit_var else [])
    work = df[cols].copy()
    for col in [outcome] + controls:
        work = _prepare_series(work, col, type_map[col])
    work = _prepare_series(work, treatment, type_map[treatment])
    # time processed separately for ordered coding
    time_idx, time_map = _ordered_time_codes(df[time_var])
    work["_time_idx"] = time_idx
    if work["_time_idx"].isna().all():
        raise AnalysisError("시간변수를 해석하지 못했습니다. 연도/월/날짜처럼 정렬 가능한 변수를 선택해주세요.")
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
        n_units = work[unit_var].nunique(dropna=True)
        if n_units > 200:
            raise AnalysisError("unit FE는 현재 200개 이하 단위에서만 권장합니다. 단위 수를 줄이거나 FE를 해제해주세요.")
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
    if {0.0, 1.0}.issubset(set(pre_post["treated"])):
        ctrl = pre_post[pre_post["treated"] == 0.0].iloc[0]
        trt = pre_post[pre_post["treated"] == 1.0].iloc[0]
        raw_did = (trt["post_mean"] - trt["pre_mean"]) - (ctrl["post_mean"] - ctrl["pre_mean"])
    else:
        raw_did = np.nan

    summary = {
        "analysis": "Difference-in-Differences",
        "n": int(model.nobs),
        "did_estimate": did_est,
        "p_value": did_p,
        "r_squared": float(model.rsquared),
    }
    narrative = (
        f"DiD 결과, treated×post 상호작용항 추정치는 {did_est:.4g}이며 p-value는 {did_p:.4g}입니다. "
        "이는 정책/처치 이후 처리집단이 통제집단 대비 얼마나 추가 변화했는지를 의미합니다. "
        "인과해석을 위해서는 평행추세 가정이 성립하는지 별도 검토가 필요합니다."
    )
    detail_tables = {
        "group_time_means": means,
        "did_pre_post_summary": pre_post,
        "coefficient_table": coef,
        "time_mapping": pd.DataFrame({"original_time": list(time_map.keys()), "time_index": list(time_map.values())}),
    }
    if pd.notna(raw_did):
        detail_tables["raw_did"] = pd.DataFrame([{"raw_difference_in_differences": raw_did}])

    return AnalysisResult(
        summary=summary,
        stats_table=coef,
        detail_tables=detail_tables,
        narrative=narrative,
        recommended_test="Difference-in-Differences",
    )


def run_event_study_analysis(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    time_var: str,
    type_map: dict[str, str],
    treated_value: Any,
    intervention_value: Any,
    controls: list[str] | None = None,
    unit_var: str | None = None,
    add_unit_fe: bool = False,
    window_pre: int = 4,
    window_post: int = 4,
) -> AnalysisResult:
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
    work["rel_period"] = work["_time_idx"] - intervention_idx
    work["rel_period"] = work["rel_period"].clip(lower=-window_pre, upper=window_post)
    if unit_var:
        work[unit_var] = df[unit_var].astype("string")
    work = work.dropna(subset=[outcome, "treated", "rel_period", "_time_idx"] + controls + ([unit_var] if unit_var else []))
    if len(work) < max(40, len(controls) + 12):
        raise AnalysisError("Event study를 수행하기에 표본 수가 부족합니다.")
    if not ((work["rel_period"] == -1).any() and (work["rel_period"] >= 0).any()):
        raise AnalysisError("기준시점 -1과 사후 기간이 모두 필요합니다. 시간 범위와 개입시점을 다시 확인해주세요.")

    formula_parts = ["treated", "C(_time_idx)", "C(rel_period, Treatment(reference=-1)):treated"]
    for col in controls:
        formula_parts.append(_formula_term(col, type_map[col]))
    if add_unit_fe and unit_var:
        n_units = work[unit_var].nunique(dropna=True)
        if n_units > 200:
            raise AnalysisError("unit FE는 현재 200개 이하 단위에서만 권장합니다. 단위 수를 줄이거나 FE를 해제해주세요.")
        formula_parts.append(f"C(Q('{unit_var}'))")
    formula = f"Q('{outcome}') ~ " + " + ".join(formula_parts)
    model = smf.ols(formula=formula, data=work).fit(cov_type="HC1")
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})

    rows = []
    for period in range(-window_pre, window_post + 1):
        if period == -1:
            rows.append({"period": period, "coef": 0.0, "std_err": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "is_reference": True})
            continue
        matching = coef[coef["term"].str.contains(fr"\[T\.{period}\].*treated|treated.*\[T\.{period}\]", regex=True, na=False)]
        if matching.empty:
            # patsy formatting fallback
            matching = coef[coef["term"].str.contains(f"{period}", regex=False, na=False) & coef["term"].str.contains("treated", regex=False, na=False)]
        if matching.empty:
            rows.append({"period": period, "coef": np.nan, "std_err": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "is_reference": False})
            continue
        r = matching.iloc[0]
        se_col = "Std.Err." if "Std.Err." in matching.columns else "Std.Err"
        p_col = "P>|t|" if "P>|t|" in matching.columns else ("P>|z|" if "P>|z|" in matching.columns else None)
        se = float(r.get(se_col, np.nan))
        coef_val = float(r["Coef."])
        rows.append({
            "period": period,
            "coef": coef_val,
            "std_err": se,
            "p_value": float(r.get(p_col, np.nan)) if p_col else np.nan,
            "ci_low": coef_val - 1.96 * se if pd.notna(se) else np.nan,
            "ci_high": coef_val + 1.96 * se if pd.notna(se) else np.nan,
            "is_reference": False,
        })
    event_table = pd.DataFrame(rows).sort_values("period")
    pre_table = event_table[(event_table["period"] < 0) & (event_table["period"] != -1)]
    joint_pre_max_abs = float(pre_table["coef"].abs().max()) if not pre_table.empty else np.nan
    summary = {
        "analysis": "Event Study",
        "n": int(model.nobs),
        "max_abs_pretrend": joint_pre_max_abs,
        "r_squared": float(model.rsquared),
    }
    narrative = (
        "Event study 결과, 기준시점(-1) 대비 각 상대시점의 처리효과를 추정했습니다. "
        "사전기간 계수들이 0에 가깝고 유의하지 않다면 평행추세 가정을 지지하는 정황으로 해석할 수 있습니다. "
        "사후기간 계수의 부호와 크기는 개입 이후 동태적 효과를 보여줍니다."
    )
    detail_tables = {
        "event_study_effects": event_table,
        "coefficient_table": coef,
        "time_mapping": pd.DataFrame({"original_time": list(time_map.keys()), "time_index": list(time_map.values())}),
    }
    return AnalysisResult(
        summary=summary,
        stats_table=event_table,
        detail_tables=detail_tables,
        narrative=narrative,
        recommended_test="Event Study",
    )


def run_survival_analysis(
    df: pd.DataFrame,
    duration_var: str,
    event_var: str,
    type_map: dict[str, str],
    event_value: Any,
    group_var: str | None = None,
    group_values: list[Any] | None = None,
    covariates: list[str] | None = None,
) -> AnalysisResult:
    covariates = covariates or []
    cols = [duration_var, event_var] + ([group_var] if group_var else []) + covariates
    work = df[cols].copy()
    work[duration_var] = pd.to_numeric(work[duration_var], errors="coerce")
    work["event_observed"] = (df[event_var].astype("string") == str(event_value)).astype(int)
    if group_var:
        work[group_var] = df[group_var].astype("string")
    for col in covariates:
        work = _prepare_series(work, col, type_map[col])
    subset_cols = [duration_var, "event_observed"] + ([group_var] if group_var else []) + covariates
    work = work.dropna(subset=subset_cols)
    work = work[work[duration_var] > 0]
    if len(work) < 20:
        raise AnalysisError("생존분석을 수행하기에 유효 표본 수가 부족합니다.")

    km_rows = []
    group_summary_rows = []
    if group_var:
        if group_values:
            work = work[work[group_var].isin([str(v) for v in group_values])]
        groups = list(work[group_var].dropna().unique())
        if len(groups) < 1:
            raise AnalysisError("생존곡선을 그릴 집단이 없습니다.")
        for g in groups:
            gdf = work[work[group_var] == g]
            sf = SurvfuncRight(gdf[duration_var].to_numpy(), gdf["event_observed"].to_numpy())
            curve = pd.DataFrame({"time": sf.surv_times, "survival_probability": sf.surv_prob})
            curve["group"] = str(g)
            km_rows.append(curve)
            median_surv = np.nan
            try:
                median_surv = float(sf.quantile(0.5))
            except Exception:
                pass
            group_summary_rows.append({
                "group": str(g),
                "n": int(len(gdf)),
                "events": int(gdf["event_observed"].sum()),
                "median_survival_time": median_surv,
            })
        if len(groups) == 2:
            lr_stat, lr_p = survdiff(work[duration_var].to_numpy(), work["event_observed"].to_numpy(), work[group_var].to_numpy())
            lr_stat = float(lr_stat)
            lr_p = float(lr_p)
        else:
            lr_stat = np.nan
            lr_p = np.nan
    else:
        sf = SurvfuncRight(work[duration_var].to_numpy(), work["event_observed"].to_numpy())
        curve = pd.DataFrame({"time": sf.surv_times, "survival_probability": sf.surv_prob})
        curve["group"] = "All"
        km_rows.append(curve)
        median_surv = np.nan
        try:
            median_surv = float(sf.quantile(0.5))
        except Exception:
            pass
        group_summary_rows.append({
            "group": "All",
            "n": int(len(work)),
            "events": int(work["event_observed"].sum()),
            "median_survival_time": median_surv,
        })
        lr_stat = np.nan
        lr_p = np.nan

    km_table = pd.concat(km_rows, ignore_index=True)
    group_summary = pd.DataFrame(group_summary_rows)

    detail_tables: Dict[str, pd.DataFrame] = {
        "kaplan_meier_curve": km_table,
        "group_summary": group_summary,
    }
    summary = {
        "analysis": "Survival Analysis",
        "n": int(len(work)),
        "events": int(work["event_observed"].sum()),
        "logrank_p": lr_p,
    }
    narrative = "Kaplan-Meier 생존곡선을 추정했습니다. 곡선이 더 천천히 감소하는 집단일수록 사건이 늦게 발생하는 경향을 의미합니다."
    if pd.notna(lr_p):
        narrative += f" 두 집단 log-rank 검정 p-value는 {lr_p:.4g}입니다."

    if covariates:
        cox_df = work[[duration_var, "event_observed"] + covariates].copy()
        cox_df = pd.get_dummies(cox_df, columns=[c for c in covariates if type_map[c] in CATEGORICAL], drop_first=True)
        exog_cols = [c for c in cox_df.columns if c not in {duration_var, "event_observed"}]
        model = PHReg(cox_df[duration_var], cox_df[exog_cols], status=cox_df["event_observed"])
        res = model.fit(disp=False)
        cox_table = pd.DataFrame({
            "term": exog_cols,
            "Coef.": res.params,
            "Std.Err.": res.bse,
            "z": res.tvalues,
            "P>|z|": res.pvalues,
            "hazard_ratio": np.exp(res.params),
        })
        detail_tables["cox_regression"] = cox_table
        # no standard concordance in PHReg result; use pseudo summary field left NA
        summary["concordance"] = np.nan
        narrative += " 선택한 공변량으로 Cox proportional hazards 모델도 적합했습니다. hazard ratio가 1보다 크면 사건 발생 위험이 높은 방향입니다."
        stats_table = cox_table
    else:
        stats_table = group_summary

    return AnalysisResult(
        summary=summary,
        stats_table=stats_table,
        detail_tables=detail_tables,
        narrative=narrative,
        recommended_test="Kaplan-Meier / CoxPH",
    )

def _prepare_series(df: pd.DataFrame, col: str, col_type: str) -> pd.DataFrame:
    if col_type == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce")
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
    if len(groups) > 20:
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
        narrative = f"두 집단 비교 결과 {method}의 p-value는 {p:.4g}입니다. 효과크기(Cohen's d)는 {eff:.3f}입니다."
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

    return AnalysisResult(summary=summary, stats_table=stats_table, detail_tables={"group_descriptives": desc}, narrative=narrative, recommended_test=summary["test"])


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
        stat, p, _, _ = stats.chi2_contingency(table)
        test = "Chi-square"
    eff = cramers_v(table)
    row_pct = table.div(table.sum(axis=1), axis=0).reset_index()
    stats_table = pd.DataFrame([
        {"x": x, "y": y, "test": test, "statistic": stat, "p_value": p, "effect_size": eff, "n": int(table.to_numpy().sum())}
    ])
    narrative = f"{test} 결과 p-value는 {p:.4g}이며, Cramér's V는 {eff:.3f}입니다. " + ("범주 간 분포 차이가 통계적으로 유의합니다." if p < alpha else "뚜렷한 분포 차이가 있다고 보기는 어렵습니다.")
    return AnalysisResult(
        summary={"test": test, "statistic": float(stat), "p_value": float(p), "effect_size": float(eff), "n": int(table.to_numpy().sum())},
        stats_table=stats_table,
        detail_tables={"contingency_table": table.reset_index(), "row_percentages": row_pct},
        narrative=narrative,
        recommended_test=test,
    )


def _formula_term(col: str, col_type: str) -> str:
    return f"C(Q('{col}'))" if col_type in CATEGORICAL else f"Q('{col}')"


def _ordered_time_codes(series: pd.Series) -> tuple[pd.Series, dict[Any, int]]:
    s = series.copy()

    # 1) 이미 datetime dtype이면 datetime으로 처리
    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = pd.to_datetime(s, errors="coerce")
        uniques = sorted(parsed.dropna().unique())
        mapping = {u: i for i, u in enumerate(uniques)}
        codes = parsed.map(mapping)
        return codes.astype(float), mapping

    # 2) 숫자형/0-1형은 datetime보다 먼저 numeric으로 처리
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        uniques = sorted(pd.Series(numeric.dropna().unique()).tolist())
        mapping = {u: i for i, u in enumerate(uniques)}
        codes = numeric.map(mapping)
        return codes.astype(float), mapping

    # 3) 숫자가 아니고, 컬럼명이 date/time 계열일 때만 datetime 시도
    col_name = str(series.name).lower() if series.name is not None else ""
    hint_keywords = ["date", "time", "day", "month", "year", "일자", "날짜", "시점", "연월"]
    if any(k in col_name for k in hint_keywords):
        parsed_dt = pd.to_datetime(s, errors="coerce")
        if parsed_dt.notna().mean() >= 0.8:
            uniques = sorted(parsed_dt.dropna().unique())
            mapping = {u: i for i, u in enumerate(uniques)}
            codes = parsed_dt.map(mapping)
            return codes.astype(float), mapping

    # 4) 나머지는 문자열 범주 순서로 처리
    string = s.astype("string").str.strip()
    uniques = sorted(string.dropna().unique().tolist())
    mapping = {u: i for i, u in enumerate(uniques)}
    codes = string.map(mapping)
    return pd.to_numeric(codes, errors="coerce"), mapping


def _find_intervention_idx(mapping: dict[Any, int], intervention_value: Any) -> int:
    # 문자열 기준 비교 (가장 안정적)
    target = str(intervention_value).strip()
    for k, v in mapping.items():
        # 문자열 비교
        if str(k).strip() == target:
            return int(v)
        # 숫자 비교 (float/int 대응)
        try:
            if float(k) == float(intervention_value):
                return int(v)
        except:
            pass
        # datetime 비교
        try:
            if pd.to_datetime(k, errors="coerce") == pd.to_datetime(intervention_value, errors="coerce"):
                return int(v)
        except:
            pass
    raise AnalysisError("개입 시점을 시간변수 값에서 찾지 못했습니다.")
