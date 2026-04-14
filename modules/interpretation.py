from __future__ import annotations

import math
import pandas as pd


def _p_text(p: float) -> str:
    if pd.isna(p):
        return "p 값을 계산하지 못했다"
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.4f}"


def _fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def build_bivariate_interpretation(result, x: str, y: str, x_type: str, y_type: str, alpha: float = 0.05) -> str:
    test = result.summary.get("test", "")
    p = result.summary.get("p_value", math.nan)
    stat = result.summary.get("statistic", math.nan)
    n = result.summary.get("n", math.nan)
    return (
        f"{x}와 {y}의 관계를 평가하기 위해 {test}을 실시하였다. "
        f"유효 표본 수는 {int(n)}개였고, 검정통계량은 {_fmt_num(stat)}이며 {_p_text(p)}로 나타났다. "
        f"이 결과는 {'통계적으로 유의한 차이/연관성' if pd.notna(p) and p < alpha else '통계적으로 유의하다고 보기 어려운 패턴'}을 시사한다. "
        f"다만 이 분석은 기초적 연관성 점검이므로 인과관계를 직접 의미하지는 않는다."
    )


def build_regression_interpretation(result, dependent: str, family: str, alpha: float = 0.05) -> str:
    coef = result.stats_table.copy()
    p_col = "P>|t|" if "P>|t|" in coef.columns else ("P>|z|" if "P>|z|" in coef.columns else None)
    coef_col = "Coef." if "Coef." in coef.columns else None
    sig_rows = []
    if p_col and coef_col:
        sig_rows = coef[(coef["term"] != "Intercept") & (coef[p_col] < alpha)].copy()
        sig_rows["abs_coef"] = sig_rows[coef_col].abs()
        sig_rows = sig_rows.sort_values("abs_coef", ascending=False).head(3).to_dict("records")

    if family == "linear":
        r2 = result.summary.get("r_squared", math.nan)
        details = []
        for row in sig_rows:
            details.append(f"{row['term']}은 β = {_fmt_num(row[coef_col])}, {_p_text(row[p_col])}")
        detail_text = "; ".join(details) if details else "선택한 유의수준에서 뚜렷하게 유의한 설명변수가 많지 않았다"
        return f"{dependent}을 종속변수로 한 선형회귀분석 결과, 모형 설명력은 R² = {_fmt_num(r2)}였다. {detail_text}. 회귀계수의 부호는 다른 조건이 일정할 때 결과변수와의 방향성을 의미한다."
    else:
        pseudo_r2 = result.summary.get("pseudo_r_squared", math.nan)
        details = []
        odds_col = "odds_ratio" if "odds_ratio" in coef.columns else None
        for row in sig_rows:
            if odds_col:
                details.append(f"{row['term']}의 오즈비는 {_fmt_num(row[odds_col])}였고 {_p_text(row[p_col])}")
        detail_text = "; ".join(details) if details else "선택한 유의수준에서 뚜렷하게 유의한 설명변수가 많지 않았다"
        return f"{dependent}을 종속변수로 한 로지스틱 회귀 결과, pseudo R²는 {_fmt_num(pseudo_r2)}였다. {detail_text}. 오즈비가 1보다 크면 사건 발생 가능성이 높아지는 방향으로 해석한다."


def build_did_interpretation(result, outcome: str, alpha: float = 0.05) -> str:
    est = result.summary.get("did_estimate", math.nan)
    p = result.summary.get("p_value", math.nan)
    joint_p = result.summary.get("pretrend_joint_p_value", math.nan)
    sig = "유의하였다" if pd.notna(p) and p < alpha else "유의하지 않았다"
    pretrend = ""
    if pd.notna(joint_p):
        pretrend = (
            f" 사전추세 joint test는 {_p_text(joint_p)}로 나타나 "
            + ("평행추세 가정에 비교적 우호적이었다." if joint_p >= alpha else "평행추세 가정이 충분히 지지되지 않을 수 있다.")
        )
    return (
        f"Difference-in-Differences 분석 결과, 처리집단×사후시점 상호작용항은 {_fmt_num(est)}로 나타났고 {_p_text(p)}에서 {sig}. "
        f"이는 개입 이후 처리집단의 {outcome}이 통제집단 대비 추가적으로 변화했음을 시사한다.{pretrend}"
    )


def build_event_study_interpretation(result, alpha: float = 0.05) -> str:
    joint_p = result.summary.get("pretrend_joint_p_value", math.nan)
    n_sig = result.summary.get("num_significant_pretrends", math.nan)
    return (
        "Event study는 기준시점(-1)을 준거로 상대시점별 처리효과를 추정한 결과이다. "
        + (f"사전추세 joint test는 {_p_text(joint_p)}였다. " if pd.notna(joint_p) else "")
        + (f"유의한 사전기간 계수는 {int(n_sig)}개였다. " if pd.notna(n_sig) else "")
        + "사전기간 계수들이 0에 가깝고 유의하지 않다면 평행추세 가정에 우호적이며, 사후기간 계수는 정책효과의 시차와 지속성을 보여준다."
    )


def build_survival_interpretation(result, alpha: float = 0.05) -> str:
    p = result.summary.get("logrank_p", math.nan)
    return (
        "Kaplan-Meier 곡선은 시간이 지남에 따라 사건이 아직 발생하지 않았을 확률을 보여준다. "
        + (f"log-rank 검정은 {_p_text(p)}였다. " if pd.notna(p) else "")
        + "곡선이 더 천천히 감소하는 집단일수록 사건 발생이 늦은 것으로 해석한다."
    )


def build_bulk_interpretation(result, alpha: float = 0.05) -> str:
    table = result.stats_table.copy()
    pcol = "adjusted_p_value" if "adjusted_p_value" in table.columns else "p_value"
    sig = table[pd.to_numeric(table[pcol], errors="coerce") < alpha]
    if sig.empty:
        return "다중비교 보정 이후 뚜렷하게 유의한 변수쌍이 많지 않았다."
    top_pairs = ", ".join([f"{r['x']}–{r['y']}" for _, r in sig.head(5).iterrows()])
    return f"전체 변수쌍을 탐색한 결과, 보정 후에도 유의한 조합이 확인되었다. 대표적인 조합은 {top_pairs} 등이었다."


def build_psm_interpretation(result: dict, outcome_label: str = "결과변수") -> str:
    att = result.get("att", math.nan)
    p = result.get("att_p_value", math.nan)
    before_n = result.get("n_before", math.nan)
    after_n = result.get("n_after", math.nan)
    return (
        f"성향점수매칭 결과, 매칭 전 표본 수는 {before_n}, 매칭 후 분석 표본 수는 {after_n}였다. "
        f"매칭 후 ATT는 {_fmt_num(att)}로 나타났고 {_p_text(p)}였다. "
        f"이는 공변량 균형을 맞춘 비교에서 처리집단의 {outcome_label}가 통제집단 대비 평균적으로 얼마나 다른지를 의미한다. "
        "다만 성향점수매칭은 관측된 공변량 기준의 선택편의만 줄여주므로, 비관측 교란이 존재하면 여전히 한계가 있다."
    )
