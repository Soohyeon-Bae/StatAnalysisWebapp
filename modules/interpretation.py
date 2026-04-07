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


def _corr_strength(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "매우 약한"
    if a < 0.3:
        return "약한"
    if a < 0.5:
        return "중간 정도의"
    if a < 0.7:
        return "비교적 강한"
    return "강한"


def _effect_strength(value: float, kind: str) -> str:
    if pd.isna(value):
        return "해석 보류"
    a = abs(value)
    if kind == "d":
        if a < 0.2:
            return "매우 작은 효과"
        if a < 0.5:
            return "작은 효과"
        if a < 0.8:
            return "중간 효과"
        return "큰 효과"
    if kind == "v":
        if a < 0.1:
            return "매우 약한 연관"
        if a < 0.3:
            return "약한 연관"
        if a < 0.5:
            return "중간 정도의 연관"
        return "강한 연관"
    if kind == "eta2":
        if a < 0.01:
            return "매우 작은 효과"
        if a < 0.06:
            return "작은 효과"
        if a < 0.14:
            return "중간 효과"
        return "큰 효과"
    return "효과 해석 보류"


def _fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def build_bivariate_interpretation(result, x: str, y: str, x_type: str, y_type: str, alpha: float = 0.05) -> str:
    test = result.summary.get("test", "")
    p = result.summary.get("p_value", math.nan)
    stat = result.summary.get("statistic", math.nan)
    n = result.summary.get("n", math.nan)

    if x_type == "continuous" and y_type == "continuous":
        direction = "양의" if stat >= 0 else "음의"
        strength = _corr_strength(stat)
        sig = "통계적으로 유의하였다" if pd.notna(p) and p < alpha else "통계적으로 유의하지 않았다"
        return (
            f"{x}와 {y}의 관계를 검토하기 위해 {test}을 실시하였다. 분석에 사용된 유효 표본 수는 {int(n)}개였다. "
            f"그 결과 상관계수는 {_fmt_num(stat)}로 나타났고, 이는 {strength} {direction} 상관관계에 해당한다 ({_p_text(p)}). "
            f"따라서 두 변수는 {sig}. 즉, {x}가 증가할수록 {y}가 {'증가' if stat >= 0 else '감소'}하는 경향이 관찰되었다. "
            f"다만 본 결과는 두 변수의 동반 변화를 보여주는 것이며, 제3의 변수나 교란요인을 통제하지 않았으므로 인과관계를 직접 의미하지는 않는다."
        )

    if (x_type in {"categorical", "binary"} and y_type == "continuous") or (x_type == "continuous" and y_type in {"categorical", "binary"}):
        detail = result.detail_tables.get("group_descriptives")
        eff = result.summary.get("effect_size", math.nan)
        eff_kind = "d" if (detail is not None and len(detail) == 2) else "eta2"
        eff_text = _effect_strength(eff, eff_kind)
        sig = "통계적으로 유의하였다" if pd.notna(p) and p < alpha else "통계적으로 유의하지 않았다"
        mean_text = ""
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            mean_bits = []
            gcol = detail.columns[0]
            for _, row in detail.iterrows():
                mean_bits.append(f"{row[gcol]} 집단 평균 {_fmt_num(row['mean'])}")
            mean_text = "; ".join(mean_bits)
        return (
            f"집단 간 평균 차이를 확인하기 위해 {test}을 실시하였다. 유효 표본 수는 {int(n)}개였다. "
            f"검정통계량은 {_fmt_num(stat)}이고 {_p_text(p)}로 나타나 집단 간 차이는 {sig}. "
            f"기술통계상 {mean_text}. 효과크기는 {_fmt_num(eff)}로 {eff_text}에 해당한다. "
            f"따라서 집단에 따라 평균 수준 차이가 존재한다고 해석할 수 있다. 다만 이는 집단 차이에 대한 통계적 판단이며, 해당 집단 변수가 결과변수를 직접 유발했다고 단정할 수는 없다."
        )

    if x_type in {"categorical", "binary"} and y_type in {"categorical", "binary"}:
        eff = result.summary.get("effect_size", math.nan)
        eff_text = _effect_strength(eff, "v")
        sig = "통계적으로 유의하였다" if pd.notna(p) and p < alpha else "통계적으로 유의하지 않았다"
        return (
            f"{x}와 {y}의 관련성을 확인하기 위해 {test}을 실시하였다. 유효 표본 수는 {int(n)}개였다. "
            f"검정통계량은 {_fmt_num(stat)}이고 {_p_text(p)}로 나타나 두 범주형 변수 간 분포 차이는 {sig}. "
            f"효과크기인 Cramér's V는 {_fmt_num(eff)}로 {eff_text} 수준이다. 즉, 한 변수의 범주에 따라 다른 변수의 분포가 달라지는 경향이 관찰되었다. 그러나 이는 연관성 결과일 뿐 인과관계를 의미하지 않는다."
        )

    return result.narrative


def build_regression_interpretation(result, dependent: str, family: str, alpha: float = 0.05) -> str:
    coef = result.stats_table.copy()
    p_col = None
    for cand in ["P>|t|", "P>|z|", "P>|T|", "P>|Z|"]:
        if cand in coef.columns:
            p_col = cand
            break
    coef_col = "Coef." if "Coef." in coef.columns else None
    sig_rows = []
    if p_col and coef_col:
        sig_rows = coef[(coef["term"] != "Intercept") & (coef[p_col] < alpha)].copy()
        sig_rows["abs_coef"] = sig_rows[coef_col].abs()
        sig_rows = sig_rows.sort_values("abs_coef", ascending=False).head(3).to_dict("records")

    if family == "linear":
        r2 = result.summary.get("r_squared", math.nan)
        bits = []
        for row in sig_rows:
            direction = "정(+)" if row[coef_col] > 0 else "부(-)"
            bits.append(f"{row['term']}은 {dependent}와 {direction}의 관련을 보였다 (β = {_fmt_num(row[coef_col])}, {_p_text(row[p_col])})")
        main = "; ".join(bits) if bits else "선택한 유의수준에서 뚜렷하게 유의한 설명변수가 많지 않았다"
        return f"{dependent}을 종속변수로 한 선형회귀분석을 실시하였다. 모형의 설명력은 R² = {_fmt_num(r2)}였다. {main}. 회귀계수의 부호는 다른 조건이 동일하다고 가정할 때 종속변수와의 방향성을 의미한다. 다만 관측자료 기반이므로 엄밀한 인과효과로 해석하는 데에는 한계가 있다."

    pseudo_r2 = result.summary.get("pseudo_r_squared", math.nan)
    encoding_info = result.summary.get("encoding_info")
    bits = []
    odds_col = "odds_ratio" if "odds_ratio" in coef.columns else None
    for row in sig_rows:
        if odds_col:
            direction = "증가" if row[coef_col] > 0 else "감소"
            bits.append(
                f"{row['term']}의 오즈비는 {_fmt_num(row[odds_col])}로, 해당 변수가 증가할수록 {dependent}=1의 가능성이 {direction}하는 방향이었다 ({_p_text(row[p_col])})"
            )
    main = "; ".join(bits) if bits else "선택한 유의수준에서 뚜렷하게 유의한 설명변수가 많지 않았다"
    mapping_text = f" 종속변수 인코딩은 {encoding_info}였다." if encoding_info else ""
    return (
        f"{dependent}을 종속변수로 한 로지스틱 회귀분석을 실시하였다. 모형 적합도 지표인 McFadden pseudo R²는 {_fmt_num(pseudo_r2)}였다."
        f"{mapping_text} {main}. 오즈비가 1보다 크면 사건 발생 가능성이 높아지는 방향으로 해석한다. 그러나 이 역시 관측자료 기반의 연관 추정이므로 추가적인 연구설계 없이는 인과효과로 단정할 수 없다."
    )


def build_bulk_interpretation(result, alpha: float = 0.05) -> str:
    table = result.stats_table.copy()
    pcol = "adjusted_p_value" if "adjusted_p_value" in table.columns else "p_value"
    sig = table[pd.to_numeric(table[pcol], errors="coerce") < alpha]
    sig = sig.sort_values(pcol).head(5)
    if sig.empty:
        return f"선택한 변수쌍 전체를 탐색한 결과, 보정된 유의수준 {alpha} 기준에서 뚜렷하게 유의한 조합이 많지 않았다. 이는 변수들 사이의 관련성이 약하거나 다중비교 보정으로 기준이 엄격해졌기 때문일 수 있다."
    pairs = ", ".join([f"{r['x']}–{r['y']}" for _, r in sig.iterrows()])
    return f"전체 변수 탐색 결과, 보정된 유의수준 {alpha} 기준에서 유의한 변수쌍이 확인되었다. 상위 조합은 {pairs} 등이었다. 다만 일괄 탐색 결과는 가설 생성 단계에 가깝고, 최종 해석 전에는 개별 변수쌍의 효과크기와 데이터 분포를 추가 검토하는 것이 바람직하다."


def build_did_interpretation(result, outcome: str, alpha: float = 0.05) -> str:
    est = result.summary.get("did_estimate", math.nan)
    p = result.summary.get("p_value", math.nan)
    r2 = result.summary.get("r_squared", math.nan)
    sig = "통계적으로 유의하였다" if pd.notna(p) and p < alpha else "통계적으로 유의하지 않았다"
    direction = "증가" if est >= 0 else "감소"
    return (
        f"Difference-in-Differences 분석 결과, 처리집단과 사후시점의 상호작용항 계수는 {_fmt_num(est)}로 나타났고 {_p_text(p)} 수준에서 {sig}. "
        f"이는 개입 이후 처리집단의 {outcome}이 통제집단 대비 추가적으로 {direction}했음을 시사한다. 모형 설명력은 R² = {_fmt_num(r2)}였다. "
        "다만 DiD 해석의 핵심 전제는 평행추세 가정이므로, 사전기간 추세가 유사한지 event study나 사전추세 검정으로 함께 확인하는 것이 적절하다."
    )


def build_event_study_interpretation(result, alpha: float = 0.05) -> str:
    table = result.stats_table.copy()
    pre = table[(table["period"] < 0) & (table["period"] != -1)]
    post = table[table["period"] >= 0]
    pre_sig = int((pre["p_value"] < alpha).fillna(False).sum()) if not pre.empty else 0
    post_sig = int((post["p_value"] < alpha).fillna(False).sum()) if not post.empty else 0
    joint_p = result.summary.get("pretrend_joint_p_value", math.nan)
    joint_text = f"사전기간 계수의 joint test p-value는 {_p_text(joint_p)}였다. " if pd.notna(joint_p) else ""
    return (
        f"Event study는 기준시점 -1을 준거로 상대시점별 효과를 추정한 결과이다. {joint_text}"
        f"사전기간에서 유의한 계수는 {pre_sig}개, 사후기간에서 유의한 계수는 {post_sig}개였다. "
        "사전기간 계수들이 0에 가깝고 유의하지 않다면 평행추세 가정을 지지하는 정황으로 볼 수 있고, 사후기간 계수의 부호와 크기는 개입 이후 효과의 지속성과 시차를 보여준다. "
        "다만 여전히 관측자료 분석이므로 동시 발생한 다른 변화나 구성 변화 가능성은 함께 검토해야 한다."
    )


def build_survival_interpretation(result, alpha: float = 0.05) -> str:
    p = result.summary.get("logrank_p", math.nan)
    concordance = result.summary.get("concordance", math.nan)
    txt = "Kaplan-Meier 곡선은 시간 경과에 따라 사건이 아직 발생하지 않았을 확률을 보여준다. 곡선이 더 위에 있고 완만할수록 사건이 늦게 발생하는 집단이다."
    if pd.notna(p):
        txt += f" 두 집단 log-rank 검정은 {_p_text(p)}로, 생존곡선 차이가 {'유의하다' if p < alpha else '유의하지 않다'}고 해석된다."
    if pd.notna(concordance):
        txt += f" Cox 모델의 concordance index는 {_fmt_num(concordance)}로, 값이 0.5를 넘을수록 사건 발생 순서를 어느 정도 구분해낸다는 뜻이다. hazard ratio가 1보다 크면 사건 발생 위험이 높은 방향이다."
    txt += " 다만 검열이 비정보적이라는 가정과 비례위험 가정이 적절한지 추가 점검이 필요하다."
    return txt
