from __future__ import annotations

import math
from datetime import date

import pandas as pd
import streamlit as st

from modules.analysis_engine import (
    AnalysisError,
    run_bivariate_analysis,
    run_bulk_pairwise,
    run_did_analysis,
    run_event_study_analysis,
    run_regression,
    run_survival_analysis,
)
from modules.file_loader import get_excel_sheet_names, load_uploaded_file
from modules.interpretation import (
    build_bivariate_interpretation,
    build_bulk_interpretation,
    build_did_interpretation,
    build_event_study_interpretation,
    build_regression_interpretation,
    build_survival_interpretation,
)
from modules.report_generator import build_excel_report, dataframe_to_csv_bytes
from modules.type_inference import TYPE_OPTIONS, build_profile_table
from modules.visualizer import (
    box_plot,
    coefficient_bar_plot,
    did_trend_plot,
    event_study_plot,
    heatmap_plot,
    kaplan_meier_plot,
    scatter_plot,
    stacked_bar_plot,
    violin_plot,
)

st.set_page_config(page_title="Stat Explorer Pro", layout="wide")
st.title("Stat Explorer Pro")
st.caption("귀여운 탭 구조로 데이터 업로드부터 기초검정, 회귀, 인과추론, 생존분석까지 한 번에 실행하는 연구용 분석 앱")

TYPE_BADGES = {
    "continuous": "연속형 숫자",
    "binary": "이분형",
    "categorical": "범주형",
    "datetime": "날짜형",
    "text": "텍스트",
    "id": "ID",
    "exclude": "제외",
}

TAB_EMOJIS = {
    "data": "🍰",
    "basic": "🧁",
    "reg": "☕",
    "causal": "🍪",
    "survival": "🍩",
    "advanced": "🍫",
}


def _display_summary_cards(summary: dict):
    items = list(summary.items())
    if not items:
        return
    cols = st.columns(min(4, len(items)))
    for i, (k, v) in enumerate(items[: len(cols)]):
        if isinstance(v, float):
            disp = "NA" if math.isnan(v) else f"{v:.4g}"
        else:
            disp = str(v)
        cols[i].metric(k, disp)


def _infer_type_map(profile_df: pd.DataFrame) -> dict[str, str]:
    return dict(zip(profile_df["column"], profile_df["user_type"]))


def _ensure_loaded(df, type_map):
    if df is None or not type_map:
        st.info(f"{TAB_EMOJIS['data']} 먼저 데이터 탭에서 파일을 업로드하고 변수 타입을 확인하세요.")
        st.stop()


def _plot_for_bivariate(df: pd.DataFrame, x: str, y: str, type_map: dict[str, str]):
    x_type, y_type = type_map[x], type_map[y]
    if x_type == "continuous" and y_type == "continuous":
        st.plotly_chart(scatter_plot(df, x, y), use_container_width=True)
    elif x_type in {"categorical", "binary"} and y_type == "continuous":
        st.plotly_chart(box_plot(df, x, y), use_container_width=True)
        st.plotly_chart(violin_plot(df, x, y), use_container_width=True)
    elif x_type == "continuous" and y_type in {"categorical", "binary"}:
        st.plotly_chart(box_plot(df, y, x), use_container_width=True)
        st.plotly_chart(violin_plot(df, y, x), use_container_width=True)
    elif x_type in {"categorical", "binary"} and y_type in {"categorical", "binary"}:
        table = pd.crosstab(df[x].astype("string"), df[y].astype("string"))
        st.plotly_chart(stacked_bar_plot(table), use_container_width=True)
        st.plotly_chart(heatmap_plot(table), use_container_width=True)


def _to_display_strings(values):
    out = []
    for v in values:
        if pd.isna(v):
            continue
        out.append(str(v))
    return out


def _get_loaded_data():
    return st.session_state.get("df"), st.session_state.get("type_map", {})


def _var_format_func(type_map: dict[str, str]):
    def _inner(col: str) -> str:
        return f"{col} ({type_map.get(col, 'unknown')})"

    return _inner


def _sync_profile_table(df: pd.DataFrame):
    profile_df = build_profile_table(df)
    old_profile = st.session_state.get("profile_df")
    if old_profile is not None and not old_profile.empty:
        prev = dict(zip(old_profile["column"], old_profile["user_type"]))
        profile_df["user_type"] = profile_df["column"].map(prev).fillna(profile_df["inferred_type"])
    else:
        profile_df["user_type"] = profile_df["inferred_type"]
    st.session_state["profile_df"] = profile_df
    st.session_state["type_map"] = _infer_type_map(profile_df)


def _update_session_df(df: pd.DataFrame):
    st.session_state["df"] = df
    _sync_profile_table(df)


def _result_block(title: str, interpretation: str, stats_table: pd.DataFrame, detail_tables: dict[str, pd.DataFrame] | None = None):
    st.subheader(f"📘 {title}")
    st.text_area("논문/보고서용 해석", interpretation, height=220)
    st.dataframe(stats_table, use_container_width=True)
    if detail_tables:
        for name, table in detail_tables.items():
            with st.expander(name, expanded=name in {"group_descriptives", "group_time_means", "did_pre_post_summary", "group_summary"}):
                st.dataframe(table, use_container_width=True)


def _basic_analysis_help(x_type: str, y_type: str, chosen_test: str) -> str:
    if x_type == "continuous" and y_type == "continuous":
        base = "두 연속형 변수의 선형 또는 단조 관계를 확인합니다. Pearson은 선형 상관, Spearman은 순위 기반 상관에 더 적합합니다. 연구 초기에 변수 간 함께 움직이는 경향이 있는지 볼 때 사용합니다."
    elif (x_type in {"categorical", "binary"} and y_type == "continuous") or (x_type == "continuous" and y_type in {"categorical", "binary"}):
        base = "집단에 따라 연속형 변수의 평균 또는 분포가 다른지 확인합니다. 두 집단이면 t-test/Mann-Whitney U, 세 집단 이상이면 ANOVA/Kruskal-Wallis가 주로 사용됩니다. 정책 수혜 여부나 성별에 따라 점수 차이가 있는지 볼 때 적합합니다."
    elif x_type in {"categorical", "binary"} and y_type in {"categorical", "binary"}:
        base = "두 범주형 변수의 분포가 서로 관련되어 있는지 확인합니다. 카이제곱검정은 가장 일반적이고, 표본이 적거나 기대도수가 작으면 Fisher exact test를 씁니다."
    else:
        base = "현재 조합은 기초분석에서 직접 지원하지 않는 유형입니다."
    if chosen_test != "auto":
        base += f" 현재는 {chosen_test}를 직접 선택한 상태입니다."
    return base


def _regression_help(dep_type: str, family: str) -> str:
    if family == "linear":
        return f"선형회귀는 종속변수({dep_type})가 연속형일 때 적합합니다. 여러 설명변수를 동시에 넣어 다른 조건을 통제한 상태의 조건부 관계를 봅니다. 결과의 핵심은 계수 부호, 유의성, R²입니다."
    return "로지스틱 회귀는 종속변수가 0/1 이분형일 때 적합합니다. 결과는 사건 발생 가능성의 방향과 크기를 보여주며, 오즈비(odds ratio) 해석이 중요합니다."


def _show_type_legend():
    st.markdown(
        "- **continuous**: 점수, 금액, 시간, 나이처럼 크기 비교가 가능한 숫자  \n"
        "- **binary**: 예/아니오, 0/1처럼 값이 두 개뿐인 변수  \n"
        "- **categorical**: 부서, 지역처럼 세 개 이상 범주를 갖는 변수  \n"
        "- **datetime**: 날짜/시점 변수  \n"
        "- **id / exclude**: 식별자 또는 분석에서 제외할 변수"
    )


def _safe_unique_sorted(series: pd.Series) -> list[str]:
    vals = series.dropna().astype("string").unique().tolist()
    return _to_display_strings(sorted(vals))


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _add_policy_after(df: pd.DataFrame, base_time_var: str, cutoff_value, new_col_name: str) -> pd.DataFrame:
    out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(out[base_time_var]) or st.session_state["type_map"].get(base_time_var) == "datetime":
        dt = _coerce_datetime(out[base_time_var])
        out[new_col_name] = (dt >= pd.to_datetime(cutoff_value)).astype("Int64")
    else:
        num = pd.to_numeric(out[base_time_var], errors="coerce")
        out[new_col_name] = (num >= float(cutoff_value)).astype("Int64")
    return out


def _add_duration_from_datetime(df: pd.DataFrame, base_time_var: str, reference_date: date, unit: str, new_col_name: str) -> pd.DataFrame:
    out = df.copy()
    dt = _coerce_datetime(out[base_time_var])
    ref = pd.Timestamp(reference_date)
    delta_days = (ref - dt).dt.days
    if unit == "days":
        duration = delta_days
    elif unit == "months":
        duration = delta_days / 30.4375
    else:
        duration = delta_days / 365.25
    out[new_col_name] = duration
    return out


def _show_analysis_result_guide():
    with st.expander("결과 해석 가이드", expanded=False):
        st.markdown(
            "- **p-value**: 통계적으로 우연일 가능성이 얼마나 작은지 보여줍니다. 보통 0.05 미만이면 유의하다고 봅니다.  \n"
            "- **effect size**: 유의성 외에 차이 또는 관계의 크기를 보여줍니다. 표본이 매우 크면 작은 차이도 유의할 수 있으므로 꼭 함께 보세요.  \n"
            "- **인과성 주의**: 기초분석과 일반 회귀는 기본적으로 연관성 분석입니다. 정책 효과처럼 인과 해석이 필요하면 인과분석 탭의 DiD/Event Study처럼 연구설계가 추가로 필요합니다."
        )


# Tabs
(tab_data, tab_basic, tab_reg, tab_causal, tab_survival, tab_advanced) = st.tabs([
    f"{TAB_EMOJIS['data']} 데이터",
    f"{TAB_EMOJIS['basic']} 기초분석",
    f"{TAB_EMOJIS['reg']} 회귀분석",
    f"{TAB_EMOJIS['causal']} 인과분석",
    f"{TAB_EMOJIS['survival']} 생존분석",
    f"{TAB_EMOJIS['advanced']} 고급탐색",
])

with tab_data:
    st.subheader(f"{TAB_EMOJIS['data']} 데이터 업로드와 변수 타입 설정")
    with st.expander("이 탭에서 무엇을 하나요?", expanded=True):
        st.markdown(
            "이 탭은 모든 분석의 출발점입니다. 파일을 업로드하고, 각 변수의 데이터 유형을 올바르게 지정해야 다음 탭에서 적절한 통계방법이 자동 추천됩니다."
        )
        _show_type_legend()

uploaded = st.file_uploader("CSV 또는 XLSX 파일 업로드", type=["csv", "xlsx"])
if uploaded is not None:
    sheet_name = None
    if uploaded.name.lower().endswith(".xlsx"):
        sheets = get_excel_sheet_names(uploaded)
        sheet_name = st.selectbox("시트 선택", sheets, key="sheet_selector")

    current_sig = (uploaded.name, getattr(uploaded, "size", None), sheet_name)
    prev_sig = st.session_state.get("uploaded_signature")

    if ("df" not in st.session_state) or (prev_sig != current_sig):
        try:
            df, active_sheet = load_uploaded_file(uploaded, sheet_name)
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.session_state["uploaded_signature"] = current_sig
        st.session_state["selected_sheet"] = active_sheet
        _update_session_df(df)
        st.success(f"파일 로드 완료: {uploaded.name} / sheet={active_sheet} / rows={len(df):,} / cols={len(df.columns):,}")
    else:
        df = st.session_state["df"]
        active_sheet = st.session_state.get("selected_sheet")
        st.info(f"현재 세션 데이터 사용 중 / sheet={active_sheet}")

    df = st.session_state["df"]
    st.dataframe(df.head(20), use_container_width=True)

    if "policy_after" in df.columns:
        st.success("생성된 변수: policy_after")
        
        st.success(f"파일 로드 완료: {uploaded.name} / sheet={active_sheet} / rows={len(df):,} / cols={len(df.columns):,}")
        st.dataframe(df.head(20), use_container_width=True)

        profile_df = st.session_state["profile_df"]
        st.subheader("변수 타입 확인 및 수정")
        edited = st.data_editor(
            profile_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "user_type": st.column_config.SelectboxColumn("user_type", options=TYPE_OPTIONS)
            },
            disabled=["column", "inferred_type", "non_null_count", "unique_count", "missing_count", "sample_values"],
            key="type_editor_v3",
        )
        st.session_state["profile_df"] = edited
        st.session_state["type_map"] = _infer_type_map(edited)
        st.info("다른 탭은 이 변수 타입 설정을 그대로 사용합니다. 변수 타입이 잘못 잡히면 분석이 지원되지 않거나 잘못된 검정이 선택될 수 있습니다.")
    else:
        st.info("분석을 시작하려면 CSV 또는 XLSX 파일을 업로드하세요.")


df, type_map = _get_loaded_data()

with tab_basic:
    _ensure_loaded(df, type_map)
    st.subheader(f"{TAB_EMOJIS['basic']} 기초분석: 두 변수 관계·차이·연관성")
    with st.expander("이 분석은 언제 쓰나요?", expanded=True):
        st.markdown(
            "두 변수만 먼저 빠르게 확인하고 싶을 때 가장 적합합니다.  \n"
            "- **continuous + continuous** → 상관분석  \n"
            "- **categorical/binary + continuous** → 평균 차이 검정  \n"
            "- **categorical/binary + categorical/binary** → 카이제곱/Fisher  \n"
            "목적은 **관계 탐색과 가설 생성**입니다. 아직 인과효과를 확정하는 단계는 아닙니다."
        )
    _show_analysis_result_guide()

    analysis_columns = [c for c, t in type_map.items() if t not in {"exclude", "text", "id"}]
    fmt = _var_format_func(type_map)
    left, right = st.columns(2)
    with left:
        x = st.selectbox("변수 X (continuous / binary / categorical)", analysis_columns, key="basic_x", format_func=fmt)
    with right:
        y = st.selectbox("변수 Y (continuous / binary / categorical)", [c for c in analysis_columns if c != x], key="basic_y", format_func=fmt)
    alpha = st.selectbox("유의수준 alpha", [0.1, 0.05, 0.01], index=1, key="basic_alpha")
    x_type, y_type = type_map[x], type_map[y]

    test_options = ["auto"]
    if x_type == "continuous" and y_type == "continuous":
        test_options += ["Pearson", "Spearman"]
    elif (x_type in {"categorical", "binary"} and y_type == "continuous") or (x_type == "continuous" and y_type in {"categorical", "binary"}):
        n_groups = df[x].nunique(dropna=True) if x_type in {"categorical", "binary"} else df[y].nunique(dropna=True)
        test_options += ["t-test", "Mann-Whitney U"] if n_groups == 2 else ["ANOVA", "Kruskal-Wallis"]
    elif x_type in {"categorical", "binary"} and y_type in {"categorical", "binary"}:
        test_options += ["Chi-square", "Fisher exact"]
    chosen_test = st.selectbox("검정 방법", test_options, key="basic_test")
    st.caption(_basic_analysis_help(x_type, y_type, chosen_test))

    if st.button("기초분석 실행", type="primary"):
        try:
            result = run_bivariate_analysis(df, x, y, x_type, y_type, alpha=alpha, method_override=None if chosen_test == "auto" else chosen_test)
        except AnalysisError as e:
            st.error(str(e))
            st.stop()
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        _result_block(
            "기초분석 결과",
            build_bivariate_interpretation(result, x, y, x_type, y_type, alpha=alpha),
            result.stats_table,
            result.detail_tables,
        )
        st.caption("주의: 기초분석은 상관성·차이·연관성을 보여주는 도구이며, 별도의 연구설계 없이 인과성을 확정하지 않습니다.")
        st.subheader("📈 그래프")
        _plot_for_bivariate(df, x, y, type_map)
        excel_bytes = build_excel_report({"main_result": result.stats_table, **result.detail_tables})
        c1, c2 = st.columns(2)
        c1.download_button("CSV 다운로드", data=dataframe_to_csv_bytes(result.stats_table), file_name="basic_analysis.csv", mime="text/csv")
        c2.download_button("엑셀 다운로드", data=excel_bytes, file_name="basic_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_reg:
    _ensure_loaded(df, type_map)
    st.subheader(f"{TAB_EMOJIS['reg']} 회귀분석: 여러 변수의 조건부 관계 보기")
    with st.expander("이 분석은 언제 쓰나요?", expanded=True):
        st.markdown(
            "회귀분석은 여러 설명변수를 동시에 고려할 때 사용합니다. 즉, 단순히 두 변수만 비교하는 것이 아니라 **다른 요인을 함께 통제한 상태에서** 종속변수와의 관계를 봅니다.  \n"
            "- **선형회귀**: 종속변수가 continuous일 때  \n"
            "- **로지스틱 회귀**: 종속변수가 binary일 때"
        )
    _show_analysis_result_guide()

    analysis_columns = [c for c, t in type_map.items() if t not in {"exclude", "text", "id", "datetime"}]
    fmt = _var_format_func(type_map)
    dependent = st.selectbox("종속변수 (continuous 또는 binary)", analysis_columns, key="reg_dep", format_func=fmt)
    independents = st.multiselect("설명변수들", [c for c in analysis_columns if c != dependent], key="reg_ind", format_func=fmt)
    dep_type = type_map[dependent]
    default_family = "logistic" if dep_type == "binary" else "linear"
    family = st.selectbox("회귀 유형", [default_family, "linear", "logistic"] if default_family == "logistic" else [default_family, "logistic"], index=0, key="reg_family")
    st.caption(_regression_help(dep_type, family))

    if st.button("회귀분석 실행", type="primary"):
        try:
            result = run_regression(df, dependent, independents, type_map, family)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        _result_block(
            "회귀분석 결과",
            build_regression_interpretation(result, dependent, family),
            result.stats_table,
            result.detail_tables,
        )
        st.caption("주의: 회귀계수는 다른 설명변수를 함께 고려한 조건부 관계입니다. 관측자료에서는 엄밀한 인과효과와 구분해서 작성하세요.")
        st.plotly_chart(coefficient_bar_plot(result.stats_table), use_container_width=True)
        excel_bytes = build_excel_report({"regression": result.stats_table, **result.detail_tables})
        st.download_button("회귀결과 엑셀 다운로드", data=excel_bytes, file_name="regression_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_causal:
    _ensure_loaded(df, type_map)
    st.subheader(f"{TAB_EMOJIS['causal']} 인과분석: DiD와 Event Study")
    with st.expander("이 분석은 언제 쓰나요?", expanded=True):
        st.markdown(
            "인과분석 탭은 **정책 시행 전/후**와 **처리집단/통제집단**이 구분될 때 사용합니다.  \n"
            "- **DiD**: 사전/사후 변화가 처리집단에서 얼마나 추가로 달라졌는지 추정  \n"
            "- **Event Study**: 개입 전후 여러 시점의 동태적 효과 확인  \n"
            "핵심은 시간변수와 처리집단 변수가 명확해야 한다는 점입니다."
        )
    _show_analysis_result_guide()

    st.markdown("#### 🛠️ Before/After 변수 자동 생성")
    dt_or_num_candidates = [c for c, t in type_map.items() if t in {"datetime", "continuous"}]
    if dt_or_num_candidates:
        c1, c2, c3 = st.columns(3)
        base_time = c1.selectbox("기준이 될 시간변수 (datetime 또는 continuous)", dt_or_num_candidates, key="post_base", format_func=_var_format_func(type_map))
        new_post_name = c2.text_input("새 변수 이름", value="policy_after", key="post_name")
        base_type = type_map[base_time]
        if base_type == "datetime":
            dt_series = _coerce_datetime(df[base_time]).dropna()
            min_d = dt_series.min().date() if not dt_series.empty else date(2020, 1, 1)
            max_d = dt_series.max().date() if not dt_series.empty else date.today()
            cutoff = c3.date_input("정책 시행일", value=min(max_d, date(2020, 1, 1)), min_value=min_d, max_value=max_d, key="post_cutoff_date")
        else:
            num_series = pd.to_numeric(df[base_time], errors="coerce").dropna()
            default_cutoff = float(num_series.median()) if not num_series.empty else 0.0
            cutoff = c3.number_input("사전/사후 기준값", value=default_cutoff, key="post_cutoff_num")
        st.caption("이 기능은 datetime/숫자형 시간변수를 0=사전, 1=사후의 binary 변수로 바꿔 DiD에 바로 사용할 수 있게 해줍니다.")
        if st.button("Before/After 변수 생성", key="make_post"):
            base_df = st.session_state["df"].copy()
            new_df = _add_policy_after(base_df, base_time, cutoff, new_post_name)
            _update_session_df(new_df)
            st.session_state["last_generated_var"] = new_post_name
            st.success(f"{new_post_name} 생성 완료")
            st.rerun()
            if "last_generated_var" in st.session_state:
                last_var = st.session_state["last_generated_var"]
                if last_var in st.session_state["df"].columns:
                    st.success(f"현재 데이터에 '{last_var}' 변수가 포함되어 있습니다.")

    st.markdown("### Difference-in-Differences")
    eligible_outcomes = [c for c, t in type_map.items() if t == "continuous"]
    eligible_treatments = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
    eligible_times = [c for c, t in type_map.items() if t in {"continuous", "datetime", "categorical", "binary"}]
    eligible_units = [c for c, t in type_map.items() if t in {"id", "categorical", "binary", "text"}]
    fmt = _var_format_func(type_map)
    did_cols = st.columns(3)
    outcome = did_cols[0].selectbox("Outcome (continuous)", eligible_outcomes, key="did_outcome", format_func=fmt)
    treatment = did_cols[1].selectbox("Treatment group variable (binary/categorical)", eligible_treatments, key="did_treat", format_func=fmt)
    time_var = did_cols[2].selectbox("Time variable (binary/categorical/datetime/continuous)", eligible_times, key="did_time", format_func=fmt)
    treat_values = _safe_unique_sorted(df[treatment])
    time_values = _safe_unique_sorted(df[time_var])
    did_cols2 = st.columns(4)
    treated_value = did_cols2[0].selectbox("처리집단 값", treat_values, key="did_treat_value")
    intervention_value = did_cols2[1].selectbox("개입 시점 값", time_values, key="did_intervention")
    unit_var = did_cols2[2].selectbox("패널 단위 ID (선택)", ["(없음)"] + eligible_units, key="did_unit", format_func=lambda x: x if x == "(없음)" else fmt(x))
    add_unit_fe = did_cols2[3].checkbox("unit FE 추가", value=False, key="did_unit_fe")
    add_time_fe = st.checkbox("time FE 추가", value=False, key="did_time_fe")
    did_controls = st.multiselect("통제변수", [c for c in eligible_outcomes + eligible_treatments if c not in {outcome, treatment, time_var, unit_var}], key="did_controls", format_func=fmt)
    st.caption("DiD는 처리집단/통제집단과 사전/사후 구분이 모두 필요합니다. time variable에는 방금 만든 policy_after 같은 변수도 사용할 수 있습니다.")

    if st.button("DiD 실행", type="primary"):
        try:
            result = run_did_analysis(
                df=df,
                outcome=outcome,
                treatment=treatment,
                time_var=time_var,
                type_map=type_map,
                treated_value=treated_value,
                intervention_value=intervention_value,
                controls=did_controls,
                unit_var=None if unit_var == "(없음)" else unit_var,
                add_unit_fe=add_unit_fe,
                add_time_fe=add_time_fe,
            )
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        _result_block(
            "DiD 결과",
            build_did_interpretation(result, outcome),
            result.stats_table,
            result.detail_tables,
        )
        if "group_time_means" in result.detail_tables:
            st.plotly_chart(did_trend_plot(result.detail_tables["group_time_means"], outcome), use_container_width=True)

    st.markdown("---")
    st.markdown("### Event Study")
    es_cols = st.columns(4)
    es_outcome = es_cols[0].selectbox("Outcome (continuous)", eligible_outcomes, key="es_outcome", format_func=fmt)
    es_treatment = es_cols[1].selectbox("Treatment group variable", eligible_treatments, key="es_treat", format_func=fmt)
    es_time = es_cols[2].selectbox("Time variable", eligible_times, key="es_time", format_func=fmt)
    es_treat_values = _safe_unique_sorted(df[es_treatment])
    es_time_values = _safe_unique_sorted(df[es_time])
    es_treated_value = es_cols[3].selectbox("처리집단 값", es_treat_values, key="es_treat_val")
    es_cols2 = st.columns(5)
    es_intervention = es_cols2[0].selectbox("개입 시점 값", es_time_values, key="es_intervention")
    es_unit = es_cols2[1].selectbox("패널 단위 ID (선택)", ["(없음)"] + eligible_units, key="es_unit", format_func=lambda x: x if x == "(없음)" else fmt(x))
    es_unit_fe = es_cols2[2].checkbox("unit FE 추가", value=False, key="es_unit_fe")
    pre_w = es_cols2[3].number_input("사전 기간 수", min_value=1, max_value=12, value=4, step=1, key="es_pre")
    post_w = es_cols2[4].number_input("사후 기간 수", min_value=1, max_value=12, value=4, step=1, key="es_post")
    es_controls = st.multiselect("통제변수", [c for c in eligible_outcomes + eligible_treatments if c not in {es_outcome, es_treatment, es_time, es_unit}], key="es_controls", format_func=fmt)
    st.caption("Event Study는 개입 전후 여러 기간의 효과를 보여줍니다. 사전기간 계수가 0에 가까우면 평행추세 가정을 점검하는 데 도움이 됩니다.")
    if st.button("Event Study 실행", type="primary"):
        try:
            result = run_event_study_analysis(
                df=df,
                outcome=es_outcome,
                treatment=es_treatment,
                time_var=es_time,
                type_map=type_map,
                treated_value=es_treated_value,
                intervention_value=es_intervention,
                controls=es_controls,
                unit_var=None if es_unit == "(없음)" else es_unit,
                add_unit_fe=es_unit_fe,
                window_pre=int(pre_w),
                window_post=int(post_w),
            )
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        _result_block(
            "Event Study 결과",
            build_event_study_interpretation(result),
            result.stats_table,
            result.detail_tables,
        )
        st.plotly_chart(event_study_plot(result.stats_table), use_container_width=True)

with tab_survival:
    _ensure_loaded(df, type_map)
    st.subheader(f"{TAB_EMOJIS['survival']} 생존분석: 사건이 언제 발생하는가")
    with st.expander("이 분석은 언제 쓰나요?", expanded=True):
        st.markdown(
            "생존분석은 단순히 사건이 발생했는지 여부뿐 아니라 **얼마나 시간이 지나서 발생했는지**를 다룹니다.  \n"
            "- **Kaplan-Meier**: 집단별 생존곡선 비교  \n"
            "- **Cox 모형**: 공변량을 통제한 위험도 비교  \n"
            "duration은 continuous, event는 binary로 두는 것이 가장 좋습니다."
        )
    _show_analysis_result_guide()

    st.markdown("#### 🛠️ duration 변수 자동 생성")
    datetime_candidates = [c for c, t in type_map.items() if t == "datetime"]
    if datetime_candidates:
        s1, s2, s3, s4 = st.columns(4)
        duration_base = s1.selectbox("시작 시점 datetime", datetime_candidates, key="dur_base", format_func=_var_format_func(type_map))
        dt_series = _coerce_datetime(df[duration_base]).dropna()
        default_ref = dt_series.max().date() if not dt_series.empty else date.today()
        ref_date = s2.date_input("기준일", value=default_ref, key="dur_ref")
        duration_unit = s3.selectbox("duration 단위", ["days", "months", "years"], index=1, key="dur_unit")
        new_duration_name = s4.text_input("새 duration 변수 이름", value=f"duration_from_{duration_base}_{duration_unit}", key="dur_name")
        st.caption("예: hire_date로부터 현재까지 경과 개월 수를 만들어 이탈·재진입까지 시간을 분석할 수 있습니다.")
        if st.button("duration 변수 생성", key="make_duration"):
            new_df = _add_duration_from_datetime(df, duration_base, ref_date, duration_unit, new_duration_name)
            _update_session_df(new_df)
            st.success(f"{new_duration_name} 생성 완료. 타입이 continuous인지 확인한 뒤 사용하세요.")
            st.rerun()

    duration_candidates = [c for c, t in type_map.items() if t == "continuous"]
    event_candidates = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
    group_candidates = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
    cov_candidates = [c for c, t in type_map.items() if t in {"continuous", "binary", "categorical"}]
    fmt = _var_format_func(type_map)
    s_cols = st.columns(3)
    duration_var = s_cols[0].selectbox("Duration variable (continuous)", duration_candidates, key="surv_duration", format_func=fmt)
    event_var = s_cols[1].selectbox("Event variable (binary/categorical)", event_candidates, key="surv_event", format_func=fmt)
    group_var = s_cols[2].selectbox("Group variable (선택)", ["(없음)"] + group_candidates, key="surv_group", format_func=lambda x: x if x == "(없음)" else fmt(x))
    event_values = _safe_unique_sorted(df[event_var])
    event_value = st.selectbox("사건 발생으로 볼 값", event_values, key="surv_event_value")
    selected_groups = None
    if group_var != "(없음)":
        group_values = _safe_unique_sorted(df[group_var])
        selected_groups = st.multiselect("비교할 그룹(최대 2개 권장)", group_values, default=group_values[: min(2, len(group_values))], key="surv_group_vals")
    covariates = st.multiselect("Cox 공변량(선택)", [c for c in cov_candidates if c not in {duration_var, event_var, group_var}], key="surv_covs", format_func=fmt)
    st.caption("duration은 0보다 큰 시간이 좋고, event는 1=사건 발생 / 0=검열 구조가 가장 해석하기 쉽습니다.")
    if st.button("생존분석 실행", type="primary"):
        try:
            result = run_survival_analysis(
                df=df,
                duration_var=duration_var,
                event_var=event_var,
                type_map=type_map,
                event_value=event_value,
                group_var=None if group_var == "(없음)" else group_var,
                group_values=selected_groups,
                covariates=covariates,
            )
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        _result_block(
            "생존분석 결과",
            build_survival_interpretation(result),
            result.stats_table,
            result.detail_tables,
        )
        if "kaplan_meier_curve" in result.detail_tables:
            st.plotly_chart(kaplan_meier_plot(result.detail_tables["kaplan_meier_curve"]), use_container_width=True)

with tab_advanced:
    _ensure_loaded(df, type_map)
    st.subheader(f"{TAB_EMOJIS['advanced']} 고급탐색: 여러 변수 한 번에 스캔")
    with st.expander("이 분석은 언제 쓰나요?", expanded=True):
        st.markdown(
            "가설이 아직 많고 어떤 변수쌍이 유의한지 전체적으로 훑어보고 싶을 때 사용합니다. 여러 쌍을 동시에 보므로 **다중비교 보정**이 중요합니다. 이 탭은 탐색 단계에 적합하고, 최종 해석은 개별 분석으로 다시 확인하는 것이 좋습니다."
        )
    _show_analysis_result_guide()

    eligible = [c for c, t in type_map.items() if t in {"continuous", "categorical", "binary"}]
    selected = st.multiselect("탐색할 변수 선택(최소 3개)", eligible, default=eligible[: min(5, len(eligible))], key="adv_selected", format_func=_var_format_func(type_map))
    correction = st.selectbox("다중비교 보정", ["none", "bonferroni", "holm", "fdr_bh"], index=3, key="adv_corr")
    alpha = st.selectbox("유의수준 alpha", [0.1, 0.05, 0.01], index=1, key="adv_alpha")
    st.caption("보정법을 쓰면 우연히 유의하게 보이는 변수쌍을 줄일 수 있습니다. 보통 fdr_bh가 탐색분석에서 많이 쓰입니다.")
    if st.button("전체 변수 탐색 실행", type="primary"):
        if len(selected) < 3:
            st.error("최소 3개 이상의 변수를 선택해주세요.")
            st.stop()
        result = run_bulk_pairwise(df, selected, type_map, correction)
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        _result_block(
            "고급탐색 결과",
            build_bulk_interpretation(result, alpha=alpha),
            result.stats_table,
            None,
        )
        st.download_button("탐색결과 CSV 다운로드", data=dataframe_to_csv_bytes(result.stats_table), file_name="bulk_pairwise.csv", mime="text/csv")
