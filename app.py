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
    run_psm_analysis,
)
from modules.file_loader import get_excel_sheet_names, load_uploaded_file
from modules.interpretation import (
    build_bivariate_interpretation,
    build_bulk_interpretation,
    build_did_interpretation,
    build_event_study_interpretation,
    build_regression_interpretation,
    build_survival_interpretation,
    build_psm_interpretation,
)
from modules.report_generator import build_excel_report, build_full_zip, dataframe_to_csv_bytes
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
    propensity_score_plot,
    love_plot,
)

st.set_page_config(page_title="Stat Explorer Pro", layout="wide")
st.title("Stat Explorer Pro")
st.caption("기존 탭 구조를 유지하면서, 인과분석 탭 안에 논문형 분석 모드(PSM + Balance + DiD + Event Study + ZIP export)를 추가한 버전")


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


def _to_display_strings(values):
    out = []
    for v in values:
        if pd.isna(v):
            continue
        out.append(str(v))
    return out


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
    profile_df.loc[profile_df["column"].astype(str).str.startswith("policy_after"), "user_type"] = "binary"
    profile_df.loc[profile_df["column"].astype(str).str.startswith("duration_from_"), "user_type"] = "continuous"
    st.session_state["profile_df"] = profile_df
    st.session_state["type_map"] = _infer_type_map(profile_df)


def _update_session_df(df: pd.DataFrame):
    st.session_state["df"] = df
    _sync_profile_table(df)


def _get_loaded_data():
    return st.session_state.get("df"), st.session_state.get("type_map", {})


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _safe_unique_sorted(series: pd.Series) -> list[str]:
    vals = series.dropna().astype("string").unique().tolist()
    return _to_display_strings(sorted(vals))


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


def _show_common_help(title: str, when_to_use: str, cautions: str):
    st.markdown(f"### 📘 {title}")
    with st.expander("이 분석을 언제 쓰나요? / 유의사항", expanded=True):
        st.markdown("**언제 쓰나요?**\n\n" + when_to_use + "\n\n**유의사항**\n\n" + cautions)


def _result_block(title: str, interpretation: str, stats_table: pd.DataFrame, detail_tables: dict[str, pd.DataFrame] | None = None, fig=None):
    st.subheader(f"📊 {title}")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(stats_table, use_container_width=True)
    if detail_tables:
        for name, table in detail_tables.items():
            with st.expander(name, expanded=name in {"group_descriptives", "group_time_means", "did_pre_post_summary", "group_summary", "parallel_trend_test", "balance_table", "match_pairs", "dependent_encoding"}):
                st.dataframe(table, use_container_width=True)
    st.subheader("📖 논문 수준 해석")
    st.text_area("해석", interpretation, height=260)


def _load_new_upload_if_needed(uploaded):
    if uploaded is None:
        return
    sheet_name = None
    if uploaded.name.lower().endswith(".xlsx"):
        sheets = get_excel_sheet_names(uploaded)
        sheet_name = st.selectbox("시트 선택", sheets, key="sheet_selector")
    upload_key = f"{uploaded.name}|{getattr(uploaded, 'size', 'na')}|{sheet_name}"
    prev_key = st.session_state.get("upload_key")
    if ("df" not in st.session_state) or (prev_key != upload_key):
        df_loaded, active_sheet = load_uploaded_file(uploaded, sheet_name)
        st.session_state["upload_key"] = upload_key
        st.session_state["active_sheet"] = active_sheet
        _update_session_df(df_loaded)
        st.success(f"파일 로드 완료: {uploaded.name} / sheet={active_sheet} / rows={len(df_loaded):,} / cols={len(df_loaded.columns):,}")
    else:
        st.info(f"현재 세션 데이터 사용 중 / sheet={st.session_state.get('active_sheet')}")


def _plot_for_bivariate(df, x, y, type_map):
    x_type, y_type = type_map[x], type_map[y]
    if x_type == "continuous" and y_type == "continuous":
        return scatter_plot(df, x, y)
    elif x_type in {"categorical", "binary"} and y_type == "continuous":
        return box_plot(df, x, y)
    elif x_type == "continuous" and y_type in {"categorical", "binary"}:
        return box_plot(df, y, x)
    elif x_type in {"categorical", "binary"} and y_type in {"categorical", "binary"}:
        table = pd.crosstab(df[x].astype("string"), df[y].astype("string"))
        return heatmap_plot(table)
    return None


def _show_step_nav(prefix: str, step_labels: dict[int, str], current_step: int):
    cols = st.columns(len(step_labels))
    for step_num, label in step_labels.items():
        with cols[step_num - 1]:
            if st.button(f"{step_num}. {label}", key=f"{prefix}_goto_{step_num}", use_container_width=True):
                st.session_state[f"{prefix}_step"] = step_num
                st.rerun()
    c1, c2, c3 = st.columns(3)
    with c1:
        if current_step > 1:
            if st.button("← 이전 단계", key=f"{prefix}_prev", use_container_width=True):
                st.session_state[f"{prefix}_step"] = current_step - 1
                st.rerun()
    with c3:
        if current_step < len(step_labels):
            if st.button("다음 단계 →", key=f"{prefix}_next", use_container_width=True):
                st.session_state[f"{prefix}_step"] = current_step + 1
                st.rerun()


(tab_data, tab_basic, tab_reg, tab_causal, tab_survival, tab_advanced) = st.tabs([
    f"{TAB_EMOJIS['data']} 데이터",
    f"{TAB_EMOJIS['basic']} 기초분석",
    f"{TAB_EMOJIS['reg']} 회귀분석",
    f"{TAB_EMOJIS['causal']} 인과분석",
    f"{TAB_EMOJIS['survival']} 생존분석",
    f"{TAB_EMOJIS['advanced']} 고급탐색",
])

with tab_data:
    _show_common_help(
        "데이터 업로드와 변수 타입 설정",
        "- 모든 분석의 출발점입니다.\n- CSV/XLSX를 업로드하고 변수 타입을 점검할 때 사용합니다.",
        "- 변수 타입이 잘못 잡히면 뒤 단계 분석이 지원되지 않거나 잘못된 검정이 선택될 수 있습니다.\n- 숫자형이 날짜로 과하게 해석되지 않도록 보수적으로 로딩합니다.",
    )
    uploaded = st.file_uploader("CSV 또는 XLSX 파일 업로드", type=["csv", "xlsx"])
    if uploaded is not None:
        _load_new_upload_if_needed(uploaded)
        df = st.session_state["df"]
        st.dataframe(df.head(20), use_container_width=True)
        generated_cols = [c for c in df.columns if str(c).startswith("policy_after") or str(c).startswith("duration_from_")]
        if generated_cols:
            st.success(f"현재 저장된 생성 변수: {', '.join(generated_cols)}")
        edited = st.data_editor(
            st.session_state["profile_df"],
            use_container_width=True,
            hide_index=True,
            column_config={"user_type": st.column_config.SelectboxColumn("user_type", options=TYPE_OPTIONS)},
            disabled=["column", "inferred_type", "non_null_count", "unique_count", "missing_count", "sample_values"],
            key="type_editor_final",
        )
        st.session_state["profile_df"] = edited
        st.session_state["type_map"] = _infer_type_map(edited)

df, type_map = _get_loaded_data()

with tab_basic:
    _ensure_loaded(df, type_map)
    _show_common_help(
        "기초분석",
        "- 두 변수의 관계를 빠르게 탐색할 때 사용합니다.\n- 상관분석, 평균 차이 검정, 범주형 연관성 검정에 적합합니다.",
        "- 인과관계를 의미하지 않습니다.\n- 표본 수가 너무 적거나 범주 수가 너무 많으면 해석이 불안정할 수 있습니다.",
    )
    analysis_columns = [c for c, t in type_map.items() if t not in {"exclude", "text", "id"}]
    fmt = _var_format_func(type_map)
    c1, c2 = st.columns(2)
    x = c1.selectbox("변수 X", analysis_columns, format_func=fmt)
    y = c2.selectbox("변수 Y", [c for c in analysis_columns if c != x], format_func=fmt)
    alpha = st.selectbox("유의수준", [0.1, 0.05, 0.01], index=1)
    if st.button("기초분석 실행", key="run_basic"):
        result = run_bivariate_analysis(df, x, y, type_map[x], type_map[y], alpha=alpha)
        fig = _plot_for_bivariate(df, x, y, type_map)
        interpretation = build_bivariate_interpretation(result, x, y, type_map[x], type_map[y], alpha)
        _display_summary_cards(result.summary)
        _result_block("기초분석 결과", interpretation, result.stats_table, result.detail_tables, fig)
        st.download_button("해석 TXT 다운로드", interpretation.encode("utf-8"), "bivariate_interpretation.txt", mime="text/plain")

with tab_reg:
    _ensure_loaded(df, type_map)
    _show_common_help(
        "회귀분석",
        "- 여러 설명변수를 동시에 고려하여 결과변수와의 조건부 관계를 볼 때 사용합니다.\n- 연속형 종속변수는 선형회귀, 이분형 종속변수는 로지스틱 회귀를 사용합니다.",
        "- 로지스틱 회귀의 종속변수는 정확히 두 범주만 가져야 합니다.\n- 관측자료의 회귀계수는 기본적으로 연관성 추정입니다.",
    )
    analysis_columns = [c for c, t in type_map.items() if t not in {"exclude", "text", "id", "datetime"}]
    fmt = _var_format_func(type_map)
    dependent = st.selectbox("종속변수", analysis_columns, format_func=fmt)
    independents = st.multiselect("설명변수", [c for c in analysis_columns if c != dependent], format_func=fmt)
    family = st.selectbox("회귀 유형", ["linear", "logistic"], index=0 if type_map[dependent] != "binary" else 1)
    if st.button("회귀분석 실행", key="run_reg"):
        result = run_regression(df, dependent, independents, type_map, family)
        interpretation = build_regression_interpretation(result, dependent, family)
        fig = coefficient_bar_plot(result.stats_table)
        _display_summary_cards(result.summary)
        _result_block("회귀분석 결과", interpretation, result.stats_table, result.detail_tables, fig)
        st.download_button("해석 TXT 다운로드", interpretation.encode("utf-8"), "regression_interpretation.txt", mime="text/plain")

with tab_causal:
    _ensure_loaded(df, type_map)
    mode = st.radio("분석 모드 선택", ["기본 DiD", "기본 Event Study", "논문형 분석 모드"], horizontal=True)
    fmt = _var_format_func(type_map)

    if mode == "기본 DiD":
        _show_common_help(
            "Difference-in-Differences (DiD)",
            "- 정책 시행 전/후와 처리집단/통제집단이 구분될 때 사용합니다.\n- 반복관측 또는 패널 구조에서 정책효과를 추정할 때 적합합니다.",
            "- 평행추세 가정이 핵심입니다.\n- 가능한 경우 unit FE와 time FE를 함께 고려해야 합니다.",
        )
        eligible_outcomes = [c for c, t in type_map.items() if t == "continuous"]
        eligible_treatments = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
        eligible_times = [c for c, t in type_map.items() if t in {"continuous", "datetime", "categorical", "binary"}]
        eligible_units = [c for c, t in type_map.items() if t in {"id", "categorical", "binary", "text"}]
        c1, c2, c3 = st.columns(3)
        outcome = c1.selectbox("Outcome", eligible_outcomes, format_func=fmt)
        treatment = c2.selectbox("Treatment group variable", eligible_treatments, format_func=fmt)
        time_var = c3.selectbox("Time variable", eligible_times, format_func=fmt)
        treat_values = _safe_unique_sorted(df[treatment])
        time_values = _safe_unique_sorted(df[time_var])
        c4, c5, c6 = st.columns(3)
        treated_value = c4.selectbox("처리집단 값", treat_values)
        intervention_value = c5.selectbox("개입 시점 값", time_values)
        unit_var = c6.selectbox("패널 단위 ID", ["(없음)"] + eligible_units, format_func=lambda x: x if x == "(없음)" else fmt(x))
        controls = st.multiselect("통제변수", [c for c in eligible_outcomes + eligible_treatments if c not in {outcome, treatment, time_var, unit_var}], format_func=fmt)
        add_unit_fe = st.checkbox("unit FE 추가", value=False)
        add_time_fe = st.checkbox("time FE 추가", value=True)
        if st.button("DiD 실행"):
            result = run_did_analysis(df, outcome, treatment, time_var, type_map, treated_value, intervention_value, controls, None if unit_var == "(없음)" else unit_var, add_unit_fe, add_time_fe)
            interpretation = build_did_interpretation(result, outcome)
            fig = did_trend_plot(result.detail_tables["group_time_means"], outcome) if "group_time_means" in result.detail_tables else None
            _display_summary_cards(result.summary)
            _result_block("DiD 결과", interpretation, result.stats_table, result.detail_tables, fig)
            st.download_button("해석 TXT 다운로드", interpretation.encode("utf-8"), "did_interpretation.txt", mime="text/plain")

    elif mode == "기본 Event Study":
        _show_common_help(
            "Event Study",
            "- 개입 전후 여러 시점에 걸친 효과의 동태를 보고 싶을 때 사용합니다.\n- 사전추세 검정까지 같이 보고 싶을 때 적합합니다.",
            "- 기준시점(-1)과 사후기간이 모두 존재해야 합니다.\n- 사전기간 계수와 joint test를 반드시 확인하세요.",
        )
        eligible_outcomes = [c for c, t in type_map.items() if t == "continuous"]
        eligible_treatments = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
        eligible_times = [c for c, t in type_map.items() if t in {"continuous", "datetime", "categorical", "binary"}]
        eligible_units = [c for c, t in type_map.items() if t in {"id", "categorical", "binary", "text"}]
        c1, c2, c3 = st.columns(3)
        outcome = c1.selectbox("Outcome", eligible_outcomes, key="es_outcome", format_func=fmt)
        treatment = c2.selectbox("Treatment group variable", eligible_treatments, key="es_treat", format_func=fmt)
        time_var = c3.selectbox("Time variable", eligible_times, key="es_time", format_func=fmt)
        treat_values = _safe_unique_sorted(df[treatment])
        time_values = _safe_unique_sorted(df[time_var])
        c4, c5, c6, c7 = st.columns(4)
        treated_value = c4.selectbox("처리집단 값", treat_values, key="es_treat_value")
        intervention_value = c5.selectbox("개입 시점 값", time_values, key="es_intervention")
        unit_var = c6.selectbox("패널 단위 ID", ["(없음)"] + eligible_units, key="es_unit", format_func=lambda x: x if x == "(없음)" else fmt(x))
        add_unit_fe = c7.checkbox("unit FE 추가", value=False, key="es_unit_fe")
        controls = st.multiselect("통제변수", [c for c in eligible_outcomes + eligible_treatments if c not in {outcome, treatment, time_var, unit_var}], key="es_controls", format_func=fmt)
        c8, c9 = st.columns(2)
        pre_w = c8.number_input("사전 기간 수", min_value=1, max_value=12, value=4)
        post_w = c9.number_input("사후 기간 수", min_value=1, max_value=12, value=4)
        if st.button("Event Study 실행"):
            result = run_event_study_analysis(df, outcome, treatment, time_var, type_map, treated_value, intervention_value, controls, None if unit_var == "(없음)" else unit_var, add_unit_fe, int(pre_w), int(post_w))
            interpretation = build_event_study_interpretation(result)
            fig = event_study_plot(result.stats_table)
            _display_summary_cards(result.summary)
            _result_block("Event Study 결과", interpretation, result.stats_table, result.detail_tables, fig)
            st.download_button("해석 TXT 다운로드", interpretation.encode("utf-8"), "event_study_interpretation.txt", mime="text/plain")

    else:
        _show_common_help(
            "논문형 분석 모드 (PSM + Balance + DiD + Event Study)",
            "- 논문처럼 ‘성향점수매칭 → 균형진단 → 처리효과 추정 → 그래프/표/해석 export’ 흐름을 재현할 때 사용합니다.\n- 업로드한 논문처럼 ATT, 공변량 균형(SMD), 매칭 전/후 비교, 처리효과 추정을 한 번에 관리하려는 목적에 적합합니다.",
            "- PSM은 관측된 공변량 기준의 선택편의만 줄여줍니다. 비관측 교란은 해결하지 못합니다.\n- matching 전후 표본 수와 SMD를 반드시 확인하세요.\n- DiD/Event Study는 매칭 표본과 원자료 둘 다 비교하는 것이 좋습니다.",
        )

        if "research_step" not in st.session_state:
            st.session_state["research_step"] = 1
        step_labels = {1: "변수설정", 2: "성향점수", 3: "매칭", 4: "균형진단", 5: "DiD", 6: "Event Study", 7: "Export"}
        current_step = st.session_state["research_step"]
        _show_step_nav("research", step_labels, current_step)

        eligible_outcomes = [c for c, t in type_map.items() if t == "continuous"]
        eligible_treatments = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
        eligible_times = [c for c, t in type_map.items() if t in {"continuous", "datetime", "categorical", "binary"}]
        eligible_units = [c for c, t in type_map.items() if t in {"id", "categorical", "binary", "text"}]
        eligible_covs = [c for c, t in type_map.items() if t in {"continuous", "categorical", "binary"}]

        if current_step == 1:
            st.markdown("#### STEP 1. 논문형 분석 변수 설정")
            c1, c2, c3 = st.columns(3)
            st.session_state["r_outcome"] = c1.selectbox("Outcome", eligible_outcomes, index=0 if "r_outcome" not in st.session_state else eligible_outcomes.index(st.session_state["r_outcome"]) if st.session_state["r_outcome"] in eligible_outcomes else 0, format_func=fmt)
            st.session_state["r_treatment"] = c2.selectbox("Treatment group variable", eligible_treatments, index=0 if "r_treatment" not in st.session_state else eligible_treatments.index(st.session_state["r_treatment"]) if st.session_state["r_treatment"] in eligible_treatments else 0, format_func=fmt)
            st.session_state["r_time"] = c3.selectbox("Time variable", eligible_times, index=0 if "r_time" not in st.session_state else eligible_times.index(st.session_state["r_time"]) if st.session_state["r_time"] in eligible_times else 0, format_func=fmt)

            treat_values = _safe_unique_sorted(df[st.session_state["r_treatment"]])
            time_values = _safe_unique_sorted(df[st.session_state["r_time"]])
            c4, c5, c6 = st.columns(3)
            st.session_state["r_treated_value"] = c4.selectbox("처리집단 값", treat_values, index=0 if "r_treated_value" not in st.session_state or st.session_state["r_treated_value"] not in treat_values else treat_values.index(st.session_state["r_treated_value"]))
            st.session_state["r_intervention_value"] = c5.selectbox("개입 시점 값", time_values, index=0 if "r_intervention_value" not in st.session_state or st.session_state["r_intervention_value"] not in time_values else time_values.index(st.session_state["r_intervention_value"]))
            unit_opts = ["(없음)"] + eligible_units
            st.session_state["r_unit"] = c6.selectbox("패널 단위 ID", unit_opts, index=0 if "r_unit" not in st.session_state or st.session_state["r_unit"] not in unit_opts else unit_opts.index(st.session_state["r_unit"]), format_func=lambda x: x if x == "(없음)" else fmt(x))
            st.session_state["r_controls"] = st.multiselect("통제변수", [c for c in eligible_covs if c not in {st.session_state['r_outcome'], st.session_state['r_treatment'], st.session_state['r_time'], st.session_state['r_unit']}], default=st.session_state.get("r_controls", []), format_func=fmt)
            st.session_state["r_match_method"] = st.selectbox("Matching method", ["nearest"], index=0)
            st.session_state["r_caliper"] = st.number_input("Caliper (0이면 사용 안 함)", min_value=0.0, value=float(st.session_state.get("r_caliper", 0.05)), step=0.01)
            st.session_state["r_with_replacement"] = st.checkbox("With replacement", value=st.session_state.get("r_with_replacement", True))
            st.session_state["r_add_unit_fe"] = st.checkbox("unit FE 추가", value=st.session_state.get("r_add_unit_fe", True))
            st.session_state["r_add_time_fe"] = st.checkbox("time FE 추가", value=st.session_state.get("r_add_time_fe", True))
            st.info("이 단계에서 선택한 설정은 이후 PSM, 균형진단, DiD, Event Study에 공통으로 사용됩니다.")

        elif current_step == 2:
            st.markdown("#### STEP 2. 성향점수 추정")
            params_ok = all(k in st.session_state for k in ["r_outcome", "r_treatment", "r_time", "r_treated_value", "r_intervention_value", "r_controls"])
            if not params_ok:
                st.warning("먼저 STEP 1에서 변수 설정을 완료하세요.")
            else:
                if st.button("성향점수 추정 실행", key="run_psm"):
                    try:
                        psm_res = run_psm_analysis(
                            df=df,
                            treatment=st.session_state["r_treatment"],
                            outcome=st.session_state["r_outcome"],
                            covariates=st.session_state["r_controls"],
                            type_map=type_map,
                            treated_value=st.session_state["r_treated_value"],
                            time_var=st.session_state["r_time"],
                            intervention_value=st.session_state["r_intervention_value"],
                            unit_var=None if st.session_state["r_unit"] == "(없음)" else st.session_state["r_unit"],
                            match_method=st.session_state["r_match_method"],
                            caliper=None if st.session_state["r_caliper"] == 0 else st.session_state["r_caliper"],
                            with_replacement=st.session_state["r_with_replacement"],
                        )
                        st.session_state["psm_result"] = psm_res
                    except Exception as e:
                        st.error(str(e))
                if "psm_result" in st.session_state:
                    psm_res = st.session_state["psm_result"]
                    fig = propensity_score_plot(psm_res["propensity_table"], matched=False)
                    interpretation = build_psm_interpretation(psm_res, st.session_state["r_outcome"])
                    _display_summary_cards({"n_before": psm_res["n_before"], "n_after": psm_res["n_after"], "ATT": psm_res["att"], "ATT p": psm_res["att_p_value"]})
                    _result_block("성향점수 추정 결과", interpretation, psm_res["model_coef"], {"propensity_table": psm_res["propensity_table"], "match_pairs": psm_res["match_pairs"]}, fig)

        elif current_step == 3:
            st.markdown("#### STEP 3. 매칭 결과 확인")
            if "psm_result" not in st.session_state:
                st.warning("STEP 2에서 성향점수 추정을 먼저 실행하세요.")
            else:
                psm_res = st.session_state["psm_result"]
                st.dataframe(psm_res["match_pairs"], use_container_width=True)
                st.info("매칭 쌍이 적절한지, 거리(distance)가 너무 큰 매칭이 많은지 확인하세요.")

        elif current_step == 4:
            st.markdown("#### STEP 4. 균형진단")
            if "psm_result" not in st.session_state:
                st.warning("STEP 2를 먼저 완료하세요.")
            else:
                psm_res = st.session_state["psm_result"]
                fig = love_plot(psm_res["balance_table"])
                interpretation = "Love plot과 SMD 표를 통해 매칭 전후 공변량 균형을 비교합니다. 일반적으로 |SMD|<0.1이면 균형이 개선된 것으로 봅니다."
                _result_block("균형진단 결과", interpretation, psm_res["balance_table"], None, fig)

        elif current_step == 5:
            st.markdown("#### STEP 5. 매칭 표본 DiD")
            if "psm_result" not in st.session_state:
                st.warning("STEP 2를 먼저 완료하세요.")
            else:
                matched_df = st.session_state["psm_result"]["matched_df"]
                try:
                    result = run_did_analysis(
                        df=matched_df,
                        outcome=st.session_state["r_outcome"],
                        treatment=st.session_state["r_treatment"],
                        time_var=st.session_state["r_time"],
                        type_map=type_map,
                        treated_value=st.session_state["r_treated_value"],
                        intervention_value=st.session_state["r_intervention_value"],
                        controls=st.session_state["r_controls"],
                        unit_var=None if st.session_state["r_unit"] == "(없음)" else st.session_state["r_unit"],
                        add_unit_fe=st.session_state["r_add_unit_fe"],
                        add_time_fe=st.session_state["r_add_time_fe"],
                    )
                    st.session_state["psm_did_result"] = result
                    fig = did_trend_plot(result.detail_tables["group_time_means"], st.session_state["r_outcome"]) if "group_time_means" in result.detail_tables else None
                    interpretation = build_did_interpretation(result, st.session_state["r_outcome"])
                    _display_summary_cards(result.summary)
                    _result_block("매칭 표본 DiD 결과", interpretation, result.stats_table, result.detail_tables, fig)
                except Exception as e:
                    st.error(str(e))

        elif current_step == 6:
            st.markdown("#### STEP 6. 매칭 표본 Event Study")
            if "psm_result" not in st.session_state:
                st.warning("STEP 2를 먼저 완료하세요.")
            else:
                c1, c2 = st.columns(2)
                pre_w = c1.number_input("사전 기간 수", min_value=1, max_value=12, value=int(st.session_state.get("r_pre_w", 4)), key="r_pre_w")
                post_w = c2.number_input("사후 기간 수", min_value=1, max_value=12, value=int(st.session_state.get("r_post_w", 4)), key="r_post_w")
                if st.button("매칭 표본 Event Study 실행", key="run_psm_es"):
                    matched_df = st.session_state["psm_result"]["matched_df"]
                    try:
                        result = run_event_study_analysis(
                            df=matched_df,
                            outcome=st.session_state["r_outcome"],
                            treatment=st.session_state["r_treatment"],
                            time_var=st.session_state["r_time"],
                            type_map=type_map,
                            treated_value=st.session_state["r_treated_value"],
                            intervention_value=st.session_state["r_intervention_value"],
                            controls=st.session_state["r_controls"],
                            unit_var=None if st.session_state["r_unit"] == "(없음)" else st.session_state["r_unit"],
                            add_unit_fe=st.session_state["r_add_unit_fe"],
                            window_pre=int(pre_w),
                            window_post=int(post_w),
                        )
                        st.session_state["psm_es_result"] = result
                    except Exception as e:
                        st.error(str(e))
                if "psm_es_result" in st.session_state:
                    result = st.session_state["psm_es_result"]
                    fig = event_study_plot(result.stats_table)
                    interpretation = build_event_study_interpretation(result)
                    _display_summary_cards(result.summary)
                    _result_block("매칭 표본 Event Study 결과", interpretation, result.stats_table, result.detail_tables, fig)

        elif current_step == 7:
            st.markdown("#### STEP 7. 전체 결과 ZIP 다운로드")
            if "psm_result" not in st.session_state:
                st.warning("PSM 결과가 없습니다. 최소한 STEP 2를 먼저 완료하세요.")
            else:
                psm_res = st.session_state["psm_result"]
                did_res = st.session_state.get("psm_did_result")
                es_res = st.session_state.get("psm_es_result")

                result_tables = {
                    "psm_model": psm_res["model_coef"],
                    "match_pairs": psm_res["match_pairs"],
                    "balance": psm_res["balance_table"],
                    "propensity": psm_res["propensity_table"],
                }
                interpretation_texts = {
                    "psm": build_psm_interpretation(psm_res, st.session_state["r_outcome"]),
                }
                figs = {
                    "propensity_distribution": propensity_score_plot(psm_res["propensity_table"], matched=False),
                    "love_plot": love_plot(psm_res["balance_table"]),
                }

                if did_res is not None:
                    result_tables["did_coef"] = did_res.stats_table
                    if "group_time_means" in did_res.detail_tables:
                        result_tables["did_group_time_means"] = did_res.detail_tables["group_time_means"]
                        figs["did_trend"] = did_trend_plot(did_res.detail_tables["group_time_means"], st.session_state["r_outcome"])
                    interpretation_texts["did"] = build_did_interpretation(did_res, st.session_state["r_outcome"])

                if es_res is not None:
                    result_tables["event_study"] = es_res.stats_table
                    if "parallel_trend_test" in es_res.detail_tables:
                        result_tables["parallel_trend_test"] = es_res.detail_tables["parallel_trend_test"]
                    figs["event_study"] = event_study_plot(es_res.stats_table)
                    interpretation_texts["event_study"] = build_event_study_interpretation(es_res)

                zip_bytes = build_full_zip(result_tables, interpretation_texts, figs)
                st.download_button("📦 전체 결과 ZIP 다운로드", data=zip_bytes, file_name="analysis_results.zip", mime="application/zip")
                st.info("ZIP에는 results.xlsx, interpretation.txt, plots 폴더가 포함됩니다.")

with tab_survival:
    _ensure_loaded(df, type_map)
    _show_common_help(
        "생존분석",
        "- 사건이 발생했는지뿐 아니라 언제 발생했는지도 함께 볼 때 사용합니다.\n- Kaplan-Meier, log-rank 검정을 기본으로 제공합니다.",
        "- duration은 0보다 큰 시간이 필요합니다.\n- 사건 변수는 0/1 또는 이분형 범주가 적합합니다.",
    )
    datetime_candidates = [c for c, t in type_map.items() if t == "datetime"]
    if datetime_candidates:
        s1, s2, s3, s4 = st.columns(4)
        duration_base = s1.selectbox("시작 시점 datetime", datetime_candidates, format_func=_var_format_func(type_map))
        dt_series = _coerce_datetime(df[duration_base]).dropna()
        default_ref = dt_series.max().date() if not dt_series.empty else date.today()
        ref_date = s2.date_input("기준일", value=default_ref)
        duration_unit = s3.selectbox("duration 단위", ["days", "months", "years"], index=1)
        new_duration_name = s4.text_input("새 duration 변수 이름", value=f"duration_from_{duration_base}_{duration_unit}")
        if st.button("duration 변수 생성"):
            new_df = _add_duration_from_datetime(st.session_state["df"].copy(), duration_base, ref_date, duration_unit, new_duration_name)
            _update_session_df(new_df)
            st.rerun()

    duration_candidates = [c for c, t in type_map.items() if t == "continuous"]
    event_candidates = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
    group_candidates = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
    cov_candidates = [c for c, t in type_map.items() if t in {"continuous", "binary", "categorical"}]
    fmt = _var_format_func(type_map)
    c1, c2, c3 = st.columns(3)
    duration_var = c1.selectbox("Duration variable", duration_candidates, format_func=fmt)
    event_var = c2.selectbox("Event variable", event_candidates, format_func=fmt)
    group_var = c3.selectbox("Group variable", ["(없음)"] + group_candidates, format_func=lambda x: x if x == "(없음)" else fmt(x))
    event_value = st.selectbox("사건 발생으로 볼 값", _safe_unique_sorted(df[event_var]))
    selected_groups = None
    if group_var != "(없음)":
        selected_groups = st.multiselect("비교할 그룹", _safe_unique_sorted(df[group_var]))
    covariates = st.multiselect("공변량", [c for c in cov_candidates if c not in {duration_var, event_var, group_var}], format_func=fmt)
    if st.button("생존분석 실행"):
        result = run_survival_analysis(df, duration_var, event_var, type_map, event_value, None if group_var == "(없음)" else group_var, selected_groups, covariates)
        fig = kaplan_meier_plot(result.detail_tables["kaplan_meier_curve"]) if "kaplan_meier_curve" in result.detail_tables else None
        interpretation = build_survival_interpretation(result)
        _display_summary_cards(result.summary)
        _result_block("생존분석 결과", interpretation, result.stats_table, result.detail_tables, fig)

with tab_advanced:
    _ensure_loaded(df, type_map)
    _show_common_help(
        "고급탐색",
        "- 여러 변수쌍을 한 번에 훑어보며 잠재적으로 유의한 조합을 찾을 때 사용합니다.",
        "- 다중비교 보정이 중요합니다.\n- 탐색 결과는 후속 개별 분석으로 다시 확인하는 것이 좋습니다.",
    )
    eligible = [c for c, t in type_map.items() if t in {"continuous", "categorical", "binary"}]
    selected = st.multiselect("탐색할 변수 선택(최소 3개)", eligible, default=eligible[: min(5, len(eligible))], format_func=_var_format_func(type_map))
    correction = st.selectbox("다중비교 보정", ["none", "bonferroni", "holm", "fdr_bh"], index=3)
    alpha = st.selectbox("유의수준", [0.1, 0.05, 0.01], index=1)
    if st.button("전체 변수 탐색 실행"):
        if len(selected) < 3:
            st.error("최소 3개 이상의 변수를 선택해주세요.")
        else:
            result = run_bulk_pairwise(df, selected, type_map, correction)
            interpretation = build_bulk_interpretation(result, alpha=alpha)
            _display_summary_cards(result.summary)
            _result_block("고급탐색 결과", interpretation, result.stats_table, None, None)
