from __future__ import annotations

import math

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
st.caption("데이터 업로드부터 기초검정, 회귀, DiD, Event Study, 생존분석까지 탭별로 실행할 수 있는 연구용 분석 앱")


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
        st.info("먼저 데이터 탭에서 파일을 업로드하고 변수 타입을 확인하세요.")
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


tab_data, tab_basic, tab_reg, tab_causal, tab_survival, tab_advanced = st.tabs([
    "데이터",
    "기초분석",
    "회귀분석",
    "인과분석",
    "생존분석",
    "고급탐색",
])

with tab_data:
    uploaded = st.file_uploader("CSV 또는 XLSX 파일 업로드", type=["csv", "xlsx"])
    if uploaded is not None:
        sheet_name = None
        if uploaded.name.lower().endswith(".xlsx"):
            sheets = get_excel_sheet_names(uploaded)
            sheet_name = st.selectbox("시트 선택", sheets, key="sheet_selector")
        try:
            df, active_sheet = load_uploaded_file(uploaded, sheet_name)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()

        st.session_state["df"] = df
        st.success(f"파일 로드 완료: {uploaded.name} / sheet={active_sheet} / rows={len(df):,} / cols={len(df.columns):,}")
        st.dataframe(df.head(20), use_container_width=True)

        profile_df = build_profile_table(df)
        if "profile_df" not in st.session_state or list(st.session_state.get("profile_df", pd.DataFrame()).get("column", [])) != list(profile_df["column"]):
            profile_df["user_type"] = profile_df["inferred_type"]
            st.session_state["profile_df"] = profile_df
        else:
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
            key="type_editor_v2",
        )
        st.session_state["profile_df"] = edited
        st.session_state["type_map"] = _infer_type_map(edited)
        st.info("다른 탭은 이 변수 타입 설정을 그대로 사용합니다. datetime은 날짜, continuous는 숫자, binary는 두 범주, categorical은 다범주 변수입니다.")
    else:
        st.info("분석을 시작하려면 CSV 또는 XLSX 파일을 업로드하세요.")


df, type_map = _get_loaded_data()

with tab_basic:
    _ensure_loaded(df, type_map)
    analysis_columns = [c for c, t in type_map.items() if t not in {"exclude", "text", "id"}]
    left, right = st.columns(2)
    with left:
        x = st.selectbox("변수 X", analysis_columns, key="basic_x")
    with right:
        y = st.selectbox("변수 Y", [c for c in analysis_columns if c != x], key="basic_y")
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
    if st.button("기초분석 실행", type="primary"):
        try:
            result = run_bivariate_analysis(df, x, y, x_type, y_type, alpha=alpha, method_override=None if chosen_test == "auto" else chosen_test)
        except AnalysisError as e:
            st.error(str(e))
            st.stop()
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        st.subheader("논문/보고서용 해석")
        st.text_area("해석 문장", build_bivariate_interpretation(result, x, y, x_type, y_type, alpha=alpha), height=220)
        st.caption("주의: 기초분석은 상관성·차이·연관성을 보여주는 도구이며, 별도의 연구설계 없이 인과성을 확정하지 않습니다.")
        st.subheader("결과 표")
        st.dataframe(result.stats_table, use_container_width=True)
        for name, table in result.detail_tables.items():
            st.markdown(f"**{name}**")
            st.dataframe(table, use_container_width=True)
        st.subheader("그래프")
        _plot_for_bivariate(df, x, y, type_map)
        excel_bytes = build_excel_report({"main_result": result.stats_table, **result.detail_tables})
        c1, c2 = st.columns(2)
        c1.download_button("CSV 다운로드", data=dataframe_to_csv_bytes(result.stats_table), file_name="basic_analysis.csv", mime="text/csv")
        c2.download_button("엑셀 다운로드", data=excel_bytes, file_name="basic_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_reg:
    _ensure_loaded(df, type_map)
    analysis_columns = [c for c, t in type_map.items() if t not in {"exclude", "text", "id", "datetime"}]
    dependent = st.selectbox("종속변수", analysis_columns, key="reg_dep")
    independents = st.multiselect("설명변수", [c for c in analysis_columns if c != dependent], key="reg_ind")
    dep_type = type_map[dependent]
    default_family = "logistic" if dep_type == "binary" else "linear"
    family = st.selectbox("회귀 유형", [default_family, "linear", "logistic"] if default_family == "logistic" else [default_family, "logistic"], index=0, key="reg_family")
    if st.button("회귀분석 실행", type="primary"):
        try:
            result = run_regression(df, dependent, independents, type_map, family)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        st.text_area("해석 문장", build_regression_interpretation(result, dependent, family), height=240)
        st.caption("주의: 회귀계수는 다른 설명변수를 함께 고려한 조건부 관계입니다. 관측자료에서는 엄밀한 인과효과와 구분해서 작성하세요.")
        st.dataframe(result.stats_table, use_container_width=True)
        st.plotly_chart(coefficient_bar_plot(result.stats_table), use_container_width=True)
        excel_bytes = build_excel_report({"regression": result.stats_table, **result.detail_tables})
        st.download_button("회귀결과 엑셀 다운로드", data=excel_bytes, file_name="regression_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_causal:
    _ensure_loaded(df, type_map)
    st.markdown("### Difference-in-Differences")
    eligible_outcomes = [c for c, t in type_map.items() if t == "continuous"]
    eligible_treatments = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
    eligible_times = [c for c, t in type_map.items() if t in {"continuous", "datetime", "categorical", "binary"}]
    eligible_units = [c for c, t in type_map.items() if t in {"id", "categorical", "binary", "text"}]
    did_cols = st.columns(3)
    outcome = did_cols[0].selectbox("Outcome", eligible_outcomes, key="did_outcome")
    treatment = did_cols[1].selectbox("Treatment group variable", eligible_treatments, key="did_treat")
    time_var = did_cols[2].selectbox("Time variable", eligible_times, key="did_time")
    treat_values = _to_display_strings(sorted(df[treatment].dropna().astype("string").unique().tolist()))
    time_values = sorted(df[time_var].dropna().astype("string").unique().tolist())
    did_cols2 = st.columns(4)
    treated_value = did_cols2[0].selectbox("처리집단 값", treat_values, key="did_treat_value")
    intervention_value = did_cols2[1].selectbox("개입 시점 값", time_values, key="did_intervention")
    unit_var = did_cols2[2].selectbox("패널 단위 ID (선택)", ["(없음)"] + eligible_units, key="did_unit")
    add_unit_fe = did_cols2[3].checkbox("unit FE 추가", value=False, key="did_unit_fe")
    add_time_fe = st.checkbox("time FE 추가", value=False, key="did_time_fe")
    did_controls = st.multiselect("통제변수", [c for c in eligible_outcomes + eligible_treatments if c not in {outcome, treatment, time_var, unit_var}], key="did_controls")
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
        st.text_area("해석 문장", build_did_interpretation(result, outcome), height=220)
        if "group_time_means" in result.detail_tables:
            st.plotly_chart(did_trend_plot(result.detail_tables["group_time_means"], outcome), use_container_width=True)
        st.dataframe(result.stats_table, use_container_width=True)
        for name, table in result.detail_tables.items():
            with st.expander(name, expanded=name in {"did_pre_post_summary", "group_time_means"}):
                st.dataframe(table, use_container_width=True)

    st.markdown("---")
    st.markdown("### Event Study")
    es_cols = st.columns(4)
    es_outcome = es_cols[0].selectbox("Outcome", eligible_outcomes, key="es_outcome")
    es_treatment = es_cols[1].selectbox("Treatment group variable", eligible_treatments, key="es_treat")
    es_time = es_cols[2].selectbox("Time variable", eligible_times, key="es_time")
    es_treat_values = _to_display_strings(sorted(df[es_treatment].dropna().astype("string").unique().tolist()))
    es_time_values = sorted(df[es_time].dropna().astype("string").unique().tolist())
    es_treated_value = es_cols[3].selectbox("처리집단 값", es_treat_values, key="es_treat_val")
    es_cols2 = st.columns(5)
    es_intervention = es_cols2[0].selectbox("개입 시점 값", es_time_values, key="es_intervention")
    es_unit = es_cols2[1].selectbox("패널 단위 ID (선택)", ["(없음)"] + eligible_units, key="es_unit")
    es_unit_fe = es_cols2[2].checkbox("unit FE 추가", value=False, key="es_unit_fe")
    pre_w = es_cols2[3].number_input("사전 기간 수", min_value=1, max_value=12, value=4, step=1, key="es_pre")
    post_w = es_cols2[4].number_input("사후 기간 수", min_value=1, max_value=12, value=4, step=1, key="es_post")
    es_controls = st.multiselect("통제변수", [c for c in eligible_outcomes + eligible_treatments if c not in {es_outcome, es_treatment, es_time, es_unit}], key="es_controls")
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
        st.text_area("해석 문장", build_event_study_interpretation(result), height=220)
        st.plotly_chart(event_study_plot(result.stats_table), use_container_width=True)
        st.dataframe(result.stats_table, use_container_width=True)
        for name, table in result.detail_tables.items():
            with st.expander(name, expanded=False):
                st.dataframe(table, use_container_width=True)

with tab_survival:
    _ensure_loaded(df, type_map)
    duration_candidates = [c for c, t in type_map.items() if t == "continuous"]
    event_candidates = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
    group_candidates = [c for c, t in type_map.items() if t in {"binary", "categorical"}]
    cov_candidates = [c for c, t in type_map.items() if t in {"continuous", "binary", "categorical"}]
    s_cols = st.columns(3)
    duration_var = s_cols[0].selectbox("Duration variable", duration_candidates, key="surv_duration")
    event_var = s_cols[1].selectbox("Event variable", event_candidates, key="surv_event")
    group_var = s_cols[2].selectbox("Group variable (선택)", ["(없음)"] + group_candidates, key="surv_group")
    event_values = _to_display_strings(sorted(df[event_var].dropna().astype("string").unique().tolist()))
    event_value = st.selectbox("사건 발생으로 볼 값", event_values, key="surv_event_value")
    selected_groups = None
    if group_var != "(없음)":
        group_values = _to_display_strings(sorted(df[group_var].dropna().astype("string").unique().tolist()))
        selected_groups = st.multiselect("비교할 그룹(최대 2개 권장)", group_values, default=group_values[: min(2, len(group_values))], key="surv_group_vals")
    covariates = st.multiselect("Cox 공변량(선택)", [c for c in cov_candidates if c not in {duration_var, event_var, group_var}], key="surv_covs")
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
        st.text_area("해석 문장", build_survival_interpretation(result), height=220)
        if "kaplan_meier_curve" in result.detail_tables:
            st.plotly_chart(kaplan_meier_plot(result.detail_tables["kaplan_meier_curve"]), use_container_width=True)
        st.dataframe(result.stats_table, use_container_width=True)
        for name, table in result.detail_tables.items():
            with st.expander(name, expanded=name in {"group_summary"}):
                st.dataframe(table, use_container_width=True)

with tab_advanced:
    _ensure_loaded(df, type_map)
    eligible = [c for c, t in type_map.items() if t in {"continuous", "categorical", "binary"}]
    selected = st.multiselect("탐색할 변수 선택(최소 3개)", eligible, default=eligible[: min(5, len(eligible))], key="adv_selected")
    correction = st.selectbox("다중비교 보정", ["none", "bonferroni", "holm", "fdr_bh"], index=3, key="adv_corr")
    alpha = st.selectbox("유의수준 alpha", [0.1, 0.05, 0.01], index=1, key="adv_alpha")
    if st.button("전체 변수 탐색 실행", type="primary"):
        if len(selected) < 3:
            st.error("최소 3개 이상의 변수를 선택해주세요.")
            st.stop()
        result = run_bulk_pairwise(df, selected, type_map, correction)
        _display_summary_cards(result.summary)
        st.info(result.narrative)
        st.text_area("해석 문장", build_bulk_interpretation(result, alpha=alpha), height=220)
        st.dataframe(result.stats_table, use_container_width=True)
        st.download_button("탐색결과 CSV 다운로드", data=dataframe_to_csv_bytes(result.stats_table), file_name="bulk_pairwise.csv", mime="text/csv")
