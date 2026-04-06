from __future__ import annotations

import math
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.analysis_engine import AnalysisError, run_bivariate_analysis, run_bulk_pairwise, run_regression
from modules.file_loader import get_excel_sheet_names, load_uploaded_file
from modules.report_generator import build_excel_report, dataframe_to_csv_bytes
from modules.type_inference import TYPE_OPTIONS, build_profile_table
from modules.visualizer import box_plot, heatmap_plot, scatter_plot, stacked_bar_plot, violin_plot


st.set_page_config(page_title="Stat Explorer", layout="wide")
st.title("Stat Explorer")
st.caption("엑셀/CSV 업로드 → 변수 타입 확인 → 통계검정 → 그래프/표 다운로드")


def _infer_type_map(profile_df: pd.DataFrame) -> dict[str, str]:
    return dict(zip(profile_df["column"], profile_df["user_type"]))


def _display_summary_cards(summary: dict):
    cols = st.columns(min(4, max(1, len(summary))))
    items = list(summary.items())
    for idx, (k, v) in enumerate(items[: len(cols)]):
        if isinstance(v, float):
            if math.isnan(v):
                display = "NA"
            else:
                display = f"{v:.4g}"
        else:
            display = str(v)
        cols[idx].metric(k, display)


def _plot_for_result(df: pd.DataFrame, x: str, y: str, type_map: dict[str, str], recommended_test: str):
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


uploaded = st.file_uploader("CSV 또는 XLSX 파일 업로드", type=["csv", "xlsx"])

if uploaded is not None:
    sheet_name = None
    if uploaded.name.lower().endswith(".xlsx"):
        sheets = get_excel_sheet_names(uploaded)
        sheet_name = st.selectbox("시트 선택", sheets)

    try:
        df, active_sheet = load_uploaded_file(uploaded, sheet_name)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()

    st.success(f"파일 로드 완료: {uploaded.name} / rows={len(df):,} / cols={len(df.columns):,}")

    with st.expander("데이터 미리보기", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    profile_df = build_profile_table(df)
    profile_df["user_type"] = profile_df["inferred_type"]

    st.subheader("1) 변수 타입 확인 및 수정")
    edited = st.data_editor(
        profile_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "user_type": st.column_config.SelectboxColumn(
                "user_type",
                help="자동 추정된 변수 타입을 수정할 수 있습니다.",
                width="medium",
                options=TYPE_OPTIONS,
            )
        },
        disabled=["column", "inferred_type", "non_null_count", "unique_count", "missing_count", "sample_values"],
        key="type_editor",
    )

    type_map = _infer_type_map(edited)
    analysis_columns = [c for c, t in type_map.items() if t != "exclude"]

    st.subheader("2) 분석 모드 선택")
    mode = st.radio(
        "분석 모드",
        ["두 변수 분석", "회귀분석(beta)", "전체 변수 탐색(beta)"],
        horizontal=True,
    )

    if mode == "두 변수 분석":
        left, right = st.columns(2)
        with left:
            x = st.selectbox("변수 X", analysis_columns)
        with right:
            y = st.selectbox("변수 Y", [c for c in analysis_columns if c != x])

        alpha = st.selectbox("유의수준 alpha", [0.1, 0.05, 0.01], index=1)
        x_type, y_type = type_map[x], type_map[y]

        test_options = ["auto"]
        if x_type == "continuous" and y_type == "continuous":
            test_options += ["Pearson", "Spearman"]
        elif (x_type in {"categorical", "binary"} and y_type == "continuous") or (x_type == "continuous" and y_type in {"categorical", "binary"}):
            n_groups = df[x].nunique(dropna=True) if x_type in {"categorical", "binary"} else df[y].nunique(dropna=True)
            if n_groups == 2:
                test_options += ["t-test", "Mann-Whitney U"]
            else:
                test_options += ["ANOVA", "Kruskal-Wallis"]
        elif x_type in {"categorical", "binary"} and y_type in {"categorical", "binary"}:
            test_options += ["Chi-square", "Fisher exact"]

        chosen_test = st.selectbox("검정 방법", test_options)
        if st.button("분석 실행", type="primary"):
            try:
                result = run_bivariate_analysis(
                    df=df,
                    x=x,
                    y=y,
                    x_type=x_type,
                    y_type=y_type,
                    alpha=alpha,
                    method_override=None if chosen_test == "auto" else chosen_test,
                )
            except AnalysisError as e:
                st.error(str(e))
                st.stop()

            st.subheader("결과 요약")
            _display_summary_cards(result.summary)
            st.info(result.narrative)

            st.subheader("결과 표")
            st.dataframe(result.stats_table, use_container_width=True)
            for name, table in result.detail_tables.items():
                st.markdown(f"**{name}**")
                st.dataframe(table, use_container_width=True)

            st.subheader("그래프")
            _plot_for_result(df, x, y, type_map, result.recommended_test)

            export_frames = {"main_result": result.stats_table}
            export_frames.update({k: v for k, v in result.detail_tables.items()})
            excel_bytes = build_excel_report(export_frames)
            csv_bytes = dataframe_to_csv_bytes(result.stats_table)
            c1, c2 = st.columns(2)
            c1.download_button(
                "결과표 CSV 다운로드",
                data=csv_bytes,
                file_name="stat_result.csv",
                mime="text/csv",
            )
            c2.download_button(
                "엑셀 리포트 다운로드",
                data=excel_bytes,
                file_name="stat_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    elif mode == "회귀분석(beta)":
        st.caption("연속형 종속변수는 선형회귀, 이분형 종속변수는 로지스틱 회귀를 권장합니다.")
        dependent = st.selectbox("종속변수", analysis_columns)
        independent_candidates = [c for c in analysis_columns if c != dependent and type_map[c] != "text" and type_map[c] != "id"]
        independents = st.multiselect("설명변수", independent_candidates)
        dep_type = type_map[dependent]
        default_family = "logistic" if dep_type == "binary" else "linear"
        family = st.selectbox("회귀 유형", [default_family, "linear", "logistic"] if default_family == "logistic" else [default_family, "logistic"], index=0)

        if st.button("회귀분석 실행", type="primary"):
            try:
                result = run_regression(df, dependent, independents, type_map, family)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
                st.stop()
            st.subheader("모형 요약")
            _display_summary_cards(result.summary)
            st.info(result.narrative)
            st.dataframe(result.stats_table, use_container_width=True)
            coef_df = result.stats_table.copy()
            if "Coef." in coef_df.columns:
                fig = go.Figure(go.Bar(x=coef_df["term"], y=coef_df["Coef."]))
                fig.update_layout(title="Regression Coefficients", xaxis_title="term", yaxis_title="coefficient")
                st.plotly_chart(fig, use_container_width=True)
            excel_bytes = build_excel_report({"regression": result.stats_table})
            st.download_button(
                "회귀결과 엑셀 다운로드",
                data=excel_bytes,
                file_name="regression_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    else:
        eligible = [c for c in analysis_columns if type_map[c] in {"continuous", "categorical", "binary"}]
        selected = st.multiselect("탐색할 변수 선택(최소 3개)", eligible, default=eligible[: min(5, len(eligible))])
        correction = st.selectbox("다중비교 보정", ["none", "bonferroni", "holm", "fdr_bh"], index=3)
        if st.button("전체 변수 탐색 실행", type="primary"):
            if len(selected) < 3:
                st.error("최소 3개 이상의 변수를 선택해주세요.")
                st.stop()
            result = run_bulk_pairwise(df, selected, type_map, correction)
            st.subheader("탐색 결과")
            _display_summary_cards(result.summary)
            st.info(result.narrative)
            st.dataframe(result.stats_table, use_container_width=True)
            st.download_button(
                "탐색 결과 CSV 다운로드",
                data=dataframe_to_csv_bytes(result.stats_table),
                file_name="pairwise_screening.csv",
                mime="text/csv",
            )
else:
    st.markdown(
        """
### 사용 방법
1. CSV 또는 XLSX 파일을 업로드합니다.
2. 변수 타입을 확인하고 필요하면 수정합니다.
3. 두 변수 분석, 회귀분석, 전체 변수 탐색 중 하나를 선택합니다.
4. 결과표와 그래프를 확인하고 CSV/XLSX로 다운로드합니다.

### 지원 기능
- 연속형-연속형: Pearson / Spearman 상관분석
- 범주형-연속형: t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis
- 범주형-범주형: Chi-square, Fisher exact
- 회귀분석(beta): OLS, Logit
- 전체 변수 탐색(beta): 다중비교 보정 포함 pairwise screening
        """
    )
