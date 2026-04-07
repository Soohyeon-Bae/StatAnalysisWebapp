# Stat Explorer Pro

업로드한 CSV/XLSX 데이터를 바탕으로 변수 타입을 자동 추정하고, 기초 통계검정부터 회귀, 인과분석, 생존분석까지 수행하는 Streamlit 웹앱입니다.

## 주요 기능
- CSV/XLSX 업로드
- 변수 타입 자동 추정 + 수동 수정
- 탭 기반 분석 UI
  - 🍰 데이터
  - 🧁 기초분석
  - ☕ 회귀분석
  - 🍪 인과분석
  - 🍩 생존분석
  - 🍫 고급탐색
- 기초분석
  - Pearson / Spearman 상관분석
  - t-test / Mann-Whitney U
  - ANOVA / Kruskal-Wallis
  - Chi-square / Fisher exact
- 회귀분석
  - 선형회귀(OLS)
  - 로지스틱 회귀(Logit)
- 인과분석
  - Difference-in-Differences (DiD)
  - Event Study
  - `policy_after` 같은 before/after 변수 자동 생성 및 세션 유지
- 생존분석
  - Kaplan-Meier
  - Cox proportional hazards
  - duration 변수 자동 생성 및 세션 유지
- 고급탐색
  - 여러 변수쌍 일괄 검정
  - Bonferroni / Holm / FDR(BH) 보정
- 결과 해석문 자동 생성
- CSV/XLSX 다운로드

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 권장 사용 순서
1. 데이터 탭에서 파일 업로드
2. 변수 타입 확인 및 수정
3. 기초분석 또는 회귀분석으로 데이터 구조 파악
4. 인과분석 탭에서 `policy_after` 생성 후 DiD/Event Study 실행
5. 생존분석 탭에서 duration 생성 후 Kaplan-Meier/Cox 실행

## policy_after 관련 안내
- 인과분석 탭의 **Before/After 변수 생성** 버튼을 누르면 새 변수가 `st.session_state['df']`에 저장됩니다.
- 같은 업로드 세션에서는 원본 파일을 다시 읽지 않도록 처리되어, 생성한 `policy_after`가 사라지지 않도록 수정했습니다.
- 데이터 탭 상단에서 현재 저장된 생성 변수를 확인할 수 있습니다.

## 파일 구조
```text
stat_analysis_webapp/
├─ app.py
├─ requirements.txt
├─ README.md
└─ modules/
   ├─ __init__.py
   ├─ analysis_engine.py
   ├─ effect_size.py
   ├─ file_loader.py
   ├─ interpretation.py
   ├─ report_generator.py
   ├─ type_inference.py
   └─ visualizer.py
```

## 주의사항
- 인과분석은 처리집단/통제집단과 사전/사후 구분이 명확할 때 해석력이 높습니다.
- 기초분석과 일반 회귀는 기본적으로 연관성 분석입니다.
- 복합표본설계, 설문 가중치, 고급 패널 인과추론, 다층모형 등은 포함하지 않았습니다.
- Streamlit Cloud에 배포할 때는 `app.py`가 저장소 루트에 있어야 합니다.
