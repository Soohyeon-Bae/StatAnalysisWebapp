# Stat Explorer

업로드한 CSV/XLSX 데이터를 바탕으로 변수 타입을 자동 추정하고, 변수 간 통계적 유의성을 표와 그래프로 보여주는 Streamlit 웹앱입니다.

## 주요 기능
- CSV/XLSX 업로드
- 변수 타입 자동 추정 + 수동 수정
- 두 변수 분석
  - Pearson / Spearman 상관분석
  - t-test / Mann-Whitney U
  - ANOVA / Kruskal-Wallis
  - Chi-square / Fisher exact
- 회귀분석(beta)
  - 선형회귀(OLS)
  - 로지스틱 회귀(Logit)
- 전체 변수 탐색(beta)
  - 여러 변수 쌍을 한 번에 검정
  - Bonferroni / Holm / FDR(BH) 보정
- 결과 CSV/XLSX 다운로드

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 파일 구조
```text
stat_analysis_webapp/
├─ app.py
├─ requirements.txt
├─ README.md
└─ modules/
   ├─ analysis_engine.py
   ├─ effect_size.py
   ├─ file_loader.py
   ├─ report_generator.py
   ├─ type_inference.py
   └─ visualizer.py
```

## 주의사항
- 현재는 가중치가 있는 survey data, 복합표본설계, 패널 고정효과 같은 고급 분석은 포함하지 않았습니다.
- 회귀분석은 beta 수준으로, 복잡한 데이터 전처리나 다중공선성 진단 UI는 아직 넣지 않았습니다.
- 파일 업로드 후 데이터는 Streamlit의 메모리 버퍼에서 처리됩니다. Streamlit 문서에 따르면 업로드 파일은 기본적으로 RAM의 `BytesIO` 버퍼에 저장되며 기본 업로드 한도는 200MB입니다. citeturn220482search0turn220482search6
- 앱이 사용하는 공식 API/함수 선택은 Streamlit file uploader, SciPy 통계 함수, statsmodels formula API 문서를 기준으로 맞췄습니다. citeturn220482search0turn220482search1turn220482search2turn220482search8turn220482search7
