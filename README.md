# Stat Explorer Pro - Research Upgrade

기존 탭 구조를 유지하면서 인과분석 탭 안에 **논문형 분석 모드(PSM + Balance + DiD + Event Study + ZIP export)** 를 추가한 최종본입니다.

## 포함 기능
- 데이터 업로드 / 변수 타입 추론 / 수동 수정
- 기초분석
- 회귀분석
- 인과분석
  - 기본 DiD
  - 기본 Event Study
  - **논문형 분석 모드**
    - Step 1 변수설정
    - Step 2 성향점수 추정
    - Step 3 매칭
    - Step 4 균형진단(Love plot, SMD)
    - Step 5 매칭 표본 DiD
    - Step 6 매칭 표본 Event Study
    - Step 7 전체 결과 ZIP 다운로드
- 생존분석
- 고급탐색

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
