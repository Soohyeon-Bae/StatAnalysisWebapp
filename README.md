# Stat Explorer Pro - Final build

이 패키지는 다음을 포함합니다.

- app.py 완전 최종본
- modules/analysis_engine.py 안정화 버전
- modules/interpretation.py 확장 버전
- modules/file_loader.py 안정화 버전
- modules/type_inference.py 안정화 버전

## 주요 개선
- policy_after 생성 후 세션 유지
- 로지스틱 회귀 binary 종속변수 자동 인코딩
- DiD / Event Study 시간변수 처리 안정화
- Event Study pre-trend joint p-value 계산
- 각 분석 해석 TXT 다운로드
- 숫자형이 datetime으로 과하게 잡히는 문제 완화

## 적용
이 압축을 풀어서 기존 프로젝트에 덮어쓰면 됩니다.
