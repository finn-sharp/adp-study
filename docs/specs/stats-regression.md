# `src/stats/regression.py`

[소스 코드](../../src/stats/regression.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
regression.py - OLS/Logit 추론, 회귀 진단 및 정보기준 변수 선택
================================================================================
설명: 통계분석 PDF의 회귀계수 추론, VIF, 잔차 진단, 전진/후진/단계적 선택을 재구성합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 회귀 추론과 변수 선택 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `RegressionProfile`

통계적 회귀 조건.

Attributes:
    method: ols 또는 logit (타깃은 0/1).
    alpha: 계수 신뢰구간의 유의수준 (0, 1).
    fit_intercept: 절편 포함 여부.
    polynomial_degree: 다항/교호 특성 차수 (1 이상). 기본 1.
    max_iter: Logit 최대 반복 수.

## `RegressionOutputDTO`

전체 입력의 회귀 추론 결과. 보류 테스트 성능이 아닙니다.

Attributes:
    model: statsmodels 적합 결과 객체. 예측 시 같은 설계행렬 구성이 필요합니다.
    design: 적합에 사용한 DataFrame (절편 및 선택적 다항 특성 포함).
    transformer: 다항 변환기. degree=1이면 None.
    coefficient_table: 계수, 표준오차, p-value, 신뢰구간.
    diagnostics: OLS의 Durbin-Watson, 잔차 정규성, Breusch-Pagan, VIF;
        Logit은 AIC/BIC/LLR p-value와 pseudo_r2 (OLS R²와 다른 지표).
    residuals: 입력 순서의 잔차.

## `run_regression_inference`

회귀계수 추론과 잔차 진단을 수행합니다.

Args:
    X: 유한한 수치 특성. DataFrame이면 고유한 열 이름을 사용합니다.
    y: 1차원 수치 타깃. Logit은 두 클래스 0/1.
    profile: 회귀 추론 설정.
Returns:
    RegressionOutputDTO: 적합 모델, 계수 표 및 진단.
Raises:
    ValueError: 잘못된 입력, 순위 부족, 잔차 자유도 부족 또는 Logit 타깃 오류.
    ImportError: statsmodels가 설치되지 않은 경우.

Notes:
    OLS 계수 추론은 모형·오차 가정에 의존합니다. 잔차 검정만으로 가정을 보장하지 않습니다.
    변수 선택 후 같은 데이터의 p-value는 선택 과정의 불확실성을 반영하지 않습니다.

## `SelectionProfile`

OLS 정보기준 변수 선택 조건.

Attributes:
    method: forward, backward, stepwise.
    criterion: aic 또는 bic (작을수록 좋음).
    max_steps: 반복 상한. 완료 여부는 result.converged에 반환합니다.
    min_improvement: 최소 정보기준 개선량 (양수).

## `SelectionOutputDTO`

변수 선택 결과.

Attributes:
    selected_features: 선택한 입력 열 이름.
    model: 절편 포함 statsmodels OLS 적합 객체.
    history: 단계별 선택 특성과 정보기준 기록.
    converged: 상한 도달 전에 개선 후보가 없어 종료했는지 여부.

## `run_variable_selection`

전진·후진·단계적 OLS 변수 선택을 수행합니다.

Args:
    X: 학습용 수치 특성. DataFrame의 열 이름은 고유해야 합니다.
    y: 학습용 수치 타깃.
    profile: 선택 알고리즘 및 정보기준.
Returns:
    SelectionOutputDTO: 최종 특성, 적합 객체, 이력 및 종료 상태.
Raises:
    ValueError: 잘못된 조건, 입력 또는 초기 모형의 순위·자유도 부족.

Notes:
    이 실행기는 탐색적 변수 선택입니다. 일반화 평가는 별도 보류 데이터에서 수행하세요.
