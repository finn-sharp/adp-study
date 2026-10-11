# `src/MachineLearning/Linear/numerical.py`

[소스 코드](../../src/MachineLearning/Linear/numerical.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
numerical.py - 최소제곱법의 정규방정식/SVD/TSVD와 릿지 폐형식 풀이
================================================================================
설명: 02/05장의 수치 풀이를 비교할 수 있는 교육용 실행기입니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 수치 풀이와 특이값 진단 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `NumericalProfile`

수치 풀이 조건.

Attributes:
    method: normal, svd, tsvd, ridge.
    fit_intercept: 절편 열을 추가할지 여부. 릿지에서는 절편을 규제하지 않음.
    n_components: TSVD에서 유지할 특이값 수. 1 이상 min(행, 열) 이하.
    rcond: 유효 특이값 판정 기준 (최대 특이값의 비율, 양수).
    alpha: 릿지 L2 규제 강도 (양수).

## `NumericalOutputDTO`

교육용 수치 회귀 결과.

Attributes:
    coefficients: 입력 특성 순서의 계수.
    intercept: 절편. fit_intercept=False이면 0.
    predictions: X_predict를 전달하면 해당 행의 예측, 생략하면 학습 적합값.
    singular_values: 절편 포함 설계행렬의 특이값.
    rank: rcond에 따른 설계행렬 유효 계수 (TSVD 유지 수와 다를 수 있음).
    condition_number: 설계행렬 조건수. 최소 특이값이 0이면 inf.

## `run_numerical_regression`

교육용 회귀 풀이를 수행합니다. 자동 분할·일반화 평가를 하지 않습니다.

Args:
    X: 적합용 수치형 2차원 특성.
    y: 길이가 같은 유한한 1차원 수치 타깃.
    profile: 풀이 조건.
    X_predict: 선택적 예측 특성. X와 같은 열 수·순서.
Returns:
    NumericalOutputDTO: 계수, 예측 및 수치 안정성 진단.
Raises:
    ValueError: 잘못된 method, 차원, rcond/alpha 또는 TSVD 유지 수.
    np.linalg.LinAlgError: 정규방정식에서 특이한 행렬을 만난 경우.

Notes:
    정규방정식은 조건수가 큰 데이터에 불안정합니다. 기본은 SVD 풀이입니다.
