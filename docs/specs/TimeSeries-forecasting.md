# `src/TimeSeries/forecasting.py`

[소스 코드](../../src/TimeSeries/forecasting.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
forecasting.py - ARIMA/SARIMAX/VARMAX 시간 순서 예측
================================================================================
설명:
    - 시계열분석 PDF의 모델을 statsmodels.tsa.arima.model 등 지원 API로 구현합니다.
    - 뒤쪽 holdout을 분리하고 AIC 탐색은 앞쪽 학습 데이터에서만 수행합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 시간 분할·후보 이력·예측구간 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `ForecastProfile`

단변량 예측 조건.

Attributes:
    method: arima 또는 sarimax.
    order: (p, d, q), 비음수 정수.
    seasonal_order: SARIMAX (P, D, Q, s). 계절항이 있으면 s >= 2.
    holdout: 마지막 테스트 관측 수. 학습에는 최소 8개 관측이 남아야 함.
    alpha: 예측구간의 유의수준 (0, 1).
    order_candidates: 학습 데이터 AIC로 비교할 order 목록. None이면 order만 적합.
    seasonal_candidates: SARIMAX 계절 후보 목록. None이면 seasonal_order만 사용.

## `ForecastOutputDTO`

시간 순서 보류 예측 결과.

Attributes:
    model: 학습 구간에만 적합된 statsmodels 결과 객체.
    train: 학습 원 시계열.
    test: 마지막 holdout 원 시계열.
    predictions: test 인덱스로 정렬된 원 단위 예측.
    intervals: test 인덱스의 lower/upper 예측구간.
    metrics: 보류 구간 MAE/MSE/RMSE.
    best_order: 선택된 비계절 order.
    best_seasonal_order: 선택된 계절 order.
    candidates: 후보별 AIC, 수렴 여부 및 실패 원인. 비수렴 후보는 선택하지 않음.

## `SeriesDiagnosticsOutputDTO`

시계열 진단 출력 DTO.

Attributes:
    transformed: 변환·차분 후 시계열, 원 인덱스 보존.
    boxcox_lambda: Box-Cox 사용 시 적합 lambda, 나머지는 None.
    adf_statistic, adf_p_value, reject_unit_root: ADF 통계량/p/단위근 기각 여부.
    acf, pacf: 0시차를 포함한 상관 배열.
    decomposition: 선택적 statsmodels 분해 객체(runtime). 미지정 시 None.
    decomposition_components: trend/seasonal/resid/observed 시계열 사전. 미지정 시 빈 사전.

## `MultivariateForecastOutputDTO`

VARMAX 출력 DTO.

Attributes:
    model: 적합 statsmodels 결과 객체(runtime).
    train, test: 시간 순서 학습/보류 DataFrame.
    predictions: test 인덱스 및 입력 열 순서의 예측 표.
    metrics: 열별 MAE/RMSE 사전.
    converged: 적합 수렴 여부; False인 결과는 실행기가 거부함.

## `validate_series`

시간 순서가 명시된 단변량 입력을 검증합니다.

Args:
    data: 1차원 배열 또는 정렬된 고유 인덱스의 Series.
    min_size: 최소 관측 수.
Returns:
    pd.Series: float 시계열. 입력을 재정렬하거나 결측치를 자동 보간하지 않습니다.
Raises:
    ValueError: 결측·무한대, 잘못된 차원·크기 또는 중복/미정렬 인덱스.

## `run_forecast`

시간 순서 분할, 선택적 학습 AIC 탐색 및 보류 구간 예측을 수행합니다.

Args:
    data: 일정 간격의 단변량 원 시계열. 모델의 d/D가 차분을 수행합니다.
    profile: 예측 모형·후보·보류 수 설정.
Returns:
    ForecastOutputDTO: 학습 모델, 원 단위 예측·구간, 평가 및 후보 이력.
Raises:
    ValueError: 잘못된 입력·차수·구간 또는 모든 후보의 적합 실패/비수렴.
    ImportError: statsmodels 미설치.

Notes:
    미리 차분한 입력에 d/D를 다시 적용하면 이중 차분됩니다. 원 시계열을 전달하세요.
    외생변수와 미래 관측을 사용한 재적합은 이 실행기에 포함하지 않습니다.

## `SeriesDiagnosticsProfile`

정상성·변환 조건.

Attributes:
    transformation: none, log, boxcox. log/boxcox 입력은 양수.
    difference_order: 반복 차분 횟수 (0 이상의 정수). diff(2)로 대체하지 않습니다.
    lags: ACF/PACF 최대 시차. 변환 후 길이의 절반 미만.
    alpha: ADF 유의수준. 귀무가설은 단위근 존재.
    seasonal_period: 분해 주기. None이면 분해 생략. 최소 두 주기 데이터 필요.

## `run_series_diagnostics`

변환·반복 차분 후 ADF/ACF/PACF와 선택적 계절 분해를 반환합니다.

Args:
    data: 원 단변량 시계열.
    profile: 변환·차분·시차·주기 설정.
Returns:
    SeriesDiagnosticsOutputDTO: transformed, boxcox_lambda, adf_statistic, adf_p_value,
        reject_unit_root, acf, pacf, decomposition. ADF 기각은 정상성의 완전한 보장이 아님.
Raises:
    ValueError: 잘못된 설정, 상수 시계열 또는 길이 부족.
    ImportError: statsmodels 미설치.

## `MultivariateForecastProfile`

다변량 VARMAX 조건.

Attributes:
    order: (p, q). 기본 VAR(1). p/q 동시 양수는 식별·수렴에 유의해야 함.
    holdout: 마지막 테스트 관측 수.
    max_iter: 적합 최대 반복 수.

## `run_multivariate_forecast`

여러 시계열을 시간 순서대로 분할하여 VARMAX 예측을 수행합니다.

Args:
    data: 최소 두 열의 유한한 수치 DataFrame. 고유한 시간 오름차순 인덱스.
    profile: VARMAX 차수·보류 수 설정.
Returns:
    MultivariateForecastOutputDTO: model, train, test, predictions, metrics (열별 MAE/RMSE), converged.
Raises:
    ValueError: 잘못된 입력/차수/보류 구간 또는 비수렴.
    ImportError: statsmodels 미설치.
