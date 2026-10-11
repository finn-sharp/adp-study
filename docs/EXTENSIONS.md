# 다중·비선형 회귀, 시계열 및 리샘플링

명세의 기준은 각 `.py` 안의 파일 설명, Profile/Result Attributes, 함수 Args/Returns/Raises/Notes다.
추가 실행 예제는 `python -m examples.extended_workflow`로 확인한다.

## 회귀와 시계열

| 모듈 | 실행 함수 | 입력/출력 | 범위 |
|---|---|---|---|
| `src/MachineLearning/Regression/regression.py` | `run_multiple_regression` | 수치 특성 2개 이상 → 모델별 SupervisedOutputDTO 하위 DTO | 다중 선형 OLS, 절편, 분할·CV |
| 동일 | `run_polynomial_regression` | 수치 특성 1개 이상 → 모델별 SupervisedOutputDTO 하위 DTO | 다항·교호항, 선택적 Ridge; 계수에는 선형 |
| 동일 | `run_nonlinear_regression` | 사용자 곡선·초기 계수 → 모델별 SupervisedOutputDTO 하위 DTO | scipy 비선형 최소제곱, 계수 범위·공분산 |
| `src/TimeSeries/arima.py` | `run_arima` | 원 단변량 시계열 → ForecastOutputDTO | (p,d,q), 학습 AIC 후보 선택 |
| 동일 | `run_sarima` | 원 단변량 시계열 → ForecastOutputDTO | (p,d,q)(P,D,Q,s), 계절 후보 선택 |
| 기존 `src/TimeSeries/forecasting.py` | `run_forecast`, `run_varmax` | 시계열 → 기존 결과 | 기존 SARIMAX 엔진·다변량 VARMAX API 유지 |

회귀 모델은 `result.model.predict(new_X)`로 예측한다. 다중회귀 계수는 마지막
`model` 단계의 `coef_`, 곡선 회귀 계수는 `parameters_`다. 비선형 함수에 전달하는
좌표는 `(n_features, n_samples)` 형태다. 최적화의 전역 최적성은 보장하지 않는다.
추론용 p-value/신뢰구간은 별도의 기존 `src/stats/regression.py`를 사용한다.

```python
from src.MachineLearning.Regression import (
    run_multiple_regression, run_polynomial_regression, run_nonlinear_regression,
    PolynomialRegressionProfile, NonlinearRegressionProfile,
)
import numpy as np

multiple = run_multiple_regression(X, y)
polynomial = run_polynomial_regression(X, y, PolynomialRegressionProfile(degree=2))
curve = lambda z, a, b, c: a * np.exp(b * z[0]) + c
nonlinear = run_nonlinear_regression(
    X[:, :1], y, curve, NonlinearRegressionProfile(initial_params=(1., .1, 0.))
)
print(nonlinear.test_metrics)
```

```python
from src.TimeSeries import run_arima, run_sarima, ARIMAProfile, SARIMAProfile
arima = run_arima(series, ARIMAProfile(order=(1, 1, 1), holdout=12))
sarima = run_sarima(series, SARIMAProfile(
    order=(1, 1, 0), seasonal_order=(1, 0, 0, 12), holdout=12
))
print(sarima.predictions, sarima.intervals, sarima.metrics)
```

시계열 입력은 시간 순서·등간격으로 준비하고 원 단위로 전달한다. d/D를 모델이
처리하므로 사전 차분하지 않는다. SARIMA는 외생변수 없이 SARIMAX를 사용하며,
학습에는 최소 두 주기가 필요하다. 보류 구간은 차수 선택이나 적합에 사용하지 않는다.

## 증강 5종: 수치형/범주형 구분

모듈은 `src/Preprocessing/sampling.py`다. 리샘플링은 기존 라벨을 이용하며 라벨을
새로 작성하는 annotation 기법이 아니다. 아래의 데이터 유형은 **특성 X의 유형**이다.
타깃 y는 항상 binary/multiclass의 이산 분류 라벨이며 연속 회귀 타깃은 지원하지 않는다.
숫자 범주 코드도 의미가 범주형이면 `data_kind='categorical'` 또는 `mixed`를 지정한다.

| method | imblearn 기법 | 수치형 | 범주형 | 혼합형 | 생성 방식 |
|---|---|---|---|---|---|
| `random_over` | RandomOverSampler | 지원 | 지원 | 지원 | 기존 행 복제 |
| `smote` | SMOTE | 지원 | 미지원 | 미지원 | 이웃 사이 수치 보간 |
| `adasyn` | ADASYN | 지원 | 미지원 | 미지원 | 이웃 분포에 따른 적응 보간 |
| `smotenc` | SMOTENC | 단독 미지원 | 단독 미지원 | 지원 | 수치 보간 + 범주 유지 |
| `smoten` | SMOTEN | 미지원 | 지원 | 미지원 | 범주형 이웃 기반 합성 |

## 감소 5종: 수치형/범주형 구분

| method | imblearn 기법 | 수치형 | 범주형 | 혼합형 | 감소 방식 |
|---|---|---|---|---|---|
| `random_under` | RandomUnderSampler | 지원 | 지원 | 지원 | 원본 행 무작위 선택 |
| `near_miss` | NearMiss | 지원 | 미지원 | 미지원 | 이웃 거리 기반 선택 |
| `tomek_links` | TomekLinks | 지원 | 미지원 | 미지원 | 경계의 Tomek 쌍 정제 |
| `edited_nearest_neighbours` | EditedNearestNeighbours | 지원 | 미지원 | 미지원 | 이웃 라벨과 불일치한 행 정제 |
| `cluster_centroids` | ClusterCentroids | 지원 | 미지원 | 미지원 | 군집 대표점으로 축약 |

이는 증강 총 5개·감소 총 5개의 분류이며 각 데이터 유형마다 5개가 존재한다는 뜻은 아니다.
범주형을 임의 숫자로 인코딩해 거리 기반 감소 기법에 전달하면 순서·거리 가정이 생긴다.
이 모듈은 이를 자동 허용하거나 모든 기법을 범주형용으로 표시하지 않는다.
TomekLinks/ENN은 정제 기법으로 정확한 균형을 보장하지 않으며 ADASYN도 목표 수와
차이가 생길 수 있다. 항상 `counts_before`/`counts_after`를 확인한다.

```python
from src.Preprocessing import SamplingProfile, resample_training_data
# train/test를 먼저 분리한 뒤, 학습 데이터만 전달한다.
result = resample_training_data(X_train, y_train, SamplingProfile(
    method='smote', data_kind='numeric', neighbors=3
))
print(result.counts_before, result.counts_after)
# 혼합형의 두 번째 열이 범주형일 때:
mixed = resample_training_data(mixed_X_train, y_train, SamplingProfile(
    method='smotenc', data_kind='mixed', categorical_features=[1], neighbors=3
))
```

## 교차검증과 인코딩 위치

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.Preprocessing import build_sampling_pipeline, SamplingProfile
pipeline = build_sampling_pipeline(
    LogisticRegression(max_iter=500), SamplingProfile(method='smote', neighbors=3),
    before_sampling=StandardScaler(),
)
search = GridSearchCV(pipeline, {'model__C': [.1, 1.]}, cv=3)
search.fit(X_train, y_train)
print(search.score(X_test, y_test))
```

Pipeline은 각 학습 폴드에만 변환기와 sampler를 적합하고 검증·테스트의 행 수를
바꾸지 않는다. SMOTENC/SMOTEN 결과를 수치 모델에 전달할 때는
`after_sampling=OneHotEncoder(handle_unknown='ignore')` 같은 인코더를 추가한다.
혼합형의 `categorical_features`는 `before_sampling` 출력 열 위치 기준이다.
ColumnTransformer로 수치형만 스케일링할 때 재배치된 열 순서도 명시해야 한다.
SMOTE/SMOTEN은 각 학습 클래스에 이웃 수보다 많은 표본이 필요하며 부족하면 예외를 낸다.

## 의존성과 검증

- **전체 모듈 Python 3.7 환경:** `requirements-py37.txt` 사용. Python 3.7.12,
  sklearn 1.0.2, imblearn 0.8.1에서 전체 51개 테스트가 skip 없이 통과했습니다.
- Python 3.12 환경: `requirements-review.txt`. 같은 51개 테스트 통과.
- 기존 시험용 `requirements.txt`/`requirements-ml.txt`의 sklearn 0.23.2 조합은 유지합니다.
  선택적 imblearn 0.7에는 SMOTEN이 없으므로, 전체 10종 사용 시 별도 Python 3.7
  가상환경에 `requirements-py37.txt`만 설치하세요. 고정 버전 파일을 서로 합쳐 설치하지 마세요.
- ARIMA/SARIMA, 회귀, 리샘플링 전체 10종, 통계 확장을 실제 Python 3.7에서 검증했습니다.
  버전·범위·선택 기능의 미검증 항목은 [검증 기록](VALIDATION.md)에 구분합니다.

공식 참고: [증강](https://imbalanced-learn.org/stable/over_sampling.html),
[감소](https://imbalanced-learn.org/stable/under_sampling.html),
[누수](https://imbalanced-learn.org/stable/common_pitfalls.html),
[curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html),
[SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html).

반환 객체는 [Output DTO 계약](OUTPUT_DTOS.md)을 따르며 기존 Result 이름은 호환 별칭입니다.
