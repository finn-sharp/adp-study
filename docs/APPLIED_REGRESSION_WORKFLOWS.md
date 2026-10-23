# 모의10회 적용 모듈과 회귀 분석 경계

기존 저장소를 탐색한 결과 Grid/Random/Optuna, 샘플링12종, OLS 추론·잔차/VIF/Cook·Feature 처리·ROC/AP/보정 그림은 재사용할 수 있었습니다. 부족한 **8개 전용 모듈**과 비선형 회귀의 CV adapter를 추가했습니다. 모든 공개 모듈은 작성 정보·변경 이력·7개 섹션·Profile/JSON/전용 DTO·전체 필드와 함수 명세를 유지합니다. 회귀 통계·Feature 처리·모형·그림 책임을 합치지 않습니다.

| 경로 | 책임 |
| --- | --- |
| stats/regression/information_selection | 같은 행·원 반응의 OLS AIC/BIC 탐욕 후진 선택, CV clone transformer |
| MachineLearning/LinearModels/weighted_regression | 학습에서 log(잔차²) 분산 모형을 추정하는 FGLS와 원단위 예측 |
| MachineLearning/GeneralizedLinearModels/logistic | 0/1 Binomial logit, z·오즈비·LR·pseudo R²·GLM 영향 |
| MachineLearning/GeneralizedLinearModels/poisson | 정수 횟수·log mean·알려진 log exposure offset·과산포 점검 |
| MachineLearning/GeneralizedLinearModels/gamma | 양수 연속량·log mean·phi mu² 분산·적절한 잔차 |
| TimeSeries/rolling_selection | 과거 expanding CV 차수 선택, 미래 봉인·naive·MASE·조건부 구간 |
| Visualization/regression/model_comparison | caller ax의 동일 원단위 CV 점수 비교 |
| Visualization/regression/glm_residuals | caller ax의 Pearson/deviance 대 적합평균, family별 해석 |

## 모형과 선택의 사용 순서

`RegressionEstimator(information_criterion='bic')`는 기존 학습 대치·변환·VIF 뒤 동일 원 반응으로 후진 선택하고 최종 OLS를 적합합니다. 새 행에도 같은 선택 열을 적용합니다. 목표 변환과 AIC/BIC 선택의 혼용은 거부하며 log(y) 후보는 CV의 원 y 단위 점수로 비교합니다. 선택기는 학습 원행을 줄이거나 순서를 바꾸지 않습니다. 각 단계에서 가능한 한 열씩 제거해 기준이 개선되는 경우만 채택합니다. 탐욕적 해가 전체 부분집합의 최적해라는 보장은 없습니다. 선택 후 p값은 선택 편향을 보정하지 않습니다.

`FeasibleWLSRegressor`는 sklearn clone 가능한 미적합 추정기로, 학습 내 중앙값 대치·표준화 후 OLS→log(max(잔차²,floor)) 분산 회귀→역분산 WLS를1~5회 수행합니다. 가중치 비율을 제한하고 평균1로 정규화합니다. 검증/시험 y를 분산 추정에 넣지 않습니다. 학습한 가중치가 참 분산은 아니므로 t/p/CI는 추정 가중치에 조건부 근사입니다. weighted_R²와 원반응 예측 R², 조건부 WLS AIC/BIC를 구분하며 다른 가중치/OLS의 정보량과 단순 비교하지 않습니다.

기존 `build_nonlinear_regressor`는 nonlinear_regression의 Profile/JSON과 동일한 평균함수·초기값·계수 경계를 사용해 단일 Feature의 CV adapter를 만듭니다. 적합은 기존 runner로 수행하고 예측만 학습 계수를 사용합니다. 시그모이드 성장곡선의 연속 반응은 이진 logit과 다릅니다. 파라미터 공분산은 국소 선형화 근사이고 초기값·비식별·외삽을 검토해야 합니다. 알려진 sigma는 adapter에서 지원하지 않고 기존 직접 runner에서 명시합니다.

## GLM의 추론·정보량·진단

설명용 runner는 **명시 학습 행을 적합**하며 자체 train/test 분할이나 튜닝을 하지 않습니다. 결측/상수/열 순서/유한값/full rank/반응 정의역을 확인하고 미수렴·분리 경고를 오류로 보고합니다. 전처리를 fit한 뒤 선택 X_new로 예측할 수 있습니다. logistic y는0/1 두 값, Poisson y는비음수 정수, Gamma y는양수 연속입니다. 알려진 Poisson log exposure는 고정 offset으로 적합·예측에 같은 행 순서로 전달합니다.

AIC는 backend 정의, BIC는 -2llf+log(n)k(k=평균계수 개수)로 반환합니다. Gamma Pearson 추정 scale에 조건부인 정보량은 근사이며 이항/Poisson의 고정 scale1 우도와 같은 성질로 주장하지 않습니다. 정보량은 같은 family·반응 측도·행·offset 조건 안에서 비교합니다. 이산 Poisson과 연속 Gamma 사이의 수치를 그대로 우도 우열로 해석하지 않습니다.

계수는 z/양측p/CI/exp(coef)입니다. Binomial/Poisson은 같은 offset의 절편만 모형에 대한 점근 LR 및 McFadden pseudo R²를 제공합니다. Gamma는 이 LR/pseudo R²를 제공하지 않습니다. pseudo R²는 보통 설명력 R²와 다릅니다. HC0의 공분산은 우도 사양 자체를 수정하지 않으므로 LR을 robust 검정으로 해석하면 안 됩니다.

표본별 Pearson/deviance 잔차·leverage·GLM Cook 근사를 반환합니다. OLS 잔차 정규성·t/전체F를 모든 GLM에 강요하지 않습니다. logistic는 log odds 사양·독립성·분리·판별/보정, Poisson은 조건부 평균=분산·과산포·노출·0, Gamma는 양수 연속량·log mean·Var=phi mu²를 검토합니다. Cook 기준 초과는 오류/삭제 증명이 아닙니다.

## 튜닝·샘플링·시간순 검증

모의는 기존 run_grid_search/run_optuna_search를 그대로 사용합니다. 회귀의 정보량 선택·명시 Cook 처리와 분산 추정은 estimator.fit 안에서 폴드마다 새로 적합합니다. 분류는 imblearn Pipeline의 대치→scale→VIF→sampler→L2 logistic를 사용해 CV 학습에서만 샘플링합니다. test는 자연 발생률·원행을 보존합니다. 사전0.5 임계값과 비용을 보고하며 test에서 임계값을 최적화하지 않습니다. 설명용 자연 발생률의 비벌점 GLM은 샘플링된 L2 예측모형과 분리해 해석합니다.

시계열 rolling은 마지막 horizon을 봉인하고 그 이전 folds×validation_size 구간을 확장 학습·연속 검증합니다. 차수는 평균 RMSE로 선택한 뒤 과거 전체에 재적합합니다. 후보 미수렴/실패는 표에 기록하고 적격 후보가 없으면 오류입니다. ARIMA 또는 고정 계절(1,0,0,s)의 SARIMA를 재사용합니다. AIC/BIC는 선택 모형의 학습 지표로 보고하고 다른 차분·우도 후보의 비교는 rolling 점수로 판단합니다. 잔차는 초기화 구간을 제외하고 Ljung-Box에서 p+q+P+Q를 차감합니다. naive는 마지막 값 또는 마지막 주기 반복, MASE 분모는 학습의 주기별 절대차 평균이며0이면 None입니다.

## 적용·실행·검증

[응용 모의10회](../기출문제/모의고사/응용10회/README.md)는 회귀36점·분류40점·시계열24점, 각15문항100점입니다. 모두 창작 합성 자료이며 기준/선택 보류 ΔR²가 음수여도 그대로 보고합니다. 검정 비기각을 가정의 증명으로 표시하지 않습니다. 기존 추론 가정과 조건부/선택 후 한계는 [회귀·분포 명세](REGRESSION_DISTRIBUTION_MODULES.md)에 있습니다.

오프라인8개 모듈 실행: `python -m examples.applied_model_examples`. 실제 UCI6개 skeleton은 각 분야 skeleton/information_selection·weighted_regression·logistic·poisson·gamma·rolling_selection입니다. Wine 품질 OLS/WLS, Wine 양성 기준6 binary logit, Wine의양수 fixed acidity Gamma, Bike의날씨 설명 Poisson, 날짜순 Bike rolling을 사용하며 분포의 완벽 적합이나 실제 미래 운영 성능을 보장하지 않습니다. import에서 다운로드·적합하지 않습니다.

단위검사는 native OLS/GLM/WLS 계수·정보량·예측과 선택 경로·clone·미래값 변화 격리·정의역·Axes/DTO 계약을 대조합니다. 콘텐츠 검사는 baseline OLS를 독립 재적합하고 원단위 지표·rank-U AUC·Brier·오분류 비용·naive·MASE·원행/미래 경계를 직접 계산합니다. 문제지 전체40쪽을 렌더하고 동일 지문·배점·A4 경계를 검사합니다.

기술 근거: [statsmodels GLM](https://www.statsmodels.org/stable/glm.html), [sklearn 탐색](https://scikit-learn.org/stable/modules/grid_search.html), [imblearn 누출](https://imbalanced-learn.org/stable/common_pitfalls.html).

## regression-information-selection

`src.stats.regression.information_selection`; JSON `regression-information-selection.json`.

### InformationSelectionProfile

```text
전용 설정의 전체 필드 명세.

Attributes:
    criterion (str): 동일 반응·동일 행 OLS의 aic/bic 기본 'bic'.
    min_features (int): 남길 최소 Feature 수, 1 이상 기본 1.
    max_steps (int): 최대 후진 제거 단계, 0 이상 기본 30.
    improvement (float): 정보량 감소 허용값, 비음수 기본 1e-08.
```
### InformationSelectionOutputDTO

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    selector (Any): 적합 InformationSelector; JSON 제외.
    model (Any): 최종 조건부 statsmodels OLS; JSON 제외.
    selected_features (List[str]): 원 열 순서의 선택 Feature.
    history (pd.DataFrame): 단계·제거 열·aic/bic·채택·후보 기록.
    parameters (Dict[str,Any]): 적용 설정.
    warnings (List[str]): 탐욕 선택·선택 후 추론·척도 비교 한계.
```

```text
run_information_selection(X, y, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    InformationSelectionOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

## glm-logistic

`src.MachineLearning.GeneralizedLinearModels.logistic`; JSON `glm-logistic.json`.

### LogisticGLMProfile

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 계수·구간 유의수준 (0,1) 기본 0.05.
    maxiter (int): IRLS 최대 적합 반복, 양수 기본 100.
    cov_type (str): nonrobust/HC0 robust 공분산 기본 'nonrobust'.
```
### LogisticGLMOutputDTO

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model (Any): 적합 statsmodels GLM; JSON 제외.
    coefficients (pd.DataFrame): term·coefficient·se·z·양측p·CI·exp(coef).
    fit_metrics (Dict[str,Any]): family·link·llf·AIC·BIC(평균계수 수 k)·deviance·Pearson dispersion·McFadden(null 비교) 또는 None.
    diagnostics (pd.DataFrame): row_position·fitted_mean·Pearson/deviance잔차·leverage·GLM Cook 근사.
    predictions (Optional[np.ndarray]): 선택 X_new의 조건부 평균; logistic은 성공확률.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 분포/링크/노출/추론·선택 한계.
```

```text
run_logistic(X, y, X_new=None, offset=None, new_offset=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    X_new: 전용 명세의 X_new 입력.
    offset: 전용 명세의 offset 입력.
    new_offset: 전용 명세의 new_offset 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    LogisticGLMOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

## glm-poisson

`src.MachineLearning.GeneralizedLinearModels.poisson`; JSON `glm-poisson.json`.

### PoissonGLMProfile

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 계수·구간 유의수준 (0,1) 기본 0.05.
    maxiter (int): IRLS 최대 적합 반복, 양수 기본 100.
    cov_type (str): nonrobust/HC0 robust 공분산 기본 'nonrobust'.
```
### PoissonGLMOutputDTO

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model (Any): 적합 statsmodels GLM; JSON 제외.
    coefficients (pd.DataFrame): term·coefficient·se·z·양측p·CI·exp(coef).
    fit_metrics (Dict[str,Any]): family·link·llf·AIC·BIC(평균계수 수 k)·deviance·Pearson dispersion·McFadden(null 비교) 또는 None.
    diagnostics (pd.DataFrame): row_position·fitted_mean·Pearson/deviance잔차·leverage·GLM Cook 근사.
    predictions (Optional[np.ndarray]): 선택 X_new의 조건부 평균; logistic은 성공확률.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 분포/링크/노출/추론·선택 한계.
```

```text
run_poisson(X, y, X_new=None, offset=None, new_offset=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    X_new: 전용 명세의 X_new 입력.
    offset: 전용 명세의 offset 입력.
    new_offset: 전용 명세의 new_offset 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    PoissonGLMOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

## glm-gamma

`src.MachineLearning.GeneralizedLinearModels.gamma`; JSON `glm-gamma.json`.

### GammaGLMProfile

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 계수·구간 유의수준 (0,1) 기본 0.05.
    maxiter (int): IRLS 최대 적합 반복, 양수 기본 100.
    cov_type (str): nonrobust/HC0 robust 공분산 기본 'nonrobust'.
```
### GammaGLMOutputDTO

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model (Any): 적합 statsmodels GLM; JSON 제외.
    coefficients (pd.DataFrame): term·coefficient·se·z·양측p·CI·exp(coef).
    fit_metrics (Dict[str,Any]): family·link·llf·AIC·BIC(평균계수 수 k)·deviance·Pearson dispersion·McFadden(null 비교) 또는 None.
    diagnostics (pd.DataFrame): row_position·fitted_mean·Pearson/deviance잔차·leverage·GLM Cook 근사.
    predictions (Optional[np.ndarray]): 선택 X_new의 조건부 평균; logistic은 성공확률.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 분포/링크/노출/추론·선택 한계.
```

```text
run_gamma(X, y, X_new=None, offset=None, new_offset=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    X_new: 전용 명세의 X_new 입력.
    offset: 전용 명세의 offset 입력.
    new_offset: 전용 명세의 new_offset 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    GammaGLMOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

## regression-weighted

`src.MachineLearning.LinearModels.weighted_regression`; JSON `regression-weighted.json`.

### WeightedRegressionProfile

```text
전용 설정의 전체 필드 명세.

Attributes:
    iterations (int): 학습 log(잔차²)의 분산 모형 재추정 반복, 1~5 기본 2.
    variance_floor (float): 양수 잔차² floor와 예측 분산 floor 기본 0.0001.
    weight_ratio (float): 평균정규화 전 가중치 최대/최소 비율 상한, 1 이상 기본 100.0.
    cov_type (str): 추정 가중치에 조건부 nonrobust/HC3 공분산 기본 'HC3'.
```
### WeightedRegressionOutputDTO

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model (Any): CV clone 가능한 적합 FeasibleWLSRegressor; JSON 제외.
    coefficients (pd.DataFrame): 조건부 WLS coefficient·se·t·p·CI.
    weights (np.ndarray): 학습 행 순서의 추정 역분산 가중치.
    weighted_residuals (np.ndarray): sqrt(weight)×원단위 잔차; 추정 분산 모형에 조건부.
    fit_metrics (Dict[str,Any]): 원 학습 y 단위 R²/RMSE 및 조건부 WLS AIC/BIC·weighted_R².
    predictions (Optional[np.ndarray]): X_new의 원 단위 평균 예측.
    parameters (Dict[str,Any]): 설정 전체.
    warnings (List[str]): 가중치 추정·추론·정보량 비교의 한계.
```

```text
build_weighted_regressor(profile=None, *, recipe_path=None, overrides=None)
구성만 수행하는 미적합 CV 가중 회귀 factory.

Args:
    profile: WeightedRegressionProfile 또는 None.
    recipe_path: 사용자 JSON 또는 None.
    overrides: Profile 생략 시 일회성 재정의.
Returns:
    FeasibleWLSRegressor: 전처리·분산 모형을 CV 학습 안에서 적합할 추정기.
Raises:
    ValueError/TypeError/OSError: 레시피·설정 오류.
```

```text
run_weighted_regression(X, y, X_new=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    X_new: 전용 명세의 X_new 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    WeightedRegressionOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

## timeseries-rolling-selection

`src.TimeSeries.rolling_selection`; JSON `timeseries-rolling-selection.json`.

### RollingSelectionProfile

```text
전용 설정의 전체 필드 명세.

Attributes:
    horizon (int): 마지막 최종평가 구간 길이, 2 이상 기본 7.
    validation_size (int): 각 rolling 검증 horizon, 2 이상 기본 7.
    folds (int): 과거 확장 학습 폴드 수, 2 이상 기본 3.
    period (int): 1은 ARIMA, 2 이상은 SARIMA (1,0,0,s)와 계절 naive 기본 1.
    candidate_orders (Optional[List[List[int]]]): 학습 rolling 비교 차수 목록; None이면 [0,0,0],[1,0,0],[2,0,0] 기본 None.
    maxiter (int): 후보 최대 적합 반복, 양수 기본 100.
    alpha (float): 선택 예측구간 유의수준 (0,1) 기본 0.05.
```
### RollingSelectionOutputDTO

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model (Any): 선택 ARIMA/SARIMA DTO; runtime JSON 제외.
    comparison (pd.DataFrame): 후보별 rolling RMSE·MAE·수렴 실패; 최종평가는 선택에 사용 안 함.
    folds (List[Dict[str,Any]]): 원행 위치의 학습/검증 연속 구간.
    selected_order (List[int]): 학습 rolling RMSE로 선택한 차수.
    train_indices (np.ndarray): 최종 적합 과거 원행 위치.
    test_indices (np.ndarray): 최종 미사용 미래 원행 위치.
    predictions (np.ndarray): 선택 모형 고정 horizon 평균 예측.
    baseline_predictions (np.ndarray): 마지막 값 또는 마지막 주기 반복 naive.
    test_metrics (Dict[str,Any]): 선택/naive의 원단위 RMSE·MAE·R² 및 MASE 학습분모.
    residual_diagnostics (Dict[str,Any]): 초기화 후 학습 잔차 Ljung-Box, p/q/P/Q 자유도 보정·AIC/BIC.
    parameters (Dict[str,Any]): Profile 전체.
    warnings (List[str]): 시간순·비교 척도·선택 후 구간 한계.
```

```text
run_rolling_selection(y, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    y: 전용 명세의 y 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    RollingSelectionOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

## plot-regression-model-comparison

`src.Visualization.regression.model_comparison`; JSON `plot-regression-model-comparison.json`.

### RegressionModelComparisonProfile

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목 기본 ''.
    color (str): 검증 점수 색 기본 'C0'.
    alpha (float): 표시 투명도 [0,1] 기본 0.8.
    metric (str): 동일 검증 행·원단위 점수 열 이름 기본 'cv_r2'.
```
### RegressionModelComparisonOutputDTO

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 Figure; runtime 제외.
    axes (Any): 동일 ax; runtime 제외.
    artists (List[Any]): 추가 artist; runtime 제외.
    data (pd.DataFrame): 입력 비교표 복사, 실제 점수 유지.
    diagnostics (Dict[str,Any]): 점수 이름·행 수; 선택/통계 검정을 대신하지 않음.
    parameters (Dict[str,Any]): Profile 전체.
    warnings (List[str]): 다른 응답 척도·분할 점수 혼용 금지.
```

```text
plot_model_comparison(comparison, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    comparison: 전용 명세의 comparison 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    RegressionModelComparisonOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

## plot-regression-glm-residuals

`src.Visualization.regression.glm_residuals`; JSON `plot-regression-glm-residuals.json`.

### GLMResidualsProfile

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목 기본 ''.
    color (str): 점 색상 기본 'C0'.
    alpha (float): 투명도 [0,1] 기본 0.6.
    residual (str): pearson/deviance 잔차 선택 기본 'pearson'.
```
### GLMResidualsOutputDTO

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 Figure; JSON 제외.
    axes (Any): 동일 ax; JSON 제외.
    artists (List[Any]): 추가 artist; JSON 제외.
    data (pd.DataFrame): 적합 GLM의 평균·선택 잔차·leverage·Cook·원행.
    diagnostics (Dict[str,Any]): family·link·선택 잔차·조건부 분산 요약.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 잔차 정규성의 자동 요구·영향값 자동 삭제 금지.
```

```text
plot_glm_residuals(result, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    result: 전용 명세의 result 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    GLMResidualsOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```
