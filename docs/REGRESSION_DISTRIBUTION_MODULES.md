# 회귀 추론·분포별 Naive Bayes·QQ 모듈

기존 7개 섹션·작성 정보·변경 이력·전체 Attributes/Args/Returns/Raises·JSON → Profile → runner → 전용 Output DTO 계약으로 **44개 별도 모듈**을 추가합니다. 기존 통합 naive_bayes.py, numeric/qq.py와 EDA 모듈을 유지합니다. 관측 데이터는 JSON에 넣지 않습니다.

## 회귀 절차와 해석

1. `regression_workflow`가 원행 위치로 학습/최종 평가를 먼저 분리합니다. random은 IID, chronological은 이미 시간순 정렬된 자료입니다. HAC는 chronological에서만 허용합니다.
2. 각 CV 학습 폴드에서 Feature 대치·변환·표준화·다항식·VIF 선택·목표 변환을 적합합니다. 사용자 FeatureSpec 전처리도 clone 가능한 sklearn Pipeline/ColumnTransformer로 전달합니다. 단순 표준화는 같은 열공간·절편 OLS의 R²를 높이지 않습니다.
3. 학습 CV의 원 목표 단위 점수만으로 후보를 선택합니다. 학습 R²가 커졌다는 이유로 다항식이나 로그 변환을 선택하지 않습니다. 최종 평가 자료는 선택에 사용하지 않으며 선택 후보와 기준 모형을 같은 행으로 비교합니다. 후보 적합 실패는 기록합니다.
4. `regression-ols`는 full-rank 설계·양의 잔차 자유도·유한 비상수 타깃을 확인합니다. 절편을 추가할 때 상수 입력 열을 거부합니다. 계수 t, 양측 p, CI, 전체 F, 모델 척도 R²/수정 R², AIC/BIC 및 새 행 평균 CI/관측 PI를 반환합니다. 절편 없는 R²는 uncentered로 구분합니다.
5. `joint_f`는 기본 전체 기울기 또는 사용자 Rβ=q를 검정합니다. 하나의 선형 대비에서는 F=t²입니다. `nested_f`는 같은 반응·같은 행 순서·실제 중첩 열공간·추가 자유도 및 고전적 공분산을 요구합니다. 두 개 임의 모형의 F 비교가 아닙니다.
6. 선형성 RESET/Rainbow, 잔차 정규성 JB/Shapiro/Dagostino/AD, 등분산 BP/Koenker/White, 순서 자기상관 DW/BG/Ljung-Box를 별도 파일에서 수행합니다. 정규성은 반응변수 자체가 아니라 모형 오차에 대한 가정입니다. 진단 비기각은 가정 성립의 증명이 아닙니다.
7. `independence`는 기본 `ordered=False`에서 DW 기술값만 제공하며 p/reject는 None입니다. `ordered=True`는 호출자가 실제 시간순·등간격을 확인한 경우입니다. 무작위 행 순서에 BG/LB를 적용하지 않습니다. 자기상관 부재가 모든 형태의 독립성을 증명하지 않습니다.
8. VIF=1/(1-Rj²)는 설계 열의 중복 정보, Cook은 관측의 계수 영향입니다. VIF 제거와 영향 관측 제거는 다른 과정입니다. 임계값 초과 관측을 결측·오류로 단정하지 않습니다.
9. Cook 기본은 **flag**입니다. 명시 `action='drop'`인 경우만 학습 행에서 제거하고 최대 제거비율·잔여 rank·자유도를 확인합니다. workflow는 유지 원행에서 전처리·목표 변환·VIF를 모두 다시 적합하며 CV 내부에도 같은 절차를 적용합니다. 최종 평가 행을 제거하지 않습니다. 원행 대응과 제거·Feature 선택 이력을 skeleton에서 보고합니다.

고전적 소표본 t/F·CI는 올바른 선형 사양·정규·독립·등분산 오차 가정에 조건부입니다. HC0–3/HAC t·Wald F는 소표본 exact 검정이 아니며 robust 공분산의 관측 PI도 일정 분산 근사입니다. 변수·변환·관측 선택 후 p값과 구간은 선택 과정을 보정하지 않습니다. 설명·예측·인과를 구분합니다.

목표 Yeo-Johnson/Box-Cox의 λ는 학습에서만 적합합니다. log/log1p는 정의역을 검사하고 역변환은 일반적으로 E[Y|X]가 아닙니다. 선택적 Duan smearing은 학습 변환 잔차로만 보정합니다. 변환 반응의 학습 R²와 원 단위 보류 예측 R²를 직접 같은 값처럼 비교하지 않습니다. 전처리가 성능 향상을 보장하지 않습니다.

## 실행 예

```python
from src.MachineLearning.LinearModels.regression_workflow import run_regression_workflow
from src.stats.regression.normality import run_normality
from src.stats.regression.influence import run_influence

result = run_regression_workflow(X, y)
print(result.selected_name, result.comparison)
print(result.selected_metrics)
normality = run_normality(result.inference)
influence = run_influence(result.inference)
print(result.to_json())
```

실행 가능한 전체 조합은 `python -m examples.regression_distribution_examples`입니다. 실제 UCI Wine 11개 특성의 전체 절차는 `src.MachineLearning.LinearModels.skeleton.regression_workflow.main(cache_dir=..., draw=False)`이며 시각화는 `draw=True` 또는 기존 axes 배열 전달로 조합합니다. 기본 예제 한 실행에서 CV 선택 feature_power의 보류 RMSE는 약0.6173, 기준0.6260이었으나 다른 자료의 개선을 보장하지 않습니다. Wine quality는 이산 순서 점수이므로 OLS 연속 반응 근사의 한계를 확인합니다.

## 별도 전처리와 시각화

`Preprocessing/regression`의 polynomial·target_transform·vif_selection·influence_filter는 Feature 기반 전처리 및 목표/행 처리 책임을 구분합니다. 다항식은 bias 없이 확장하고 생성 열 수 한도를 먼저 검사합니다. 보호 열·최소 열 수·최대 VIF 제거 단계·상수 열 처리를 명시할 수 있습니다.

회귀 Axes 모듈은 residual_fitted, scale_location, residual_order, cooks_distance, leverage, coefficient_intervals, observed_predicted, prediction_interval, partial_residual, vif입니다. **기존 ax가 필수**이고 figure 생성·show·save·전역 스타일 변경이 없습니다. OLS 잔차·계수·부분잔차·PI는 적합한 반응 좌표계이며 observed_predicted만 원 단위 y와 원 단위 예측을 짝지어 전달할 수 있습니다. 평균 CI와 새 관측 PI를 구분합니다. Cook contour는 leverage와 내부 studentized residual 좌표에서 계산합니다. VIF 표시 cap은 artist만 제한하고 실제 무한대 값은 보존합니다.

## 분포별 QQ

normal·student_t·exponential·gamma·lognormal·weibull·beta·uniform·chi_square·logistic·poisson·binomial·bernoulli 13개 파일/레시피입니다. p_i=(i+0.5)/n의 이론 분위수와 정렬 관측값을 비교하고 reference=identity/quartile/none을 지원합니다. 기본은 알려진 모수이며 `fit_parameters=True`면 표본에서 추정합니다. 양수 분포는 고정 loc, beta는 고정 loc/scale 아래 모양 추정 등 각 파일의 모수 계약을 따릅니다. support 위반은 보고하며 추정 불가능 자료는 거부합니다. 이산 분포는 정수 성공/횟수와 지지 범위를 구분합니다. 분위수 동률로 사분위 기준선이 정의되지 않으면 생략하고 경고합니다. 같은 표본의 적합 QQ는 정식 독립 적합도 검정이나 p값이 아닙니다.

## 분포별 Naive Bayes

| 모듈 | Feature 정의역·우도 | 구현 |
| --- | --- | --- |
| gaussian | 연속값, 클래스별 정규 평균/분산 | sklearn GaussianNB, var_smoothing |
| bernoulli | 0/1 또는 명시 binarize 임계값, 성공/실패 모두 반영 | sklearn BernoulliNB |
| multinomial | 비음수 빈도·가중 빈도, 고정 문서/관측 내 상대 구성 | sklearn MultinomialNB |
| complement | 비음수 빈도, 다른 클래스 보완 통계 | sklearn ComplementNB |
| poisson | 비음수 정수 횟수, 관측 노출시간 동일 | 별도 PoissonNB, Gamma posterior-mean rate |
| binomial | 알려진 공통 n_trials의 0..n 정수 성공 수 | 별도 BinomialNB, Beta posterior-mean p |
| categorical | 명목 Feature 원 라벨, 범주별 확률 | 별도 CategoricalNB, 학습 vocabulary·미지/결측 버킷 |

Poisson rate=(클래스별 합+rate_alpha)/(클래스 표본수+rate_beta), Binomial p=(성공 합+alpha)/(클래스 표본수×n_trials+alpha+beta)입니다. 분포의 plug-in log PMF+클래스 prior를 logsumexp로 정규화하며 베이즈 예측분포를 적분한 것으로 주장하지 않습니다. Binomial은 성공·실패 로그 질량을 따로 계산해 작은 양수 prior에서 p가1로 반올림되어 생기는 0×log0 오류를 피합니다. n_trials=1, 대칭 prior는 Bernoulli와 대응합니다. Poisson과 Binomial을 단순 빈도 다항모형으로 대신하지 않습니다.

Categorical은 unknown=0,missing=1,관측 범주=2...를 분리하고 새 자료로 vocabulary를 적합하지 않습니다. 숫자 범주 코드의 크기를 연속 거리로 해석하지 않습니다. 반환 Pipeline은 새 Feature 이름·순서를 확인합니다. 오직 Gaussian에서만 표준화/minmax를 허용합니다. smoothing 후보는 양수입니다. 최종 test_indices를 명시해 공식 UCI train/test를 보존할 수 있고, 튜닝·전처리는 학습 CV 안에서만 수행합니다. 조건부 독립 가정과 모델 우도의 적합성은 별도 검토 대상이며 posterior를 보정된 실제 확률로 보장하지 않습니다.

## 실제 자료 skeleton과 검증 경계

분야별 추가 13개 프로그램: 회귀 workflow·통계 진단·순서 진단·전처리·회귀 그림·QQ board·분포 NB7입니다. 기존 UCI 모델 skeleton46개와 EDA board11개를 덮어쓰지 않습니다.

- UCI Wine: 원11개 특성; 회귀 설계·비순서 진단·전처리/QQ.
- UCI Bike Sharing: 일별 규칙적 시간순 예제로 HAC/BG/LB; Poisson은 같은 일 노출의 count로 달력 주말을 설명하는 분류 연습이며 미래 수요예측이나 Poisson 분포 적합성의 증거가 아닙니다.
- UCI Iris: Gaussian 분류.
- UCI Optdigits: 공식 train3,823/test1,797 유지. 원32×32 bitmap의4×4블록에서 켜진 픽셀 수0..16이므로 Binomial n=16을 명시합니다. Bernoulli는 고정 >0 존재 변환, Multinomial/Complement는 원 비음수 빈도입니다.
- UCI Adult: 공식 train/test의 앞1,000행씩, 원 범주6개와 미지/결측 정책입니다. 전체 Adult 성능으로 보고하지 않습니다.

단위검사는 statsmodels/SciPy/sklearn 참조값, F=t², 수동 Cook 투영식·VIF, rank·행 대응·정의역, 실제 CV 학습 폴드의 대치 통계와 재적합, QQ ppf, 축 소유권, 전용 명세를 비교합니다. 실제 UCI 다운로드 예제는 CI의 오프라인 단위검사와 구분합니다.

기술 근거: [statsmodels 진단](https://www.statsmodels.org/stable/diagnostic.html), [OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html), [sklearn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html), [SciPy 확률 그림](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html), [UCI Optdigits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

## 기법별 전체 입력·출력 명세

### regression-ols

`src.stats.regression.ols`; 기본 JSON `regression-ols.json`.

**OLSProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
    fit_intercept (bool): 절편을 추가; 입력 상수 Feature는 거부. 기본 True.
    cov_type (str): nonrobust/HC0/HC1/HC2/HC3/HAC 공분산. 기본 'nonrobust'.
    hac_maxlags (int): HAC의 비음수 최대 지연; 행 순서/간격 책임은 호출자. 기본 1.
```

**OLSOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model (Any): 적합 statsmodels OLS; runtime JSON 제외.
    coefficients (pd.DataFrame): term/coefficient/standard_error/t_statistic/p_value/ci_lower/ci_upper/reject_h0.
    fit_metrics (Dict[str,Any]): 학습 R²·수정 R²·F·p·자유도·AIC/BIC; 절편 없는 R²는 uncentered.
    design (pd.DataFrame): 적합 행 순서의 절편 포함 설계 행렬.
    fitted_values (np.ndarray): 학습 적합값; 일반화 평가 아님.
    residuals (np.ndarray): y-fitted; 모델 반응변수 좌표계.
    new_predictions (Optional[pd.DataFrame]): 새 행의 mean·평균 CI·관측 PI; 목표 변환 시 그 좌표계.
    parameters (Dict[str,Any]): 적용 설정.
    warnings (List[str]): 추론 가정·robust/선택 후 해석 정보.
```

```text
run_ols(X, y, X_new=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    X_new: 전용 명세의 X_new 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    OLSOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-joint-f

`src.stats.regression.joint_f`; 기본 JSON `regression-joint-f.json`.

**JointFProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
```

**JointFOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
    details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 검정 가정·선택 후 추론 정보.
```

```text
run_joint_f(model, contrasts=None, rhs=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    contrasts: 전용 명세의 contrasts 입력.
    rhs: 전용 명세의 rhs 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    JointFOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-nested-f

`src.stats.regression.nested_f`; 기본 JSON `regression-nested-f.json`.

**NestedFProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
```

**NestedFOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
    details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 검정 가정·선택 후 추론 정보.
```

```text
run_nested_f(restricted, full, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    restricted: 전용 명세의 restricted 입력.
    full: 전용 명세의 full 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    NestedFOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-linearity

`src.stats.regression.linearity`; 기본 JSON `regression-linearity.json`.

**LinearityProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
    method (str): reset/rainbow 선형 사양 진단. 기본 'reset'.
    power (int): RESET 추가 fitted 거듭제곱 최대 차수, 2 이상. 기본 3.
    fraction (float): Rainbow 중심 부분 표본 비율 (0,1). 기본 0.5.
```

**LinearityOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
    details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 검정 가정·선택 후 추론 정보.
```

```text
run_linearity(model, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    LinearityOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-normality

`src.stats.regression.normality`; 기본 JSON `regression-normality.json`.

**RegressionNormalityProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
    method (str): jarque_bera/shapiro/dagostino/anderson 잔차 정규성. 기본 'jarque_bera'.
```

**RegressionNormalityOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
    details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 검정 가정·선택 후 추론 정보.
```

```text
run_normality(model, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    RegressionNormalityOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-heteroscedasticity

`src.stats.regression.heteroscedasticity`; 기본 JSON `regression-heteroscedasticity.json`.

**HeteroscedasticityProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
    method (str): breusch_pagan/white. 기본 'breusch_pagan'.
    koenker (bool): BP robust=True의 Koenker 버전; 정규성 완화. 기본 True.
```

**HeteroscedasticityOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
    details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 검정 가정·선택 후 추론 정보.
```

```text
run_heteroscedasticity(model, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    HeteroscedasticityOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-independence

`src.stats.regression.independence`; 기본 JSON `regression-independence.json`.

**IndependenceProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
    ordered (bool): True는 시간 순·동일 간격 자료임을 호출자가 명시. 기본 False.
    method (str): breusch_godfrey/ljung_box/durbin_watson. 기본 'breusch_godfrey'.
    nlags (int): 양의 잔차 지연 수. 기본 1.
```

**IndependenceOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
    details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 검정 가정·선택 후 추론 정보.
```

```text
run_independence(model, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    IndependenceOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-collinearity

`src.stats.regression.collinearity`; 기본 JSON `regression-collinearity.json`.

**CollinearityProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    vif_threshold (float): 1 초과 VIF 검토 기준; 자동 삭제 아님. 기본 10.0.
```

**CollinearityOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    table (pd.DataFrame): feature/vif/tolerance/flag; VIF는 검정 p가 아님.
    condition_number (float): 중심화·표준화 Feature 설계의 조건수; 상수 제외.
    rank (int): 절편 포함 원 설계 rank.
    parameters (Dict[str,Any]): 적용 설정.
    warnings (List[str]): 공선성·단위·구조적 rank 정보.
```

```text
run_collinearity(X, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    CollinearityOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-influence

`src.stats.regression.influence`; 기본 JSON `regression-influence.json`.

**InfluenceProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
    cook_threshold (Optional[float]): None이면 4/n 검토 기준. 기본 None.
    leverage_multiplier (float): 평균 leverage p/n에 곱할 양수 배율. 기본 2.0.
    studentized_threshold (float): 외적 studentized 잔차의 양수 절댓값 기준. 기본 3.0.
    p_adjust (str): outlier p의 bonferroni/holm/fdr_bh/none. 기본 'bonferroni'.
    max_samples (int): leave-one-out 영향 계산 표본 한도. 기본 3000.
```

**InfluenceOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    table (pd.DataFrame): position/row_label/residual/studentized_internal/studentized_external/leverage/cooks_distance/dffits/outlier_p/adjusted_outlier_p 및 flag.
    dfbetas (pd.DataFrame): 원 행×설계 항의 계수 변화 표준화 지표.
    thresholds (Dict[str,float]): Cook·leverage·studentized·DFFITS 검토 기준.
    parameters (Dict[str,Any]): 적용 설정.
    warnings (List[str]): 영향·오류/결측의 구별과 선택 편향.
```

```text
run_influence(model, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    InfluenceOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-polynomial

`src.Preprocessing.regression.polynomial`; 기본 JSON `regression-polynomial.json`.

**PolynomialProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    degree (int): 다항식 최대 차수, 1~3. 기본 2.
    interaction_only (bool): 상호작용만 생성; 절편은 OLS가 추가. 기본 False.
    max_features (int): 생성 Feature 수 제한. 기본 500.
```

**PolynomialOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    preprocessed_feature (Any): 학습 Feature/타깃 변환.
    new_preprocessed_feature (Any): 새 자료 transform 또는 None.
    input_features (List[str]): 원 열 순서.
    output_features (List[str]): 생성/선택한 열 순서.
    parameters (Dict[str,Any]): 적용 설정.
    transformer (Any): 학습 자료에만 fit한 재사용 변환기; JSON 제외.
    n_samples (int): 학습 행 수.
    warnings (List[str]): 추론·선택 해석 정보.
```

```text
build_polynomial_transformer(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    PolynomialDesignTransformer: 미적합 sklearn clone 호환 변환기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_polynomial(X_train, X_new=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X_train: 전용 명세의 X_train 입력.
    X_new: 전용 명세의 X_new 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    PolynomialOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-target-transform

`src.Preprocessing.regression.target_transform`; 기본 JSON `regression-target-transform.json`.

**TargetTransformProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    method (str): none/log/log1p/yeo-johnson/box-cox 반응변수 변환. 기본 'none'.
```

**TargetTransformOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    transformed_target (np.ndarray): 학습 타깃의 모델 좌표계.
    new_transformed_target (Optional[np.ndarray]): 새 타깃 transform 또는 None.
    transformer (Any): 학습 전용 변환기; JSON 제외.
    parameters (Dict[str,Any]): 적용 설정.
    warnings (List[str]): inverse 예측 평균과 중앙값·추론 좌표계 구별.
```

```text
build_target_transformer(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    RegressionTargetTransformer: 미적합 타깃 변환기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_target_transform(y_train, y_new=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    y_train: 전용 명세의 y_train 입력.
    y_new: 전용 명세의 y_new 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    TargetTransformOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-vif-selection

`src.Preprocessing.regression.vif_selection`; 기본 JSON `regression-vif-selection.json`.

**VIFSelectionProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    threshold (Optional[float]): VIF 제거 기준 1 초과; None은 상수만 제거. 기본 10.0.
    drop_constant (bool): 상수 Feature 제거 여부. 기본 True.
    max_steps (int): 순차 제거 단계 한도. 기본 100.
    min_features (int): 최소 남길 Feature 수. 기본 1.
    preserve_columns (Optional[List[str]]): 삭제 금지 원/변환 Feature 이름. 기본 None.
```

**VIFSelectionOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    preprocessed_feature (Any): 학습 Feature/타깃 변환.
    new_preprocessed_feature (Any): 새 자료 transform 또는 None.
    input_features (List[str]): 원 열 순서.
    output_features (List[str]): 생성/선택한 열 순서.
    parameters (Dict[str,Any]): 적용 설정.
    transformer (Any): 학습 자료에만 fit한 재사용 변환기; JSON 제외.
    n_samples (int): 학습 행 수.
    warnings (List[str]): 추론·선택 해석 정보.
    selection_history (pd.DataFrame): 순차 제거 순서와 VIF/상수 이유.
```

```text
build_vif_selector(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    VIFSelector: 학습 Feature만 선택하는 clone 호환 변환기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_vif_selection(X_train, X_new=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X_train: 전용 명세의 X_train 입력.
    X_new: 전용 명세의 X_new 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    VIFSelectionOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-influence-filter

`src.Preprocessing.regression.influence_filter`; 기본 JSON `regression-influence-filter.json`.

**InfluenceFilterProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    action (str): flag/drop; 기본은 원 자료 유지. 기본 'flag'.
    cook_threshold (Optional[float]): None이면 4/n; 양수 검토 기준. 기본 None.
    max_remove_fraction (float): drop 시 최대 제거 비율 [0,1). 기본 0.1.
```

**InfluenceFilterOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    cleaned_features (pd.DataFrame): 동일 Feature 순서의 유지 학습 자료.
    cleaned_target (np.ndarray): 유지한 대응 학습 타깃.
    kept_positions (np.ndarray): 원 학습 입력 내 유지 행 위치.
    removed_positions (np.ndarray): 명시적 drop으로 제거한 학습 행 위치.
    review_table (pd.DataFrame): position/row_label/cooks_distance/flag/removed.
    parameters (Dict[str,Any]): 적용 설정.
    warnings (List[str]): 관측 삭제의 근거·선택 편향·결측과의 구별.
```

```text
run_influence_filter(X_train, y_train, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X_train: 전용 명세의 X_train 입력.
    y_train: 전용 명세의 y_train 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    InfluenceFilterOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### regression-workflow

`src.MachineLearning.LinearModels.regression_workflow`; 기본 JSON `regression-workflow.json`.

**RegressionWorkflowProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    test_size (float): 최종 보류 비율 (0,1); 선택에 사용하지 않음. 기본 0.3.
    random_state (int): IID 분할·CV seed. 기본 1234.
    cv (int): 학습 구간 선택 CV 폴드 수, 2 이상. 기본 3.
    split_strategy (str): random/chronological; 후자는 입력 순서 보존. 기본 'random'.
    scoring (str): 원 타깃 단위 CV 지표; neg_root_mean_squared_error/r2/neg_mean_absolute_error. 기본 'neg_root_mean_squared_error'.
    target_missing (str): raise/drop; drop 시 미관측 타깃 행 위치 기록. 기본 'raise'.
    alpha (float): 선택 모형의 조건부 추론 alpha. 기본 0.05.
    cov_type (str): 선택 모형의 nonrobust/HC0/HC1/HC2/HC3/HAC. 기본 'nonrobust'.
    hac_maxlags (int): HAC 최대 지연; chronological에서만 허용. 기본 1.
    candidates (Optional[List[Dict[str,Any]]]): name과 RegressionEstimator 옵션의 명시 후보 목록; None은 기본 5개. 기본 None.
```

**RegressionWorkflowOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model (Any): CV로 선택해 전체 train에서 재적합한 RegressionEstimator; JSON 제외.
    baseline_model (Any): 처리 전 기준 모형; runtime JSON 제외.
    comparison (pd.DataFrame): 후보별 CV 평균/표준편차·학습 지표·fit 실패 오류; test 지표는 선택 후 두 모형만 평가.
    selected_name (str): CV 점수가 가장 큰 후보 이름.
    train_indices (np.ndarray): 원 입력의 train 위치.
    test_indices (np.ndarray): 원 입력의 최종 test 위치.
    excluded_target_positions (np.ndarray): 명시 target_missing=drop으로 제외한 원 위치.
    baseline_metrics (Dict[str,Any]): 기준 모형 원 y 단위 train/test RMSE·MAE·R² 및 fit R²/수정 R².
    selected_metrics (Dict[str,Any]): 선택 모형의 같은 정의 지표.
    predictions (np.ndarray): 원 y 단위 최종 test 예측.
    y_test (np.ndarray): 원 단위 test 참값.
    inference (Any): 최종 선택 설계의 OLS DTO; 그 모형 좌표계 추론.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 선택 후 추론·R² 비교·타깃 변환·삭제 해석.
```

```text
build_regression_estimator(preprocessor=None, **options)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    preprocessor: 전용 명세의 preprocessor 입력.
    **options: 전용 명세의 **options 입력.
Returns:
    RegressionEstimator: 미적합 sklearn/CV 호환 회귀기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_regression_workflow(X, y, preprocessor=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    preprocessor: 전용 명세의 preprocessor 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    RegressionWorkflowOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### naive-bayes-gaussian

`src.MachineLearning.NaiveBayes.gaussian`; 기본 JSON `naive-bayes-gaussian.json`.

**GaussianProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    test_size (float): 최종 보류 비율; fixed test_indices가 우선. 기본 0.3.
    random_state (int): 분할·CV seed. 기본 1234.
    stratify (bool): random 분할의 클래스 계층화. 기본 True.
    split_strategy (str): random/chronological; 후자는 입력 행 순서. 기본 'random'.
    scaling (str): Gaussian만 none/standard/minmax; 나머지는 none. 기본 'none'.
    cv (int): 선택용 학습 CV 폴드, 2 이상. 기본 3.
    scoring (Optional[str]): GridSearchCV 분류 지표; None은 accuracy. 기본 None.
    n_jobs (int): CV 병렬 작업 수; 0 불가. 기본 1.
    param_grid (Optional[Dict[str,Any]]): 분포별 estimator 인자명 또는 model__인자명 탐색. 기본 None.
    var_smoothing (float): 분산 안정화 양수 비율. 기본 1e-09.
```

**GaussianOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model_name (str): 적합 분포별 클래스 이름.
    model (Any): 전처리 포함 Pipeline; JSON 제외.
    train_metrics (Dict[str,float]): 학습 accuracy·weighted precision/recall/F1.
    test_metrics (Dict[str,float]): 보류 같은 지표 및 이진 ROC AUC.
    y_test (np.ndarray): 보류 원 라벨.
    predictions (np.ndarray): 보류 예측 원 라벨.
    probabilities (np.ndarray): classes 열 순서 posterior.
    classes (np.ndarray): 학습 클래스 순서.
    class_counts (np.ndarray): 학습 클래스 빈도; 전체 자료 빈도 아님.
    confusion (np.ndarray): classes 순서의 참값×예측 혼동행렬.
    train_indices (np.ndarray): 원 입력 학습 행 위치.
    test_indices (np.ndarray): 원 입력 보류 행 위치.
    feature_names (List[str]): 원 Feature 순서.
    best_params (Dict[str,Any]): 학습 CV 선택 파라미터.
    best_cv_score (Optional[float]): 최고 CV 점수 또는 None.
    cv_results (Optional[Dict[str,Any]]): 전체 CV 결과 또는 None.
    warnings (List[str]): 분포·독립·확률 해석 가정.
```

```text
build_gaussian_classifier(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    GaussianNB: 미적합 분포별 분류기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_gaussian(X, y, test_indices=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    test_indices: 전용 명세의 test_indices 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    GaussianOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### naive-bayes-bernoulli

`src.MachineLearning.NaiveBayes.bernoulli`; 기본 JSON `naive-bayes-bernoulli.json`.

**BernoulliProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    test_size (float): 최종 보류 비율; fixed test_indices가 우선. 기본 0.3.
    random_state (int): 분할·CV seed. 기본 1234.
    stratify (bool): random 분할의 클래스 계층화. 기본 True.
    split_strategy (str): random/chronological; 후자는 입력 행 순서. 기본 'random'.
    scaling (str): Gaussian만 none/standard/minmax; 나머지는 none. 기본 'none'.
    cv (int): 선택용 학습 CV 폴드, 2 이상. 기본 3.
    scoring (Optional[str]): GridSearchCV 분류 지표; None은 accuracy. 기본 None.
    n_jobs (int): CV 병렬 작업 수; 0 불가. 기본 1.
    param_grid (Optional[Dict[str,Any]]): 분포별 estimator 인자명 또는 model__인자명 탐색. 기본 None.
    alpha (float): 양수 Bernoulli 가산 평활화. 기본 1.0.
    binarize (Optional[float]): None이면 엄격 0/1; 유한 수면 고정 임계값으로 이진화. 기본 None.
```

**BernoulliOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model_name (str): 적합 분포별 클래스 이름.
    model (Any): 전처리 포함 Pipeline; JSON 제외.
    train_metrics (Dict[str,float]): 학습 accuracy·weighted precision/recall/F1.
    test_metrics (Dict[str,float]): 보류 같은 지표 및 이진 ROC AUC.
    y_test (np.ndarray): 보류 원 라벨.
    predictions (np.ndarray): 보류 예측 원 라벨.
    probabilities (np.ndarray): classes 열 순서 posterior.
    classes (np.ndarray): 학습 클래스 순서.
    class_counts (np.ndarray): 학습 클래스 빈도; 전체 자료 빈도 아님.
    confusion (np.ndarray): classes 순서의 참값×예측 혼동행렬.
    train_indices (np.ndarray): 원 입력 학습 행 위치.
    test_indices (np.ndarray): 원 입력 보류 행 위치.
    feature_names (List[str]): 원 Feature 순서.
    best_params (Dict[str,Any]): 학습 CV 선택 파라미터.
    best_cv_score (Optional[float]): 최고 CV 점수 또는 None.
    cv_results (Optional[Dict[str,Any]]): 전체 CV 결과 또는 None.
    warnings (List[str]): 분포·독립·확률 해석 가정.
```

```text
build_bernoulli_classifier(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    BernoulliNB: 미적합 분포별 분류기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_bernoulli(X, y, test_indices=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    test_indices: 전용 명세의 test_indices 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    BernoulliOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### naive-bayes-multinomial

`src.MachineLearning.NaiveBayes.multinomial`; 기본 JSON `naive-bayes-multinomial.json`.

**MultinomialProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    test_size (float): 최종 보류 비율; fixed test_indices가 우선. 기본 0.3.
    random_state (int): 분할·CV seed. 기본 1234.
    stratify (bool): random 분할의 클래스 계층화. 기본 True.
    split_strategy (str): random/chronological; 후자는 입력 행 순서. 기본 'random'.
    scaling (str): Gaussian만 none/standard/minmax; 나머지는 none. 기본 'none'.
    cv (int): 선택용 학습 CV 폴드, 2 이상. 기본 3.
    scoring (Optional[str]): GridSearchCV 분류 지표; None은 accuracy. 기본 None.
    n_jobs (int): CV 병렬 작업 수; 0 불가. 기본 1.
    param_grid (Optional[Dict[str,Any]]): 분포별 estimator 인자명 또는 model__인자명 탐색. 기본 None.
    alpha (float): 양수 빈도 가산 평활화. 기본 1.0.
```

**MultinomialOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model_name (str): 적합 분포별 클래스 이름.
    model (Any): 전처리 포함 Pipeline; JSON 제외.
    train_metrics (Dict[str,float]): 학습 accuracy·weighted precision/recall/F1.
    test_metrics (Dict[str,float]): 보류 같은 지표 및 이진 ROC AUC.
    y_test (np.ndarray): 보류 원 라벨.
    predictions (np.ndarray): 보류 예측 원 라벨.
    probabilities (np.ndarray): classes 열 순서 posterior.
    classes (np.ndarray): 학습 클래스 순서.
    class_counts (np.ndarray): 학습 클래스 빈도; 전체 자료 빈도 아님.
    confusion (np.ndarray): classes 순서의 참값×예측 혼동행렬.
    train_indices (np.ndarray): 원 입력 학습 행 위치.
    test_indices (np.ndarray): 원 입력 보류 행 위치.
    feature_names (List[str]): 원 Feature 순서.
    best_params (Dict[str,Any]): 학습 CV 선택 파라미터.
    best_cv_score (Optional[float]): 최고 CV 점수 또는 None.
    cv_results (Optional[Dict[str,Any]]): 전체 CV 결과 또는 None.
    warnings (List[str]): 분포·독립·확률 해석 가정.
```

```text
build_multinomial_classifier(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    MultinomialNB: 미적합 분포별 분류기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_multinomial(X, y, test_indices=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    test_indices: 전용 명세의 test_indices 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    MultinomialOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### naive-bayes-complement

`src.MachineLearning.NaiveBayes.complement`; 기본 JSON `naive-bayes-complement.json`.

**ComplementProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    test_size (float): 최종 보류 비율; fixed test_indices가 우선. 기본 0.3.
    random_state (int): 분할·CV seed. 기본 1234.
    stratify (bool): random 분할의 클래스 계층화. 기본 True.
    split_strategy (str): random/chronological; 후자는 입력 행 순서. 기본 'random'.
    scaling (str): Gaussian만 none/standard/minmax; 나머지는 none. 기본 'none'.
    cv (int): 선택용 학습 CV 폴드, 2 이상. 기본 3.
    scoring (Optional[str]): GridSearchCV 분류 지표; None은 accuracy. 기본 None.
    n_jobs (int): CV 병렬 작업 수; 0 불가. 기본 1.
    param_grid (Optional[Dict[str,Any]]): 분포별 estimator 인자명 또는 model__인자명 탐색. 기본 None.
    alpha (float): 양수 보완 클래스 평활화. 기본 1.0.
    norm (bool): Complement 가중치의 두 번째 정규화. 기본 False.
```

**ComplementOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model_name (str): 적합 분포별 클래스 이름.
    model (Any): 전처리 포함 Pipeline; JSON 제외.
    train_metrics (Dict[str,float]): 학습 accuracy·weighted precision/recall/F1.
    test_metrics (Dict[str,float]): 보류 같은 지표 및 이진 ROC AUC.
    y_test (np.ndarray): 보류 원 라벨.
    predictions (np.ndarray): 보류 예측 원 라벨.
    probabilities (np.ndarray): classes 열 순서 posterior.
    classes (np.ndarray): 학습 클래스 순서.
    class_counts (np.ndarray): 학습 클래스 빈도; 전체 자료 빈도 아님.
    confusion (np.ndarray): classes 순서의 참값×예측 혼동행렬.
    train_indices (np.ndarray): 원 입력 학습 행 위치.
    test_indices (np.ndarray): 원 입력 보류 행 위치.
    feature_names (List[str]): 원 Feature 순서.
    best_params (Dict[str,Any]): 학습 CV 선택 파라미터.
    best_cv_score (Optional[float]): 최고 CV 점수 또는 None.
    cv_results (Optional[Dict[str,Any]]): 전체 CV 결과 또는 None.
    warnings (List[str]): 분포·독립·확률 해석 가정.
```

```text
build_complement_classifier(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    ComplementNB: 미적합 분포별 분류기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_complement(X, y, test_indices=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    test_indices: 전용 명세의 test_indices 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    ComplementOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### naive-bayes-poisson

`src.MachineLearning.NaiveBayes.poisson`; 기본 JSON `naive-bayes-poisson.json`.

**PoissonProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    test_size (float): 최종 보류 비율; fixed test_indices가 우선. 기본 0.3.
    random_state (int): 분할·CV seed. 기본 1234.
    stratify (bool): random 분할의 클래스 계층화. 기본 True.
    split_strategy (str): random/chronological; 후자는 입력 행 순서. 기본 'random'.
    scaling (str): Gaussian만 none/standard/minmax; 나머지는 none. 기본 'none'.
    cv (int): 선택용 학습 CV 폴드, 2 이상. 기본 3.
    scoring (Optional[str]): GridSearchCV 분류 지표; None은 accuracy. 기본 None.
    n_jobs (int): CV 병렬 작업 수; 0 불가. 기본 1.
    param_grid (Optional[Dict[str,Any]]): 분포별 estimator 인자명 또는 model__인자명 탐색. 기본 None.
    rate_alpha (float): Gamma 양수 shape; lambda posterior 평균의 pseudo count. 기본 1.0.
    rate_beta (float): Gamma 양수 rate; 클래스 표본 수에 더하는 노출 prior. 기본 1.0.
    fit_prior (bool): 학습 클래스 빈도 prior; False면 균등. 기본 True.
```

**PoissonOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model_name (str): 적합 분포별 클래스 이름.
    model (Any): 전처리 포함 Pipeline; JSON 제외.
    train_metrics (Dict[str,float]): 학습 accuracy·weighted precision/recall/F1.
    test_metrics (Dict[str,float]): 보류 같은 지표 및 이진 ROC AUC.
    y_test (np.ndarray): 보류 원 라벨.
    predictions (np.ndarray): 보류 예측 원 라벨.
    probabilities (np.ndarray): classes 열 순서 posterior.
    classes (np.ndarray): 학습 클래스 순서.
    class_counts (np.ndarray): 학습 클래스 빈도; 전체 자료 빈도 아님.
    confusion (np.ndarray): classes 순서의 참값×예측 혼동행렬.
    train_indices (np.ndarray): 원 입력 학습 행 위치.
    test_indices (np.ndarray): 원 입력 보류 행 위치.
    feature_names (List[str]): 원 Feature 순서.
    best_params (Dict[str,Any]): 학습 CV 선택 파라미터.
    best_cv_score (Optional[float]): 최고 CV 점수 또는 None.
    cv_results (Optional[Dict[str,Any]]): 전체 CV 결과 또는 None.
    warnings (List[str]): 분포·독립·확률 해석 가정.
```

```text
build_poisson_classifier(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    PoissonNB: 미적합 정수 계수 분류기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_poisson(X, y, test_indices=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    test_indices: 전용 명세의 test_indices 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    PoissonOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### naive-bayes-binomial

`src.MachineLearning.NaiveBayes.binomial`; 기본 JSON `naive-bayes-binomial.json`.

**BinomialProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    test_size (float): 최종 보류 비율; fixed test_indices가 우선. 기본 0.3.
    random_state (int): 분할·CV seed. 기본 1234.
    stratify (bool): random 분할의 클래스 계층화. 기본 True.
    split_strategy (str): random/chronological; 후자는 입력 행 순서. 기본 'random'.
    scaling (str): Gaussian만 none/standard/minmax; 나머지는 none. 기본 'none'.
    cv (int): 선택용 학습 CV 폴드, 2 이상. 기본 3.
    scoring (Optional[str]): GridSearchCV 분류 지표; None은 accuracy. 기본 None.
    n_jobs (int): CV 병렬 작업 수; 0 불가. 기본 1.
    param_grid (Optional[Dict[str,Any]]): 분포별 estimator 인자명 또는 model__인자명 탐색. 기본 None.
    n_trials (int): 모든 Feature/관측에 알려진 동일 시행 수; 양수 정수. 기본 1.
    alpha (float): 양수 Beta 성공 pseudo count. 기본 1.0.
    beta (float): 양수 Beta 실패 pseudo count. 기본 1.0.
    fit_prior (bool): 경험 클래스 prior 여부. 기본 True.
```

**BinomialOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model_name (str): 적합 분포별 클래스 이름.
    model (Any): 전처리 포함 Pipeline; JSON 제외.
    train_metrics (Dict[str,float]): 학습 accuracy·weighted precision/recall/F1.
    test_metrics (Dict[str,float]): 보류 같은 지표 및 이진 ROC AUC.
    y_test (np.ndarray): 보류 원 라벨.
    predictions (np.ndarray): 보류 예측 원 라벨.
    probabilities (np.ndarray): classes 열 순서 posterior.
    classes (np.ndarray): 학습 클래스 순서.
    class_counts (np.ndarray): 학습 클래스 빈도; 전체 자료 빈도 아님.
    confusion (np.ndarray): classes 순서의 참값×예측 혼동행렬.
    train_indices (np.ndarray): 원 입력 학습 행 위치.
    test_indices (np.ndarray): 원 입력 보류 행 위치.
    feature_names (List[str]): 원 Feature 순서.
    best_params (Dict[str,Any]): 학습 CV 선택 파라미터.
    best_cv_score (Optional[float]): 최고 CV 점수 또는 None.
    cv_results (Optional[Dict[str,Any]]): 전체 CV 결과 또는 None.
    warnings (List[str]): 분포·독립·확률 해석 가정.
```

```text
build_binomial_classifier(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    BinomialNB: 알려진 시행 수의 미적합 분류기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_binomial(X, y, test_indices=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    test_indices: 전용 명세의 test_indices 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    BinomialOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### naive-bayes-categorical

`src.MachineLearning.NaiveBayes.categorical`; 기본 JSON `naive-bayes-categorical.json`.

**CategoricalProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    test_size (float): 최종 보류 비율; fixed test_indices가 우선. 기본 0.3.
    random_state (int): 분할·CV seed. 기본 1234.
    stratify (bool): random 분할의 클래스 계층화. 기본 True.
    split_strategy (str): random/chronological; 후자는 입력 행 순서. 기본 'random'.
    scaling (str): Gaussian만 none/standard/minmax; 나머지는 none. 기본 'none'.
    cv (int): 선택용 학습 CV 폴드, 2 이상. 기본 3.
    scoring (Optional[str]): GridSearchCV 분류 지표; None은 accuracy. 기본 None.
    n_jobs (int): CV 병렬 작업 수; 0 불가. 기본 1.
    param_grid (Optional[Dict[str,Any]]): 분포별 estimator 인자명 또는 model__인자명 탐색. 기본 None.
    alpha (float): 양수 범주 평활화; 미지/결측 bucket 포함. 기본 1.0.
    fit_prior (bool): 학습 클래스 prior 여부. 기본 True.
    max_categories (int): Feature별 학습 수준 수 한도; 고카디널리티 보호. 기본 1000.
```

**CategoricalOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    model_name (str): 적합 분포별 클래스 이름.
    model (Any): 전처리 포함 Pipeline; JSON 제외.
    train_metrics (Dict[str,float]): 학습 accuracy·weighted precision/recall/F1.
    test_metrics (Dict[str,float]): 보류 같은 지표 및 이진 ROC AUC.
    y_test (np.ndarray): 보류 원 라벨.
    predictions (np.ndarray): 보류 예측 원 라벨.
    probabilities (np.ndarray): classes 열 순서 posterior.
    classes (np.ndarray): 학습 클래스 순서.
    class_counts (np.ndarray): 학습 클래스 빈도; 전체 자료 빈도 아님.
    confusion (np.ndarray): classes 순서의 참값×예측 혼동행렬.
    train_indices (np.ndarray): 원 입력 학습 행 위치.
    test_indices (np.ndarray): 원 입력 보류 행 위치.
    feature_names (List[str]): 원 Feature 순서.
    best_params (Dict[str,Any]): 학습 CV 선택 파라미터.
    best_cv_score (Optional[float]): 최고 CV 점수 또는 None.
    cv_results (Optional[Dict[str,Any]]): 전체 CV 결과 또는 None.
    warnings (List[str]): 분포·독립·확률 해석 가정.
```

```text
build_categorical_classifier(profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    profile: 전용 명세의 profile 입력.
    recipe_path: 전용 명세의 recipe_path 입력.
    overrides: 전용 명세의 overrides 입력.
Returns:
    CategoricalNB: 원 범주와 미지/결측 bucket 분류기.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

```text
run_categorical(X, y, test_indices=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    X: 전용 명세의 X 입력.
    y: 전용 명세의 y 입력.
    test_indices: 전용 명세의 test_indices 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    CategoricalOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-normal

`src.Visualization.qq.normal`; 기본 JSON `plot-qq-normal.json`.

**QQNormalProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
```

**QQNormalOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_normal(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQNormalOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-student-t

`src.Visualization.qq.student_t`; 기본 JSON `plot-qq-student-t.json`.

**QQStudentTProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
    df (float): 양수 자유도; fit_parameters=True에서는 함께 추정. 기본 5.0.
```

**QQStudentTOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_student_t(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQStudentTOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-exponential

`src.Visualization.qq.exponential`; 기본 JSON `plot-qq-exponential.json`.

**QQExponentialProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
```

**QQExponentialOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_exponential(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQExponentialOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-gamma

`src.Visualization.qq.gamma`; 기본 JSON `plot-qq-gamma.json`.

**QQGammaProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
    shape (float): 양수 gamma shape; fit에서는 추정. 기본 2.0.
```

**QQGammaOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_gamma(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQGammaOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-lognormal

`src.Visualization.qq.lognormal`; 기본 JSON `plot-qq-lognormal.json`.

**QQLognormalProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
    shape (float): 양수 로그 표준편차 s; fit에서는 추정. 기본 1.0.
```

**QQLognormalOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_lognormal(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQLognormalOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-weibull

`src.Visualization.qq.weibull`; 기본 JSON `plot-qq-weibull.json`.

**QQWeibullProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
    shape (float): 양수 Weibull c; fit에서는 추정. 기본 1.5.
```

**QQWeibullOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_weibull(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQWeibullOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-beta

`src.Visualization.qq.beta`; 기본 JSON `plot-qq-beta.json`.

**QQBetaProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
    shape_a (float): 양수 beta a; fit에서는 추정. 기본 2.0.
    shape_b (float): 양수 beta b; fit에서는 추정. 기본 2.0.
```

**QQBetaOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_beta(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQBetaOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-uniform

`src.Visualization.qq.uniform`; 기본 JSON `plot-qq-uniform.json`.

**QQUniformProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
```

**QQUniformOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_uniform(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQUniformOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-chi-square

`src.Visualization.qq.chi_square`; 기본 JSON `plot-qq-chi-square.json`.

**QQChiSquareProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
    df (float): 양수 자유도; fit에서 추정. 기본 5.0.
```

**QQChiSquareOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_chi_square(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQChiSquareOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-logistic

`src.Visualization.qq.logistic`; 기본 JSON `plot-qq-logistic.json`.

**QQLogisticProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
    scale (float): 양수 명시 척도. 기본 1.0.
```

**QQLogisticOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_logistic(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQLogisticOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-poisson

`src.Visualization.qq.poisson`; 기본 JSON `plot-qq-poisson.json`.

**QQPoissonProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (int): 명시 정수 지지 위치. 기본 0.
    mu (float): 비음수 평균 mu; fit에서 관측 평균. 기본 1.0.
```

**QQPoissonOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_poisson(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQPoissonOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-binomial

`src.Visualization.qq.binomial`; 기본 JSON `plot-qq-binomial.json`.

**QQBinomialProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (int): 명시 정수 지지 위치. 기본 0.
    n_trials (int): 알려진 양수 시행 수; fit에서도 고정. 기본 10.
    success_probability (float): 성공 확률 [0,1]; fit에서 평균/n. 기본 0.5.
```

**QQBinomialOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_binomial(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQBinomialOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-qq-bernoulli

`src.Visualization.qq.bernoulli`; 기본 JSON `plot-qq-bernoulli.json`.

**QQBernoulliProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
    color (str): 관측 표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
    fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
    loc (int): 명시 정수 지지 위치. 기본 0.
    success_probability (float): 성공 확률 [0,1]; fit에서 평균. 기본 0.5.
```

**QQBernoulliOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
    diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
    parameters (Dict[str,Any]): 전용 Profile 설정.
    warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
```

```text
plot_bernoulli(values, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    values: 전용 명세의 values 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    QQBernoulliOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-residual-fitted

`src.Visualization.regression.residual_fitted`; 기본 JSON `plot-regression-residual-fitted.json`.

**ResidualFittedProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    smooth (bool): LOWESS 탐색 곡선. 기본 True.
    fraction (float): LOWESS 이웃 비율 (0,1]. 기본 0.5.
```

**ResidualFittedOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_residual_fitted(model, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    ResidualFittedOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-scale-location

`src.Visualization.regression.scale_location`; 기본 JSON `plot-regression-scale-location.json`.

**ScaleLocationProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    smooth (bool): scale-location LOWESS 탐색. 기본 True.
    fraction (float): LOWESS 비율 (0,1]. 기본 0.5.
```

**ScaleLocationOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_scale_location(model, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    ScaleLocationOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-residual-order

`src.Visualization.regression.residual_order`; 기본 JSON `plot-regression-residual-order.json`.

**ResidualOrderProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    ordered (bool): True는 시간 순·동일 간격임을 호출자가 명시. 기본 False.
```

**ResidualOrderOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_residual_order(model, order=None, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    order: 전용 명세의 order 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    ResidualOrderOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-cooks-distance

`src.Visualization.regression.cooks_distance`; 기본 JSON `plot-regression-cooks-distance.json`.

**CooksDistanceProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    threshold (Optional[float]): 양수 검토 기준; None=4/n. 기본 None.
```

**CooksDistanceOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_cooks_distance(model, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    CooksDistanceOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-leverage

`src.Visualization.regression.leverage`; 기본 JSON `plot-regression-leverage.json`.

**LeverageProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    cook_levels (Optional[List[float]]): 양수 Cook 등고선; None은 [0.5,1.0]. 기본 None.
```

**LeverageOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_leverage(model, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    LeverageOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-coefficient-intervals

`src.Visualization.regression.coefficient_intervals`; 기본 JSON `plot-regression-coefficient-intervals.json`.

**CoefficientIntervalsProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    confidence (float): 계수 구간 신뢰수준 (0,1). 기본 0.95.
    include_intercept (bool): 절편을 함께 표시할지 여부. 기본 False.
```

**CoefficientIntervalsOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_coefficient_intervals(model, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    CoefficientIntervalsOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-observed-predicted

`src.Visualization.regression.observed_predicted`; 기본 JSON `plot-regression-observed-predicted.json`.

**ObservedPredictedProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    sample_role (str): train/validation/test 출처 선언. 기본 'test'.
```

**ObservedPredictedOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_observed_predicted(observed, predictions, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    observed: 전용 명세의 observed 입력.
    predictions: 전용 명세의 predictions 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    ObservedPredictedOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-prediction-interval

`src.Visualization.regression.prediction_interval`; 기본 JSON `plot-regression-prediction-interval.json`.

**PredictionIntervalProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    confidence (float): 평균 CI/새 관측 PI 신뢰수준. 기본 0.95.
    max_points (int): 표시 행 한도; 반환은 전체 새 행. 기본 200.
```

**PredictionIntervalOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_prediction_interval(model, X_new, observed=None, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    X_new: 전용 명세의 X_new 입력.
    observed: 전용 명세의 observed 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    PredictionIntervalOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-partial-residual

`src.Visualization.regression.partial_residual`; 기본 JSON `plot-regression-partial-residual.json`.

**PartialResidualProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    smooth (bool): LOWESS 비선형 탐색 요약. 기본 True.
    fraction (float): LOWESS 비율. 기본 0.5.
```

**PartialResidualOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_partial_residual(model, feature, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    model: 전용 명세의 model 입력.
    feature: 전용 명세의 feature 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    PartialResidualOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```


### plot-regression-vif

`src.Visualization.regression.vif`; 기본 JSON `plot-regression-vif.json`.

**VIFPlotProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
    color (str): 선/표식 색. 기본 'C0'.
    alpha (float): 표식 투명도 [0,1]. 기본 0.8.
    display_cap (float): 시각적 막대 상한; 실제 inf/수치는 data에 보존. 기본 20.0.
```

**VIFPlotOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 ax Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; JSON 제외.
    artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
    data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
    diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
    parameters (Dict[str,Any]): 적용 Profile.
    warnings (List[str]): 진단 가정·선택 후 해석 정보.
```

```text
plot_vif(collinearity, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

Args:
    collinearity: 전용 명세의 collinearity 입력.
    ax: 전용 명세의 ax 입력.
    profile: 전용 Profile 또는 None.
    recipe_path: Profile 생략 시 사용자 JSON 경로.
    overrides: Profile 생략 시 JSON 위에 적용할 재정의.
Returns:
    VIFPlotOutputDTO: 입력 순서와 설정을 보존한 결과.
Raises:
    ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
```

