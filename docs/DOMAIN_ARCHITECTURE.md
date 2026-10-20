# 분석 도메인 공통화 명세

승인된 데모를 바탕으로 19개 공개 모듈·19개 JSON·19개 skeleton을 추가했습니다. 모듈은 기존 작성 정보와 7개 섹션, Profile/Output DTO, Args/Returns/Raises 계약을 유지합니다. 전체 등록은 211개, JSON은 238개, 기본 레시피 로드 조합은 226개입니다.

## 도메인과 의존 방향

입력 데이터와 레시피 → Profile 검증 → 도메인 실행기 → 전용 Output DTO → 평가·시각화·시험 보고 순서입니다. src는 기출문제·tools를 import하지 않습니다. 문항별 가정·업무 해석·데이터 결합·PDF 서식은 시험 계층에 둡니다. JSON에는 실행 설정을 두며 관측값·적합 객체를 넣지 않습니다.

| 기능 | 공개 모듈 | 지원 범위 |
| --- | --- | --- |
| 생존 추정 | stats.survival.kaplan_meier / rmst | 우측 절단·동시 사건·log-log Greenwood 점별 CI·중앙수명·관측 내 시점 조회·RMST |
| 생존 검정·회귀 | stats.survival.logrank / cox | 다집단·모든 쌍 Holm, Cox HR/CI·생존예측·rank Schoenfeld/Holm |
| 시계열 | TimeSeries.arima / sarima | 학습 AIC 차수 선택·수렴 기록·고정 horizon 예측구간·미래 외생변수 검증 |
| 비선형 회귀 | MachineLearning.NonlinearModels.nonlinear_regression | 단일 설명변수 exponential/logistic/Michaelis-Menten, 초기값·경계·학습/보류 분리 |

## 불균형 처리의 지원표

| 기법 | 수치 | 전체 범주 | 혼합 | 원행 위치 |
| --- | --- | --- | --- | --- |
| RandomOverSampler / RandomUnderSampler | 가능 | 가능 | 가능 | 복제·선택 위치 |
| SMOTE / ADASYN / BorderlineSMOTE / SVMSMOTE | 가능 | 직접 적용 금지 | 직접 적용 금지 | 합성이므로 None |
| SMOTENC | 전부 수치이면 사용 금지 | 전부 범주이면 사용 금지 | 범주 열 위치 명시 | 합성이므로 None |
| SMOTEN | 연속량으로 사용 금지 | 가능 | 직접 적용 금지 | 합성이므로 None |
| TomekLinks / EditedNearestNeighbours / NearMiss | 가능 | 직접 적용 금지 | 먼저 타당한 거리 표현 필요 | 선택 위치 |
| ClusterCentroids | 가능 | 직접 적용 금지 | 먼저 타당한 거리 표현 필요 | 중심점이므로 None |

오버는 수치형을 지원하는 5종(무작위·SMOTE·ADASYN·Borderline·SVM)과 범주·혼합 특화 2종, 언더는 5종입니다. 범주형에 모든 거리 기법을 일괄 적용하지 않습니다. 숫자 코드도 의미상 범주이면 수치 거리로 취급하면 안 됩니다. cleaning 기법은 클래스 수의 균형을 보장하지 않습니다.

`run_*()`는 전용 DTO를 반환합니다. CV에서는 `src.Preprocessing.resampling._common.load_resampler()`로 같은 JSON 설정의 네이티브 sampler를 구성하여 imblearn Pipeline 안에 넣습니다. 전처리·선택·보완 모두 학습 폴드 안에서 수행하고 보류 자료 분포는 유지합니다. sampler factory는 적합 결과를 반환하는 분석 API와 구분됩니다.

## 시험 풀이 연결과 남은 경계

- 기존 reconstruction 공통부의 SARIMA·KM 직접 계산을 src 호출로 이관했습니다(18·19·19 보충·25·28회 관련 문항).
- 30·33회 표본 보완과 33회 ARMA 차수 비교는 새 공통 모듈을 사용합니다. 동일 시드·후보·평가 규칙을 유지합니다.
- 생존분석 14문항의 KM/CI·RMST·로그순위·Cox/PH 계산은 src DTO를 사용합니다. 그림·표·문항 보고는 해설 계층에서 조합합니다.
- 그 밖의 ML·통계에는 기존 src 호출과 sklearn/statsmodels/SciPy 직접 호출이 함께 남아 있습니다. 이들의 분할·후처리 계약을 바꾸어 억지로 단일 runner에 맞추지 않았습니다. 문항-모듈 정적 참조표는 남은 직접 계산을 별도로 표시합니다.
- 비선형 회귀의 새로운 함수 모형과 모든 리샘플링 기법을 기출 출제 확인 유형으로 주장하지 않습니다. 기존 지문에 없던 기법을 임의로 정답에 추가하지 않았습니다.

## 환경과 검증

일반 분석 환경과 분리한 `requirements-domain-py37.txt` / `requirements-domain-py312.txt`를 사용합니다. lifelines·imblearn은 실행 시 지연 import하므로 Profile/레시피 조회에 설치가 필수는 아닙니다. Python 3.7은 전용 CI에서 실제 실행하며, 로컬 검증 여부와 CI 성공 여부는 검증 기록에서 구분합니다.

`tests/test_domain_extension.py`는 손계산·공식 구현 대조·경계값·학습/보류 분리·CV clone·모든 신규 레시피를 검사합니다. `tools/exams/build_catalog.py`는 30묶음·229문항과 등록부 수치를 생성하고 `tools/exams/audit_catalog.py`가 생성 결과·문서의 불일치를 검사합니다.

## 기술 근거

- https://lifelines.readthedocs.io/en/stable/fitters/univariate/KaplanMeierFitter.html
- https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
- https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html
- https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
- https://imbalanced-learn.org/stable/over_sampling.html
- https://imbalanced-learn.org/stable/under_sampling.html

## 기법별 전체 Profile / DTO 필드

### survival-kaplan-meier

경로: `src.stats.survival.kaplan_meier`; 기본 JSON: `survival-kaplan-meier.json`.

**KaplanMeierProfile**

```text
입력 설정. 관측 자료는 설정 JSON과 분리합니다.

Attributes:
    label (str): 집단 이름. 공백이 아닌 문자열, 기본 'sample'.
    alpha (float): 점별 log-log Greenwood 신뢰구간 유의수준, 기본 0.05.
    evaluation_times (Optional[List[float]]): 조회할 시각. 기본 None.
        유한한 비음수 값이며 최대 관측시간 이하. 입력 순서·중복을 보존합니다.
```

**KaplanMeierOutputDTO**

```text
생존 추정 결과. 모든 표 배열은 times의 오름차순과 대응합니다.

Attributes:
    method (str): 추정 방법 이름.
    parameters (Dict[str, Any]): 적용한 Profile 전체.
    sample_size (int): 관측 대상 수.
    times (np.ndarray): 사건 또는 절단이 발생한 고유 관측 시각.
    at_risk (np.ndarray): 해당 시각 직전 위험집합 수.
    events (np.ndarray): 해당 시각 사건 수.
    censored (np.ndarray): 해당 시각 중도절단 수.
    survival (np.ndarray): 사건 반영 후 우연속 S(t)=P(T>t) 추정값.
    confidence_lower (np.ndarray): times에 대응하는 점별 신뢰구간 하한.
    confidence_upper (np.ndarray): times에 대응하는 점별 신뢰구간 상한.
    cumulative_failure (np.ndarray): 1-S(t). 경쟁위험 누적발생함수가 아닙니다.
    median_survival_time (Optional[float]): S(t)<=0.5인 최초 시각.
        관찰 중 미도달이면 None이며 JSON null로 저장합니다.
    evaluation_times (Optional[np.ndarray]): 요청 조회 시각, 미요청 None.
    survival_at_times (Optional[np.ndarray]): 요청 시각의 S(t), 미요청 None.
    warnings (List[str]): 해석과 지원 범위의 제한.
```

### survival-rmst

경로: `src.stats.survival.rmst`; 기본 JSON: `survival-rmst.json`.

**RMSTProfile**

```text
Attributes:
tau (float): 적분 상한, 기본 24. 0<tau<=최대 관측시간.
```

**RMSTOutputDTO**

```text
Attributes:
method (str): Kaplan-Meier step integral.
tau (float): 사용한 공통 적분 상한.
estimate (float): 0~tau 생존곡선 아래 면적, 시간과 같은 단위.
warnings (List[str]): 추론·외삽 제한.
```

### survival-logrank

경로: `src.stats.survival.logrank`; 기본 JSON: `survival-logrank.json`.

**LogrankProfile**

```text
Attributes:
alpha (float): 유의수준, 기본 0.05, 0<alpha<1.
pairwise (bool): 모든 쌍 비교와 Holm 보정을 반환, 기본 False.
    전체 검정 기각 후 수행하는 절차는 호출자가 사전 규칙에 따라 제어합니다.
```

**LogrankOutputDTO**

```text
Attributes:
statistic (float): (k-1)차원 초기하 공분산의 카이제곱 통계량.
p_value (float): 점근 카이제곱 상측 p값.
reject_h0 (bool): p_value<alpha.
alpha (float): 실제 유의수준.
degrees_of_freedom (int): 집단수-1.
null_hypothesis (str): 모든 집단의 생존함수가 동일.
alternative_hypothesis (str): 적어도 한 집단의 생존함수가 다름.
groups (List[Any]): 정렬한 집단 라벨.
pairwise (List[Dict[str, Any]]): 모든 쌍의 가설·통계량·원/보정 p값·판정.
warnings (List[str]): 점근 검정·연관성 해석 제한.
```

### survival-cox

경로: `src.stats.survival.cox`; 기본 JSON: `survival-cox.json`.

**CoxProfile**

```text
Attributes:
duration_col (str): 시간 열 이름, 기본 time.
event_col (str): 사건 열 이름, 기본 event. 나머지 열은 모두 수치 공변량.
alpha (float): 계수 구간·PH Holm 판정 유의수준, 기본 0.05.
penalizer (float): L2 벌점, 기본 0, 비음수. 양수면 벌점 추정의 근사 추론.
ph_diagnostics (bool): rank 변환 Schoenfeld 검정+Holm, 기본 True.
prediction_times (Optional[List[float]]): 새 관측 생존예측 시각, 기본 None.
    새 관측과 함께 지정하며 0~최대 학습 관측시간의 증가하는 고유값.
```

**CoxOutputDTO**

```text
Attributes:
method (str): Efron ties/Breslow baseline Cox.
coefficients (pd.DataFrame): 공변량별 coef, SE, HR, CI, z, 원 p값.
    Wald 검정 H0: beta_j=0, H1: beta_j!=0, 각각 양측 탐색 검정.
ph_diagnostics (Optional[pd.DataFrame]): H0: 계수 시간불변, H1: 시간의존.
    rank Schoenfeld 통계량·원/보정 p값·판정. 미요청 None.
predictions (Optional[pd.DataFrame]): 행=시각, 열=새 입력 행 라벨의 생존확률.
sample_size (int): 학습 대상 수.
events (int): 사건 수.
alpha (float): 적용 유의수준.
model (Any): 적합 lifelines CoxPHFitter. 기본 JSON에서 제외.
warnings (List[str]): 벌점·예측·추론 조건.
```

### timeseries-arima

경로: `src.TimeSeries.arima`; 기본 JSON: `timeseries-arima.json`.

**ARIMAProfile**

```text
Attributes:
order (List[int]): 기본 비계절 (p,d,q), 기본 [1,0,0].
candidate_orders (Optional[List[List[int]]]): 학습 AIC 비교 후보. 기본 None.
steps (int): 예측 horizon, 기본 12, 양의 정수.
alpha (float): 예측구간 유의수준, 기본 0.05.
trend (Optional[str]): None/n/c/t/ct, 기본 None. backend의 적분 차수 제약 적용.
maxiter (int): 최적화 최대 반복, 기본 100.
enforce_stationarity (bool): 정상 AR 제약, 기본 True.
enforce_invertibility (bool): 가역 MA 제약, 기본 True.
require_convergence (bool): 수렴 후보만 선택, 기본 True.
```

**ARIMAOutputDTO**

```text
Attributes:
method (str): backend 종류.
parameters (Dict[str, Any]): 적용한 전체 Profile.
selected_order (List[int]): 학습 AIC 선택 비계절 차수.
seasonal_order (List[int]): 계절 차수. ARIMA는 [0,0,0,0].
candidates (List[Dict[str, Any]]): 후보별 AIC·BIC·수렴 또는 실패 이유.
predictions (np.ndarray): steps개 미래 시점 예측 평균.
prediction_lower (np.ndarray): 같은 순서의 예측구간 하한.
prediction_upper (np.ndarray): 같은 순서의 예측구간 상한.
forecast_index (Optional[List[Any]]): backend 미래 인덱스, 배열 입력은 None 가능.
residuals (np.ndarray): 학습 관측 순서의 잔차. 초기화 구간은 별도 고려.
aic (float): 선택 후보 학습 AIC.
bic (float): 선택 후보 학습 BIC.
converged (bool): 최적화 수렴 여부.
model (Any): 적합 statsmodels 결과. 기본 JSON에서 제외.
warnings (List[str]): 선택 편향·외생변수 가용성·미수렴 제한.
```

### timeseries-sarima

경로: `src.TimeSeries.sarima`; 기본 JSON: `timeseries-sarima.json`.

**SARIMAProfile**

```text
Attributes:
order (List[int]): 기본 비계절 (p,d,q), 기본 [1,0,0].
candidate_orders (Optional[List[List[int]]]): 학습 AIC 비교 후보. 기본 None.
steps (int): 예측 horizon, 기본 12, 양의 정수.
alpha (float): 예측구간 유의수준, 기본 0.05.
trend (Optional[str]): None/n/c/t/ct, 기본 None. backend의 적분 차수 제약 적용.
maxiter (int): 최적화 최대 반복, 기본 100.
enforce_stationarity (bool): 정상 AR 제약, 기본 True.
enforce_invertibility (bool): 가역 MA 제약, 기본 True.
require_convergence (bool): 수렴 후보만 선택, 기본 True.
seasonal_order (List[int]): 계절 (P,D,Q,s), 기본 [1,0,0,12], s>=2.
```

**SARIMAOutputDTO**

```text
Attributes:
method (str): backend 종류.
parameters (Dict[str, Any]): 적용한 전체 Profile.
selected_order (List[int]): 학습 AIC 선택 비계절 차수.
seasonal_order (List[int]): 계절 차수. ARIMA는 [0,0,0,0].
candidates (List[Dict[str, Any]]): 후보별 AIC·BIC·수렴 또는 실패 이유.
predictions (np.ndarray): steps개 미래 시점 예측 평균.
prediction_lower (np.ndarray): 같은 순서의 예측구간 하한.
prediction_upper (np.ndarray): 같은 순서의 예측구간 상한.
forecast_index (Optional[List[Any]]): backend 미래 인덱스, 배열 입력은 None 가능.
residuals (np.ndarray): 학습 관측 순서의 잔차. 초기화 구간은 별도 고려.
aic (float): 선택 후보 학습 AIC.
bic (float): 선택 후보 학습 BIC.
converged (bool): 최적화 수렴 여부.
model (Any): 적합 statsmodels 결과. 기본 JSON에서 제외.
warnings (List[str]): 선택 편향·외생변수 가용성·미수렴 제한.
```

### nonlinear-regression

경로: `src.MachineLearning.NonlinearModels.nonlinear_regression`; 기본 JSON: `nonlinear-regression.json`.

**NonlinearRegressionProfile**

```text
Attributes:
model_name (str): exponential/logistic/michaelis_menten. 기본 exponential.
    exponential=a*exp(b*x)+c; logistic=L*expit(k*(x-x0)); michaelis=a*x/(b+x).
initial_parameters (List[float]): 위 순서의 초기값, 기본 [1,0.1,0].
lower_bounds (Optional[List[float]]): 동일 길이 유한 하한. 기본 None=무한 경계.
upper_bounds (Optional[List[float]]): 동일 길이 유한 상한. 기본 None=무한 경계.
maxfev (int): 최대 함수 평가 수, 기본 10000, 양의 정수.
absolute_sigma (bool): sigma를 알려진 관측 표준편차로 해석, 기본 False.
    False면 공분산을 학습 잔차 분산으로 재조정. True면 sigma 입력 필수.
```

**NonlinearRegressionOutputDTO**

```text
Attributes:
model_name (str): 함수 종류.
parameters (Dict[str, Any]): 전체 실행 설정.
parameter_names (List[str]): 계수·공분산의 대응 순서.
coefficients (np.ndarray): 비선형 최소제곱 추정 계수.
covariance (np.ndarray): 국소 선형화 근사 계수 공분산. 비식별이면 비유한 가능.
fitted_values (np.ndarray): 학습 입력 순서의 적합값.
residuals (np.ndarray): 학습 y-적합값.
training_rmse (float): 학습 RMSE. 일반화 성능이 아님.
predictions (Optional[np.ndarray]): x_test 순서의 예측값. 미요청 None.
test_rmse (Optional[float]): y_test를 함께 제공한 경우의 보류 RMSE.
warnings (List[str]): 함수형태·초기값·공분산 해석 제한.
```

### resampling-random-over

경로: `src.Preprocessing.resampling.random_over`; 기본 JSON: `resampling-random-over.json`.

**RandomOverProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
```

**RandomOverOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-smote

경로: `src.Preprocessing.resampling.smote`; 기본 JSON: `resampling-smote.json`.

**SMOTEProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
k_neighbors (int: 합성 이웃 수. 기본 5, 양수; 클래스에 충분한 표본 필요.)
```

**SMOTEOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-adasyn

경로: `src.Preprocessing.resampling.adasyn`; 기본 JSON: `resampling-adasyn.json`.

**ADASYNProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
n_neighbors (int: 거리 이웃 수. 양수. 기본 5.)
```

**ADASYNOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-borderline-smote

경로: `src.Preprocessing.resampling.borderline_smote`; 기본 JSON: `resampling-borderline-smote.json`.

**BorderlineSMOTEProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
k_neighbors (int: 합성 이웃 수. 기본 5, 양수; 클래스에 충분한 표본 필요.)
m_neighbors (int: 경계 판별 이웃 수. 기본 10, 양수.)
```

**BorderlineSMOTEOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-svm-smote

경로: `src.Preprocessing.resampling.svm_smote`; 기본 JSON: `resampling-svm-smote.json`.

**SVMSMOTEProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
k_neighbors (int: 합성 이웃 수. 기본 5, 양수; 클래스에 충분한 표본 필요.)
m_neighbors (int: 경계 판별 이웃 수. 기본 10, 양수.)
```

**SVMSMOTEOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-smotenc

경로: `src.Preprocessing.resampling.smotenc`; 기본 JSON: `resampling-smotenc.json`.

**SMOTENCProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
k_neighbors (int: 합성 이웃 수. 기본 5, 양수; 클래스에 충분한 표본 필요.)
categorical_features (List[int]: 범주 열의 위치. 기본 [0]. 일부 범주·일부 수치가 필수.)
```

**SMOTENCOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-smoten

경로: `src.Preprocessing.resampling.smoten`; 기본 JSON: `resampling-smoten.json`.

**SMOTENProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
k_neighbors (int: 합성 이웃 수. 기본 5, 양수; 클래스에 충분한 표본 필요.)
```

**SMOTENOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-random-under

경로: `src.Preprocessing.resampling.random_under`; 기본 JSON: `resampling-random-under.json`.

**RandomUnderProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
```

**RandomUnderOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-tomek-links

경로: `src.Preprocessing.resampling.tomek_links`; 기본 JSON: `resampling-tomek-links.json`.

**TomekLinksProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
```

**TomekLinksOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-edited-nearest-neighbours

경로: `src.Preprocessing.resampling.edited_nearest_neighbours`; 기본 JSON: `resampling-edited-nearest-neighbours.json`.

**EditedNearestNeighboursProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
n_neighbors (int: 거리 이웃 수. 양수. 기본 3.)
```

**EditedNearestNeighboursOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-near-miss

경로: `src.Preprocessing.resampling.near_miss`; 기본 JSON: `resampling-near-miss.json`.

**NearMissProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
n_neighbors (int: 거리 이웃 수. 양수. 기본 3.)
version (int: NearMiss 방식 1/2/3. 기본 1.)
```

**NearMissOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

### resampling-cluster-centroids

경로: `src.Preprocessing.resampling.cluster_centroids`; 기본 JSON: `resampling-cluster-centroids.json`.

**ClusterCentroidsProfile**

```text
Attributes:
sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
random_state (int: 재현 시드. 기본 1234, 0 이상.)
```

**ClusterCentroidsOutputDTO**

```text
Attributes:
method (str): 실행 기법 이름.
X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
y_resampled (Any): 보완한 학습 타깃.
class_counts_before (Dict[Any,int]): 원래 클래스별 수.
class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
    복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
warnings (List[str]): 평가 누출·원행 대응 제한.
```

