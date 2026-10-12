# 모델별 레시피 관리

모델 구현·입력 및 결과 명세는 각 `.py`에, 실행 설정은 `src/Recipes/*.json`에 별도로 관리합니다.
공개 실행 함수에 Profile을 생략하면 해당 모델·목적의 JSON을 매번 읽어 Profile을 구성합니다.
모든 모듈의 `Example Execution`도 `load_profile()`로 같은 레시피를 사용합니다.

## 파일과 목적

| 모델/검정 이름 | 분류 레시피 | 회귀 또는 통계 레시피 |
| --- | --- | --- |
| `xgboost` | `xgboost-classification.json` | `xgboost-regression.json` |
| `catboost` | `catboost-classification.json` | `catboost-regression.json` |
| `lightgbm` | `lightgbm-classification.json` | `lightgbm-regression.json` |
| `adaboost` | `adaboost-classification.json` | `adaboost-regression.json` |
| `mlp` | `mlp-classification.json` | `mlp-regression.json` |
| `perceptron` | `perceptron-classification.json` | 지원 없음 |
| `adaline` | `adaline-classification.json` | 지원 없음 |
| `linear-regression` | 지원 없음 | `linear-regression-regression.json` |
| `oneway-anova` | 해당 없음 | `oneway-anova.json` |
| `twoway-anova` | 해당 없음 | `twoway-anova.json` |
| `manova` | 해당 없음 | `manova.json` |
| `posthoc` | 해당 없음 | `posthoc.json` |

기존 모델·검정 레시피는 위 17개 파일입니다. 전체 추가 기법까지 총 145개 파일입니다. 지도 학습·변수 선택의 기본 목적은 선형회귀를 제외하면 `classification`,
선형회귀에서는 `regression`입니다. 군집·차원 축소·통계·Apriori·SNA에는 task 인자를 지정하지 않습니다.

## 설정 순서와 형식

JSON은 Profile 필드 이름을 키로 쓰는 평면 객체입니다. 기존 XGBoost 회귀 파일은 원문을 유지합니다.
레시피에 없는 필드만 Profile에 명시된 기본값을 사용하고, `overrides`가 레시피 값보다 우선합니다.
Profile 기본값은 입력 계약의 기본값이며, 실제 기본 실행 설정은 모델별 JSON에서 관리합니다.

```python
from src.MachineLearning.Trees.xgboost import run_xgboost

# 기존 회귀 레시피 전체 적용: early_stopping_rounds=10도 실제 학습에 반영
result = run_xgboost(X, y, task="regression")

# 모델별 JSON을 기반으로 이번 실행만 변경
result = run_xgboost(X, y, task="regression", overrides={"n_estimators": 200})

# 다른 평면 JSON 사용; JSON의 task와 선택한 task는 일치해야 함
result = run_xgboost(X, y, task="regression", recipe_path="recipes/my-xgboost.json")
```

명시적 Profile이 필요한 경우에도 레시피를 먼저 읽을 수 있습니다.

```python
from src.Recipes import load_profile, load_recipe

profile = load_profile("xgboost", task="regression", overrides={"cv": 3})
result = run_xgboost(X, y, profile)
raw_settings = load_recipe("xgboost", task="regression")
```

기존 `run_*(X, y, ModelProfile(...))` 호출은 계속 지원합니다. 명시적 Profile은 JSON을 읽지 않습니다.
이 경우 `recipe_path`, `task`, `overrides`를 함께 지정하면 어느 설정이 우선인지 모호하므로 오류입니다.
반복 사용하는 설정은 JSON에서 수정하고, 일회성 실험만 overrides로 지정하세요.
기본 파일 경로는 loader 파일 위치 기준이므로 작업 디렉터리 변경에 영향을 받지 않습니다.

통계 검정도 같은 규칙을 사용합니다.

```python
from src.stats.tests.parametric.Average.Multivariate.OneWay import run_oneway_anova
from src.stats.tests.parametric.Average.Multivariate.PostHoc import run_posthoc

anova = run_oneway_anova(groups)
welch = run_oneway_anova(groups, overrides={"equal_var": False})
posthoc = run_posthoc(groups, overrides={"method": "holm", "equal_var": False})
```

## 기존 XGBoost 설정 호환

`alpha`는 `reg_alpha`, `lambda`는 `reg_lambda`로 변환합니다. 같은 설정 계층에서 별칭과 정식 키를
동시에 지정하면 오류입니다. overrides에서는 정식 키로 기존 별칭 값을 변경할 수 있습니다.
`gamma`, `objective`, `eval_metric`, `early_stopping_rounds`도 Profile과 실제 모델에 반영됩니다.
목적 함수는 현재 지원 계약에 따라 회귀 `reg:squarederror`, 이진 분류 `binary:logistic`, 다중 분류
`multi:softprob`만 허용합니다. `objective=null`이면 자료·task에 맞게 선택합니다.
MLP의 JSON 배열 `hidden_layer_sizes`는 Profile의 tuple로 변환합니다.

조기 종료가 활성화되면 현재 학습 자료 안에서 `validation_fraction`(기본 0.2)만큼 분리합니다.
GridSearchCV에서도 각 학습 폴드 내부에서 분리하며, 보류 테스트와 CV 평가 폴드는 조기 종료에 사용하지 않습니다.
전처리기는 현재 학습 폴드 전체에 적합되고, 조기 종료 분할은 전처리 후 모델 내부에서 수행됩니다.
반환된 DTO의 `early_stopping_train_indices`, `early_stopping_validation_indices`는 원 입력 행 위치이며,
`best_iteration`은 0부터 시작하는 최적 반복 번호입니다. 비활성 상태에서는 세 필드 모두 None입니다.
적합 Pipeline의 `named_steps["model"].native_model_`로 원래 XGBoost 추정기에 접근할 수 있습니다.
어댑터는 Python 3.7의 XGBoost 0.90과 Python 3.12의 XGBoost 3.0.5 API 차이를 처리하며,
최적 반복까지만 예측에 사용합니다.

## 오류와 검증

파일 누락·중복 키·알 수 없는 키·task 불일치·NaN/Infinity·객체가 아닌 JSON은 오류입니다.
설정을 조용히 무시하거나 파일 누락을 내장 기본값으로 대체하지 않습니다.
키와 형식은 로더가, 수치 범위와 데이터 조건은 공개 실행기가 검증합니다.
JSON은 설정과 CV 후보만 저장하며 적합 모델이나 결과 자료를 저장하지 않습니다.

`tests/test_recipes.py`는 145개 파일, 설정 우선순위, 파일 변경 반영, 작업 디렉터리 독립성,
잘못된 설정 거부, 조기 종료 분리 및 CV, 통계 설정 연결을 검증합니다.

## 추가 분석 기법 레시피

| 기법 | 레시피 |
| --- | --- |
| `kmeans` | `kmeans.json` |
| `dbscan` | `dbscan.json` |
| `hdbscan` | `hdbscan.json` |
| `k-distance` | `k-distance.json` |
| `pca` | `pca.json` |
| `lda` | `lda.json` |
| `umap` | `umap.json` |
| `tsne` | `tsne.json` |
| `mds` | `mds.json` |
| `som` | `som.json` |
| `naive-bayes` | `naive-bayes-classification.json` |
| `wrapper-selection` | `wrapper-selection-classification.json`, `wrapper-selection-regression.json` |
| `regularized-selection` | `regularized-selection-classification.json`, `regularized-selection-regression.json` |
| `apriori` | `apriori.json` |
| `sna` | `sna.json` |

Wrapper의 method와 규제 선택의 penalty도 각 레시피에서 관리합니다. 분류·회귀별 파일은 별도로 유지합니다.
전체 Profile·Output DTO 필드와 알고리즘 경계는 [전체 명세](ANALYSIS_CATALOG.md)를 참고하세요.


## 추가 검정과 TensorFlow 레시피

통계 검정은 `chi-square.json`, `independence.json`, `mcnemar.json`, `f-test.json`,
`normality.json`, `homogeneity.json`을 각각 사용합니다.
TensorFlow는 `tf-classification.json`, `tf-regression.json`, `tf-rnn-classification.json`,
`tf-rnn-regression.json`, `tf-time-series.json`으로 분리합니다. RNN 기본 목적은 regression입니다.
`hidden_layer_sizes`와 `target_columns` JSON 배열은 Profile tuple로 변환됩니다.
통계와 TensorFlow의 Profile에는 현재 `param_grid`가 없으며 자동 튜닝을 제공하지 않습니다.

## 통계 PDF 확장 레시피

신규/표준화 검정26개와 분포16종에 JSON42개를 추가했습니다. 분포 키는 distribution-normal 등으로 기존검정과 구별합니다. [전체 필드](STATISTICS_COMPLETE.md)를 참고하세요.

## Feature 전처리·상관분석 확장

수치/범주형 Feature의 순서가 있는 전처리, 별도 RoPE 임베딩 회전, 수치/범주 상관분석과
상관행렬 변환을 추가했습니다. [전체 명세](PREPROCESSING_CORRELATION_MODULES.md)를 참고하세요.

## Axes 기반 EDA 시각화

`src/Visualization`의 35개 모듈은 단일 ax에 그려 subplot으로 조합합니다. 회귀 시각화는 제외하고
분포·결측·범주·군집·차원 축소·분류 결정·연관/연결망·시간 패턴을 포함합니다.
[전체 Profile/DTO·의사결정 근거·UCI 조합 예제](EDA_VISUALIZATION_MODULES.md)를 참고하세요.
