# 전체 분석 모듈 명세

현재 main의 기존 구현과 이번 추가 구현을 기준으로 작성했습니다. 표준 명세의 모듈은 211개, 모델·목적별 JSON 레시피는 238개입니다.
기존 단일·독립 두 표본 Z·t 검정 파일 2개도 아래에 별도로 명세합니다.
비모수 검정의 빈 파일은 구현 완료로 집계하지 않습니다.

## 공통 규약

- 표준 모듈은 파일 설명·분석 가정·작성 정보·변경 이력·메타데이터와 같은 7개 섹션을 갖습니다.
- 각 모듈의 Profile과 Output DTO는 공통 필드도 포함해 모든 필드를 직접 정의합니다.
- Profile 생략 시 모델·목적별 JSON을 읽고, 일회성 `overrides`를 적용합니다. 파일 누락·알 수 없는 키·중복 키는 오류입니다.
- 명시적 Profile과 레시피 옵션을 동시에 지정하지 않습니다. 입력의 결측·범주 인코딩은 호출 전에 처리합니다.
- 지도 학습·변수 선택은 무작위 IID 학습/보류 분할입니다. 테스트 자료는 스케일링·선택·CV 적합에 사용하지 않습니다.
- 군집화·차원 축소는 입력 전체를 적합하는 탐색 분석입니다. 보류 성능을 계산하지 않으며, 예측 작업의 전처리로 쓰려면 학습 자료에만 적합해야 합니다.
- 지도 결과는 `test_indices` 행 순서, 탐색 결과는 원 입력 행 순서입니다. 선택 열은 원 특성 순서입니다.
- `to_dict()` / `to_json()`은 실행 객체를 제외하고 결과 배열·표·지표를 반환합니다. JSON은 모델 복원 형식이 아닙니다.

## 구현 목록

### 지도 학습

| 기법 | 지원 범위 | 실행기 | 상태 |
| --- | --- | --- | --- |
| `xgboost` | 분류·회귀; 조기 종료와 학습 구간 CV | `run_xgboost` | 기존 구현 |
| `catboost` | 분류·회귀; 현재 수치 입력만 지원 | `run_catboost` | 기존 구현 |
| `lightgbm` | 분류·회귀 | `run_lightgbm` | 기존 구현 |
| `adaboost` | 분류·회귀; 의사결정나무 기반 | `run_adaboost` | 기존 구현 |
| `mlp` | 분류·회귀; 다층 퍼셉트론 | `run_mlp` | 기존 구현 |
| `perceptron` | 분류; 단층 선형 모델, 확률 없음 | `run_perceptron` | 기존 구현 |
| `adaline` | 이진 분류; 선형 활성화의 배치 평균제곱 손실 | `run_adaline` | 기존 구현 |
| `linear-regression` | 단순·다중 OLS / positive=True일 때 NNLS | `run_linear_regression` | 기존 구현 |
| `naive-bayes` | Gaussian / Multinomial / Bernoulli / Complement 분류 | `run_naive_bayes` | 이번 추가 |

### 군집화·거리 진단

| 기법 | 지원 범위 | 실행기 | 상태 |
| --- | --- | --- | --- |
| `kmeans` | 고정 k 군집; 중심·inertia·silhouette | `run_kmeans` | 이번 추가 |
| `dbscan` | eps 밀도 군집; 핵심점·잡음(-1) | `run_dbscan` | 이번 추가 |
| `hdbscan` | 계층 밀도 군집; 소속 강도·GLOSH 이상 점수 | `run_hdbscan` | 이번 추가 |
| `k-distance` | 자기 자신 포함 k번째 이웃 거리와 정렬 곡선; eps 후보 진단 | `run_k_distance` | 이번 추가 |

### 차원 축소

| 기법 | 지원 범위 | 실행기 | 상태 |
| --- | --- | --- | --- |
| `pca` | 선형 비지도 축소; 주성분·분산 설명·재구성 오차 | `run_pca` | 이번 추가 |
| `lda` | 지도 판별 축소; y 필요; 최대 min(p, 클래스 수-1)축 | `run_lda` | 이번 추가 |
| `umap` | 비선형 비지도 축소; 새 입력 transform | `run_umap` | 이번 추가 |
| `tsne` | t-SNE; 시각화용 전 자료 적합; 새 입력 transform 없음 | `run_tsne` | 이번 추가 |
| `som` | 온라인 Kohonen SOM; 2차원 BMU 격자·코드북·양자화/위상 오차 | `run_som` | 이번 추가 |
| `mds` | metric/nonmetric SMACOF; 유클리드 입력; 새 입력 transform 없음 | `run_mds` | 이번 추가 |

### 변수 선택

| 기법 | 지원 범위 | 실행기 | 상태 |
| --- | --- | --- | --- |
| `wrapper-selection` | Forward / Backward / CV 점수 기반 양방향 Stepwise; 분류·회귀 | `run_wrapper_selection` | 이번 추가 |
| `regularized-selection` | L1·L2; 분류 Logistic / 회귀 Lasso·Ridge 계수 기반 선택 | `run_regularized_selection` | 이번 추가 |

### 연관규칙·연결망

| 기법 | 지원 범위 | 실행기 | 상태 |
| --- | --- | --- | --- |
| `apriori` | 빈발 항목집합; support·confidence·lift·leverage·conviction | `run_apriori` | 이번 추가 |
| `sna` | 유향·무향 연결망; 중심성·PageRank·커뮤니티·모듈성 | `run_sna` | 이번 추가 |

### 통계 검정

| 기법 | 지원 범위 | 실행기 | 상태 |
| --- | --- | --- | --- |
| `oneway-anova` | 고전·Welch 일원 ANOVA | `run_oneway_anova` | 기존 구현 |
| `twoway-anova` | Type I·II·III; 교호작용 포함/제외 | `run_twoway_anova` | 기존 구현 |
| `manova` | 일원 요인 MANOVA; Pillai·Wilks·Hotelling-Lawley·Roy | `run_manova` | 기존 구현 |
| `posthoc` | Tukey HSD / Student·Welch 쌍별 t + Holm·Bonferroni | `run_posthoc` | 기존 구현 |

### 추가 검정·가정 진단

| 기법 | 지원 범위 | 실행기 |
| --- | --- | --- |
| chi-square | Pearson 적합도, 지정 기대도수와 ddof | run_chi_square |
| independence | 교차표 Pearson 독립성, Yates, Cramér’s V | run_independence |
| mcnemar | 대응 이진 정확/근사 검정 | run_mcnemar |
| f-test | 정규 독립 두 표본 분산비, 세 방향 | run_f_test |
| normality | Shapiro/D’Agostino–Pearson/Jarque–Bera | run_normality |
| homogeneity | Levene/Bartlett/Fligner–Killeen | run_homogeneity |

### TensorFlow

| 기법 | 지원 범위 | 실행기 |
| --- | --- | --- |
| tf-classification | Dense 이진/다중 클래스 softmax | run_tf_classification |
| tf-regression | Dense 단일/다중 타깃 수치 회귀 | run_tf_regression |
| tf-rnn | SimpleRNN/LSTM/GRU sequence-to-one 분류·회귀 | run_tf_rnn |
| tf-time-series | 시간 순 윈도·single-shot 미래 단일/다중 스텝 | run_tf_time_series |

## 용어와 알고리즘 경계

- 요청의 HDBCAN은 HDBSCAN, 다차원 척도법은 MDS로 구현했습니다.
- 사용자 확인에 따라 Social-neural net 항목은 **SNA 사회연결망 분석**으로 구현했습니다. 학습 신경망을 뜻하지 않습니다.
- LDA는 Linear Discriminant Analysis이며, 토픽 모델 Latent Dirichlet Allocation이 아닙니다.
- k-distance의 k는 자기 자신을 포함하는 이웃 수입니다. DBSCAN의 `min_samples`와 같은 정의로 eps 후보 곡선을 만듭니다. 자동 최적 eps를 반환하지 않습니다.
- Wrapper의 Forward/Backward는 목표 변수 수까지 greedy CV 선택을 수행합니다. `n_features_to_select=None`은 점수 개선이 없으면 중단합니다.
- Stepwise는 CV 점수가 `min_improvement`보다 개선되는 추가·제거를 번갈아 허용하는 탐색입니다. 방문 집합을 다시 탐색하지 않고, p-value 기반 회귀 검정은 구현하지 않습니다. 지정 변수 수는 상한이며 조기 종료할 수 있습니다.
- Wrapper의 분류 평가 모델은 LogisticRegression, 회귀는 OLS입니다. 내부 후보 CV마다 스케일러를 해당 학습 폴드에서 적합합니다.
- L1 회귀는 Lasso, L2 회귀는 Ridge, 분류는 규제 LogisticRegression의 OvR입니다. 최종 선택 열의 평가 모델은 별도의 LogisticRegression/OLS입니다.
- L1은 희소 계수 기준(기본 1e-8), L2는 평균 절대 계수 중요도 기준으로 선택합니다. L2 자체가 계수를 0으로 만드는 변수 선택이라고 해석하지 않습니다.
- 계수는 선택용 스케일러 좌표계입니다. 다중 클래스는 계수 절대값의 클래스 평균을 중요도로 사용합니다. 선택 변수가 0개이면 오류입니다.
- SOM은 2차원 직사각형 격자의 온라인 Kohonen 갱신입니다. BMU 좌표는 같은 노드에 해당하는 관측에서 중복될 수 있으며 군집 수를 자동 추정하지 않습니다.
- MDS의 `stress`는 라이브러리 목적 함수 값입니다. 구버전 metric MDS에서는 최종 갱신 전 값일 수 있습니다. `distance_reconstruction_error`는 최종 좌표의 거리 오차를 재계산하며, nonmetric 목적 함수와는 다릅니다.
- Apriori는 거래 목록만 지원합니다. 거래 내 중복 항목은 한 번, 빈 거래는 지지도 분모에 포함합니다. 후보·규칙 한도 초과는 오류이며 결과를 조용히 잘라내지 않습니다. conviction의 무한대는 엄격 JSON에서 null로 변환됩니다.
- SNA는 문자열 노드의 단순 Graph/DiGraph입니다. 중복 연결의 강도를 합산하고 자기 연결은 거부합니다. 중심성의 거리는 강도의 역수, PageRank는 강도 가중입니다. 유향 closeness는 들어오는 거리 기준입니다.
- SNA 커뮤니티는 무향 투영의 greedy modularity입니다. 유향 양방향 연결은 강도를 합산하며 고립 노드는 포함합니다.

## 새 자료의 실행 지원

| 기법 | 지원 메서드 | 제한 |
| --- | --- | --- |
| 지도 학습 9개 / 변수 선택 2개 | `result.model.predict(X_new)` | 학습과 같은 열 순서; 확률은 지원 모델만 |
| K-means | `model.predict(X_new)`, `model.transform(X_new)` | 기존 중심에 배정 또는 중심까지 거리 |
| DBSCAN / 이 실행기의 HDBSCAN | 적합 라벨만 반환 | 새 관측 predict 제공 없음 |
| k-distance | 스케일러와 NearestNeighbors 실행 객체 | 진단용; 새 이웃 조회 시 자기 자신 포함 규칙이 달라짐 |
| PCA / LDA / UMAP / SOM | `model.transform(X_new)` | LDA 적합에는 y가 필요, SOM은 BMU 격자 좌표 |
| t-SNE / MDS | 새 입력 transform 없음 | 좌표는 적합된 자료에만 대응 |
| Apriori | 입력 거래를 다시 분석 | 연관은 인과 관계를 뜻하지 않음 |
| SNA | networkx Graph/DiGraph 분석 | 예측 신경망이 아님 |

## 사용 예제

```python
from src.MachineLearning.Clustering.k_distance import run_k_distance
from src.MachineLearning.Clustering.dbscan import run_dbscan
from src.MachineLearning.FeatureSelection.wrapper import run_wrapper_selection
from src.MachineLearning.FeatureSelection.regularized import run_regularized_selection
from src.MachineLearning.Decomposition.lda import run_lda

curve = run_k_distance(X, overrides={"k": 5})
# 곡선을 보고 eps를 결정: 기본 standard 스케일링 좌표계
clusters = run_dbscan(X, overrides={"eps": 0.5, "min_samples": 5})
selected = run_wrapper_selection(X, y, task="regression",
    overrides={"method": "backward", "n_features_to_select": 3})
l2_selected = run_regularized_selection(X, y, task="regression",
    overrides={"penalty": "l2"})
# 이진 클래스는 판별 축이 최대 1개
coordinates = run_lda(X_binary, y_binary, overrides={"n_components": 1})
print(selected.selected_features, selected.test_metrics)
print(selected.model.predict(X_new))
```

## 모듈별 전체 입력·결과 필드

Profile 기본값은 입력 계약의 기본값입니다. 실제 기본 실행은 별도 JSON 설정을 우선하며,
예컨대 기존 XGBoost 회귀 JSON에는 조기 종료가 활성화되어 있습니다.

### xgboost

분류·회귀; 조기 종료와 학습 구간 CV.

파일: [`src/MachineLearning/Trees/xgboost.py`](../src/MachineLearning/Trees/xgboost.py)

Profile: `XGBoostProfile` · DTO: `XGBoostOutputDTO`
레시피: `xgboost-classification.json`, `xgboost-regression.json`

```python
run_xgboost(X: Any, y: Any, profile: Optional[src.MachineLearning.Trees.xgboost.XGBoostProfile] = None, *, recipe_path: Optional[str] = None, task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Trees.xgboost.XGBoostOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. 기본 classification. classification/regression 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'none'` | 스케일링 방법. none/standard/minmax; 기본 none. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고. |
| `n_estimators` | `int` | `100` | 부스팅 단계 수. 기본 100, 1 이상 정수. |
| `max_depth` | `int` | `3` | 트리 최대 깊이. 기본 3, 1 이상 정수. |
| `learning_rate` | `float` | `0.1` | 학습률. 기본 0.1, 유한한 양수. |
| `subsample` | `float` | `1.0` | 트리별 행 표본 비율. 기본 1.0, (0,1]. |
| `colsample_bytree` | `float` | `1.0` | 트리별 특성 표본 비율. 기본 1.0, (0,1]. |
| `min_child_weight` | `float` | `1.0` | 리프 최소 Hessian 합. 기본 1.0, 0 이상. |
| `reg_alpha` | `float` | `0.0` | L1 규제 강도. 기본 0.0, 0 이상. |
| `reg_lambda` | `float` | `1.0` | L2 규제 강도. 기본 1.0, 0 이상. |
| `gamma` | `float` | `0.0` | 트리 분할 최소 손실 감소. 기본 0.0, 유한한 0 이상 값. |
| `objective` | `Optional[str]` | `None` | 목적함수. 기본 None이면 task와 클래스 수에 맞춰 선택. 분류는 binary:logistic 또는 multi:softprob, 회귀는 reg:squarederror. |
| `eval_metric` | `Optional[str]` | `None` | 조기 종료 평가 지표. 기본 None이면 회귀 rmse, 이진 logloss, 다중 mlogloss. |
| `early_stopping_rounds` | `Optional[int]` | `None` | 개선 없는 라운드 허용 수. 기본 None이면 조기 종료 없음. 지정 시 양의 정수. |
| `validation_fraction` | `float` | `0.2` | 조기 종료 시 현재 학습 폴드 내부 검증 비율. 기본 0.2, (0,1). 보류 테스트는 사용하지 않음. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. |
| `feature_importances` | `Optional[numpy.ndarray]` | 적합 모델의 입력 열 순서 특성 중요도 (n_features,). 원 라이브러리 값으로 모델 간 절대값 비교를 하지 않음. |
| `best_iteration` | `Optional[int]` | 조기 종료의 최적 라운드. 0부터 시작. 조기 종료를 사용하지 않으면 None. |
| `early_stopping_train_indices` | `Optional[numpy.ndarray]` | 최종 모델의 내부 적합 행 위치. 원 입력 기준. 조기 종료 미사용 시 None. |
| `early_stopping_validation_indices` | `Optional[numpy.ndarray]` | 최종 모델의 내부 검증 행 위치. 원 입력 기준. 보류 테스트와 겹치지 않음. 조기 종료 미사용 시 None. |

### catboost

분류·회귀; 현재 수치 입력만 지원.

파일: [`src/MachineLearning/Trees/catboost.py`](../src/MachineLearning/Trees/catboost.py)

Profile: `CatBoostProfile` · DTO: `CatBoostOutputDTO`
레시피: `catboost-classification.json`, `catboost-regression.json`

```python
run_catboost(X: Any, y: Any, profile: Optional[src.MachineLearning.Trees.catboost.CatBoostProfile] = None, *, recipe_path: Optional[str] = None, task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Trees.catboost.CatBoostOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. 기본 classification. classification/regression 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'none'` | 스케일링 방법. none/standard/minmax; 기본 none. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고. |
| `iterations` | `int` | `100` | 부스팅 반복 수. 기본 100, 1 이상 정수. |
| `depth` | `int` | `6` | 대칭 트리 깊이. 기본 6, CPU 모듈에서 1~16. |
| `learning_rate` | `float` | `0.1` | 학습률. 기본 0.1, 유한한 양수. |
| `l2_leaf_reg` | `float` | `3.0` | 리프 L2 규제 강도. 기본 3.0, 유한한 0 이상 값. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. |
| `feature_importances` | `Optional[numpy.ndarray]` | 적합 모델의 입력 열 순서 특성 중요도 (n_features,). 원 라이브러리 값으로 모델 간 절대값 비교를 하지 않음. |

### lightgbm

분류·회귀.

파일: [`src/MachineLearning/Trees/lightgbm.py`](../src/MachineLearning/Trees/lightgbm.py)

Profile: `LightGBMProfile` · DTO: `LightGBMOutputDTO`
레시피: `lightgbm-classification.json`, `lightgbm-regression.json`

```python
run_lightgbm(X: Any, y: Any, profile: Optional[src.MachineLearning.Trees.lightgbm.LightGBMProfile] = None, *, recipe_path: Optional[str] = None, task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Trees.lightgbm.LightGBMOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. 기본 classification. classification/regression 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'none'` | 스케일링 방법. none/standard/minmax; 기본 none. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고. |
| `n_estimators` | `int` | `100` | 부스팅 단계 수. 기본 100, 1 이상 정수. |
| `num_leaves` | `int` | `31` | 트리별 최대 리프 수. 기본 31, 2 이상 정수. |
| `learning_rate` | `float` | `0.1` | 학습률. 기본 0.1, 유한한 양수. |
| `max_depth` | `int` | `-1` | 최대 깊이. 기본 -1은 제한 없음, 이외에는 양수. |
| `min_child_samples` | `int` | `20` | 리프 최소 관측 수. 기본 20, 1 이상 정수. |
| `subsample` | `float` | `1.0` | 행 표본 비율. 기본 1.0, (0,1]. |
| `subsample_freq` | `int` | `0` | 행 샘플링 주기. 기본 0은 비활성; subsample<1이면 1 이상 필요. |
| `colsample_bytree` | `float` | `1.0` | 특성 표본 비율. 기본 1.0, (0,1]. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. |
| `feature_importances` | `Optional[numpy.ndarray]` | 적합 모델의 입력 열 순서 특성 중요도 (n_features,). 원 라이브러리 값으로 모델 간 절대값 비교를 하지 않음. |

### adaboost

분류·회귀; 의사결정나무 기반.

파일: [`src/MachineLearning/Trees/adaboost.py`](../src/MachineLearning/Trees/adaboost.py)

Profile: `AdaBoostProfile` · DTO: `AdaBoostOutputDTO`
레시피: `adaboost-classification.json`, `adaboost-regression.json`

```python
run_adaboost(X: Any, y: Any, profile: Optional[src.MachineLearning.Trees.adaboost.AdaBoostProfile] = None, *, recipe_path: Optional[str] = None, task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Trees.adaboost.AdaBoostOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. 기본 classification. classification/regression 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'none'` | 스케일링 방법. none/standard/minmax; 기본 none. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고. |
| `n_estimators` | `int` | `50` | 약한 학습기 최대 개수. 기본 50, 1 이상 정수. |
| `learning_rate` | `float` | `1.0` | 약한 학습기 기여도. 기본 1.0, 유한한 양수. |
| `max_depth` | `int` | `1` | 기본 결정트리 깊이. 기본 1, 1 이상 정수. |
| `loss` | `str` | `'linear'` | 회귀 손실. linear/square/exponential; 기본 linear. 분류에서는 사용하지 않음. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. |
| `feature_importances` | `Optional[numpy.ndarray]` | 적합 모델의 입력 열 순서 특성 중요도 (n_features,). 원 라이브러리 값으로 모델 간 절대값 비교를 하지 않음. |

### mlp

분류·회귀; 다층 퍼셉트론.

파일: [`src/MachineLearning/NeuralNetworks/mlp.py`](../src/MachineLearning/NeuralNetworks/mlp.py)

Profile: `MLPProfile` · DTO: `MLPOutputDTO`
레시피: `mlp-classification.json`, `mlp-regression.json`

```python
run_mlp(X: Any, y: Any, profile: Optional[src.MachineLearning.NeuralNetworks.mlp.MLPProfile] = None, *, recipe_path: Optional[str] = None, task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.NeuralNetworks.mlp.MLPOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. 기본 classification. classification/regression 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'standard'` | 스케일링 방법. none/standard/minmax; 기본 standard. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고. |
| `hidden_layer_sizes` | `Tuple[int, ...]` | `(20, 10)` | 은닉층별 뉴런 수. 기본 (20,10), 비어 있지 않은 양수 정수 튜플. |
| `solver` | `str` | `'adam'` | 최적화 방법. adam/sgd/lbfgs; 기본 adam. |
| `activation` | `str` | `'relu'` | 은닉층 활성화. identity/logistic/tanh/relu; 기본 relu. |
| `alpha` | `float` | `0.0001` | L2 규제 강도. 기본 0.0001, 유한한 0 이상 값. |
| `learning_rate_init` | `float` | `0.001` | sgd/adam 초기 학습률. 기본 0.001, 유한한 양수. lbfgs에서는 사용하지 않음. |
| `max_iter` | `int` | `1000` | 최대 반복 수. 기본 1000, 1 이상 정수. |
| `early_stopping` | `bool` | `False` | 학습 구간 내부 검증 분할로 조기 종료. 기본 False; lbfgs에서는 지원하지 않음. |
| `validation_fraction` | `float` | `0.1` | 조기 종료 시 학습 구간 내부 검증 비율. 기본 0.1, (0,1). |
| `n_iter_no_change` | `int` | `10` | sgd/adam 개선 없는 반복 허용 수. 기본 10, 1 이상 정수. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. |
| `weights` | `Optional[List[numpy.ndarray]]` | 층별 가중치 행렬. 입력층부터 출력층 순서; 스케일러 변환 좌표계. |
| `intercepts` | `Optional[List[numpy.ndarray]]` | 층별 절편 벡터. weights와 같은 층 순서. |
| `loss_curve` | `Optional[numpy.ndarray]` | sgd/adam 반복별 학습 손실. lbfgs에서는 None. |
| `n_iter` | `Optional[int]` | 실제 최적화 반복 수. 수렴 여부 자체를 뜻하지 않음. |

### perceptron

분류; 단층 선형 모델, 확률 없음.

파일: [`src/MachineLearning/NeuralNetworks/perceptron.py`](../src/MachineLearning/NeuralNetworks/perceptron.py)

Profile: `PerceptronProfile` · DTO: `PerceptronOutputDTO`
레시피: `perceptron-classification.json`

```python
run_perceptron(X: Any, y: Any, profile: Optional[src.MachineLearning.NeuralNetworks.perceptron.PerceptronProfile] = None, *, recipe_path: Optional[str] = None, task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.NeuralNetworks.perceptron.PerceptronOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. 기본 classification. classification만 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'standard'` | 스케일링 방법. none/standard/minmax; 기본 standard. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고. |
| `max_iter` | `int` | `1000` | 최대 학습 epoch 수. 기본 1000, 1 이상 정수. |
| `tol` | `Optional[float]` | `0.001` | 수렴 허용오차. 기본 0.001, 양수 또는 None. None이면 max_iter까지 학습. |
| `eta0` | `float` | `1.0` | 상수 학습률. 기본 1.0, 유한한 양수. |
| `penalty` | `Optional[str]` | `None` | 규제 종류. None/l1/l2/elasticnet; 기본 None. |
| `alpha` | `float` | `0.0001` | 규제 강도. 기본 0.0001, 유한한 0 이상 값. |
| `fit_intercept` | `bool` | `True` | 절편 적합 여부. 기본 True. |
| `shuffle` | `bool` | `True` | epoch마다 학습 관측 순서 섞기. 기본 True. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. |
| `coefficients` | `Optional[numpy.ndarray]` | 선형 가중치 (1,n_features) 또는 (n_classes,n_features). 스케일러 변환 좌표계. |
| `intercepts` | `Optional[numpy.ndarray]` | 클래스별 절편. 이진 분류는 한 개. |
| `n_iter` | `Optional[int]` | 실제 학습 epoch 수. max_iter 도달 자체로 수렴을 보장하지 않음. |

### adaline

이진 분류; 선형 활성화의 배치 평균제곱 손실.

파일: [`src/MachineLearning/NeuralNetworks/adaline.py`](../src/MachineLearning/NeuralNetworks/adaline.py)

Profile: `AdalineProfile` · DTO: `AdalineOutputDTO`
레시피: `adaline-classification.json`

```python
run_adaline(X: Any, y: Any, profile: Optional[src.MachineLearning.NeuralNetworks.adaline.AdalineProfile] = None, *, recipe_path: Optional[str] = None, task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.NeuralNetworks.adaline.AdalineOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. 기본 classification. 이진 classification만 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV 난수 시드. 모델은 영벡터로 초기화. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'standard'` | 스케일링 방법. none/standard/minmax; 기본 standard. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고. |
| `max_iter` | `int` | `1000` | 최대 배치 학습 epoch 수. 기본 1000, 1 이상 정수. |
| `eta0` | `float` | `0.01` | 평균 제곱 손실의 상수 학습률. 기본 0.01, 유한한 양수. |
| `tol` | `Optional[float]` | `1e-06` | 연속 epoch의 손실 감소가 이 값 이하이면 종료. 기본 0.000001, 양수 또는 None. |
| `fit_intercept` | `bool` | `True` | 절편 학습 여부. 기본 True. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 accuracy·weighted precision/recall/F1. 일반화 성능을 뜻하지 않음. |
| `test_metrics` | `Dict[str, float]` | 보류 accuracy·weighted precision/recall/F1 (0~1). 확률 모델이 아니므로 공통 실행기는 ROC-AUC를 추가하지 않음. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `classes` | `Optional[numpy.ndarray]` | 원래 클래스 정렬 순서 (2,). 학습 중 classes[0]→-1, classes[1]→+1. |
| `probabilities` | `Optional[numpy.ndarray]` | Adaline은 확률을 지원하지 않으므로 None. 선형 출력은 activation_scores로 반환. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. |
| `coefficients` | `Optional[numpy.ndarray]` | 선형 가중치 (1,n_features). 스케일러 변환 좌표계. |
| `intercepts` | `Optional[numpy.ndarray]` | 절편 (1,). fit_intercept=False이면 0. |
| `activation_scores` | `Optional[numpy.ndarray]` | 보류 자료의 연속 선형 출력 (n_test,). 확률이 아니며 0 이상이면 classes[1]. |
| `loss_curve` | `Optional[numpy.ndarray]` | 각 epoch 갱신 후 학습 손실 0.5*mean((z-target)²). target은 -1/+1. |
| `n_iter` | `Optional[int]` | 실제 학습 epoch 수. loss_curve 길이와 동일. |
| `converged` | `Optional[bool]` | tol 기준 종료 여부. None 설정 또는 기준 미충족 시 False. |

### linear-regression

단순·다중 OLS / positive=True일 때 NNLS.

파일: [`src/MachineLearning/LinearModels/linear_regression.py`](../src/MachineLearning/LinearModels/linear_regression.py)

Profile: `LinearRegressionProfile` · DTO: `LinearRegressionOutputDTO`
레시피: `linear-regression-regression.json`

```python
run_linear_regression(X: Any, y: Any, profile: Optional[src.MachineLearning.LinearModels.linear_regression.LinearRegressionProfile] = None, *, recipe_path: Optional[str] = None, task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.LinearModels.linear_regression.LinearRegressionOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'regression'` | 분석 목적. 기본 regression. regression만 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV 난수 시드. OLS 적합 자체는 난수를 사용하지 않음. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'none'` | 스케일링 방법. none/standard/minmax; 기본 none. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고. |
| `fit_intercept` | `bool` | `True` | 절편 적합 여부. 기본 True. False이면 절편은 0. |
| `copy_X` | `bool` | `True` | 모델 적합 시 입력 복사 여부. 기본 True. |
| `positive` | `bool` | `False` | 계수를 0 이상으로 제한할지 여부. 기본 False. True이면 NNLS 사용. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 MAE/MSE/RMSE/R². 일반화 성능을 뜻하지 않음. |
| `test_metrics` | `Dict[str, float]` | 보류 MAE/MSE/RMSE/R². MAE/RMSE는 y의 단위, MSE는 제곱 단위. R²는 음수일 수 있음. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `classes` | `Optional[numpy.ndarray]` | 선형회귀에서는 None. |
| `probabilities` | `Optional[numpy.ndarray]` | 회귀에서는 None. 선형회귀는 확률 분류를 지원하지 않음. |
| `confusion` | `Optional[numpy.ndarray]` | 선형회귀에서는 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. |
| `coefficients` | `Optional[numpy.ndarray]` | 특성별 회귀 계수 (n_features,). 스케일러 변환 좌표계. |
| `intercept` | `Optional[float]` | 회귀 절편. fit_intercept=False이면 0.0. |
| `residuals` | `Optional[numpy.ndarray]` | 보류 잔차 y_test-predictions (n_test,). y와 같은 단위. |
| `rank` | `Optional[int]` | 적합 설계 행렬의 rank. 절편 적합 시 중심화한 행렬 기준. positive=True이면 None. |
| `singular_values` | `Optional[numpy.ndarray]` | 적합 설계 행렬의 특이값. positive=True이면 None. |

### naive-bayes

Gaussian / Multinomial / Bernoulli / Complement 분류.

파일: [`src/MachineLearning/NaiveBayes/naive_bayes.py`](../src/MachineLearning/NaiveBayes/naive_bayes.py)

Profile: `NaiveBayesProfile` · DTO: `NaiveBayesOutputDTO`
레시피: `naive-bayes-classification.json`

```python
run_naive_bayes(X: Any, y: Any, profile: Optional[src.MachineLearning.NaiveBayes.naive_bayes.NaiveBayesProfile] = None, *, task: Optional[str] = None, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.NaiveBayes.naive_bayes.NaiveBayesOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. 기본 classification. classification만 지원. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'none'` | 스케일링 방법. none/standard/minmax; 기본 none. 빈도·Bernoulli 모델은 none만 지원. 학습 폴드 안에서만 적합. 기본 "none". |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 후보 평가/CV의 작업 수. 선택용 선형 모델은 별도 스레드 설정을 지정하지 않음. |
| `variant` | `str` | `'gaussian'` | gaussian/multinomial/bernoulli/complement. 기본 "gaussian". |
| `alpha` | `float` | `1.0` | 가산 평활화 강도. 유한한 양수. Gaussian에는 미사용. 기본 1.0. |
| `var_smoothing` | `float` | `1e-09` | Gaussian 분산 안정화 강도. 양수. 기본 1e-9. |
| `binarize` | `Optional[float]` | `0.0` | Bernoulli 이진화 임계값. 유한한 수 또는 None. None이면 입력은 0/1이어야 함. 기본 0.0. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `class_counts` | `np.ndarray` | 클래스별 학습 관측 수. classes 순서. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. 기본 None. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. 기본 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. 기본 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. 기본 field(default_factory=dict). |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. 기본 None. |

### kmeans

고정 k 군집; 중심·inertia·silhouette.

파일: [`src/MachineLearning/Clustering/kmeans.py`](../src/MachineLearning/Clustering/kmeans.py)

Profile: `KMeansProfile` · DTO: `KMeansOutputDTO`
레시피: `kmeans.json`

```python
run_kmeans(X: Any, profile: Optional[src.MachineLearning.Clustering.kmeans.KMeansProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Clustering.kmeans.KMeansOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `random_state` | `int` | `1234` | 재현용 0 이상 정수 시드. 기본 1234. |
| `n_clusters` | `int` | `3` | 요청 군집 수. 2 이상이며 서로 다른 입력 행 수 이하. 기본 3. |
| `n_init` | `int` | `10` | 초기화 반복 횟수. 양의 정수. 기본 10. |
| `max_iter` | `int` | `300` | 최대 반복 횟수. 양의 정수. 기본 300. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `labels` | `np.ndarray` | (n_samples,) 군집 라벨. 밀도 군집의 -1은 잡음. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `n_clusters` | `int` | 잡음 -1을 제외한 실제 군집 수. |
| `noise_mask` | `np.ndarray` | (n_samples,) 잡음 여부. |
| `silhouette` | `Optional[float]` | 잡음을 제외한 silhouette. 두 군집 미만 또는 각 행이 다른 군집이면 None. |
| `centers` | `np.ndarray` | (n_clusters, n_features) 스케일링된 좌표계의 군집 중심. |
| `inertia` | `float` | 군집 중심까지 거리 제곱 합. |
| `n_iter` | `int` | 실제 반복 횟수. |

### dbscan

eps 밀도 군집; 핵심점·잡음(-1).

파일: [`src/MachineLearning/Clustering/dbscan.py`](../src/MachineLearning/Clustering/dbscan.py)

Profile: `DBSCANProfile` · DTO: `DBSCANOutputDTO`
레시피: `dbscan.json`

```python
run_dbscan(X: Any, profile: Optional[src.MachineLearning.Clustering.dbscan.DBSCANProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Clustering.dbscan.DBSCANOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `eps` | `float` | `0.5` | 이웃 반경. 유한한 양수. 기본 0.5. |
| `min_samples` | `int` | `5` | 핵심점 최소 이웃 수. 자기 자신 포함. 2 이상. 기본 5. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `labels` | `np.ndarray` | (n_samples,) 군집 라벨. 밀도 군집의 -1은 잡음. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `n_clusters` | `int` | 잡음 -1을 제외한 실제 군집 수. |
| `noise_mask` | `np.ndarray` | (n_samples,) 잡음 여부. |
| `silhouette` | `Optional[float]` | 잡음을 제외한 silhouette. 두 군집 미만 또는 각 행이 다른 군집이면 None. |
| `core_indices` | `np.ndarray` | 원 입력 기준 핵심점 행 위치. |

### hdbscan

계층 밀도 군집; 소속 강도·GLOSH 이상 점수.

파일: [`src/MachineLearning/Clustering/hdbscan.py`](../src/MachineLearning/Clustering/hdbscan.py)

Profile: `HDBSCANProfile` · DTO: `HDBSCANOutputDTO`
레시피: `hdbscan.json`

```python
run_hdbscan(X: Any, profile: Optional[src.MachineLearning.Clustering.hdbscan.HDBSCANProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Clustering.hdbscan.HDBSCANOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `min_cluster_size` | `int` | `5` | 최소 군집 크기. 2 이상 표본 수 이하. 기본 5. |
| `min_samples` | `Optional[int]` | `None` | 핵심 거리 이웃 수. None이면 min_cluster_size 사용. 1 이상 표본 수 미만. 기본 None. |
| `cluster_selection_method` | `str` | `'eom'` | 군집 선택 eom/leaf. 기본 "eom". |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `labels` | `np.ndarray` | (n_samples,) 군집 라벨. 밀도 군집의 -1은 잡음. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `n_clusters` | `int` | 잡음 -1을 제외한 실제 군집 수. |
| `noise_mask` | `np.ndarray` | (n_samples,) 잡음 여부. |
| `silhouette` | `Optional[float]` | 잡음을 제외한 silhouette. 두 군집 미만 또는 각 행이 다른 군집이면 None. |
| `probabilities` | `np.ndarray` | (n_samples,) 군집 소속 강도. 잡음은 0. |
| `outlier_scores` | `np.ndarray` | (n_samples,) GLOSH 이상 점수. 일반화 성능 지표 아님. |

### k-distance

자기 자신 포함 k번째 이웃 거리와 정렬 곡선; eps 후보 진단.

파일: [`src/MachineLearning/Clustering/k_distance.py`](../src/MachineLearning/Clustering/k_distance.py)

Profile: `KDistanceProfile` · DTO: `KDistanceOutputDTO`
레시피: `k-distance.json`

```python
run_k_distance(X: Any, profile: Optional[src.MachineLearning.Clustering.k_distance.KDistanceProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Clustering.k_distance.KDistanceOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `k` | `int` | `5` | 조회 이웃 수. 자기 자신 포함하므로 DBSCAN min_samples와 같은 정의. 2 이상 표본 수 이하. 기본 5. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `distances` | `np.ndarray` | (n_samples,) 자기 자신 포함 k번째 이웃까지 거리. |
| `sorted_indices` | `np.ndarray` | 거리 오름차순 정렬의 원 행 위치. |
| `sorted_distances` | `np.ndarray` | 오름차순 거리. 곡선의 굽은 부분을 eps 후보로 검토. |

### pca

선형 비지도 축소; 주성분·분산 설명·재구성 오차.

파일: [`src/MachineLearning/Decomposition/pca.py`](../src/MachineLearning/Decomposition/pca.py)

Profile: `PCAProfile` · DTO: `PCAOutputDTO`
레시피: `pca.json`

```python
run_pca(X: Any, profile: Optional[src.MachineLearning.Decomposition.pca.PCAProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Decomposition.pca.PCAOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `random_state` | `int` | `1234` | 재현용 0 이상 정수 시드. 기본 1234. |
| `n_components` | `int` | `2` | 출력 축 수. 알고리즘별 표본·특성·클래스 제약을 만족해야 함. 기본 2. |
| `whiten` | `bool` | `False` | 출력 분산 정규화 여부. bool. 기본 False. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `embedding` | `np.ndarray` | (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `supports_transform` | `bool` | 새 자료의 model.transform 지원 여부. False이면 재적합 필요. |
| `components` | `np.ndarray` | (n_components, n_features) 주성분 축. 스케일링 좌표계. |
| `explained_variance_ratio` | `np.ndarray` | 축별 분산 설명 비율. |
| `reconstruction_error` | `float` | 전처리된 X와 inverse_transform 좌표의 평균 제곱 차이. |

### lda

지도 판별 축소; y 필요; 최대 min(p, 클래스 수-1)축.

파일: [`src/MachineLearning/Decomposition/lda.py`](../src/MachineLearning/Decomposition/lda.py)

Profile: `LDAProfile` · DTO: `LDAOutputDTO`
레시피: `lda.json`

```python
run_lda(X: Any, y: Any, profile: Optional[src.MachineLearning.Decomposition.lda.LDAProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Decomposition.lda.LDAOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `random_state` | `int` | `1234` | 재현용 0 이상 정수 시드. 기본 1234. |
| `n_components` | `int` | `2` | 출력 축 수. 알고리즘별 표본·특성·클래스 제약을 만족해야 함. 기본 2. |
| `tol` | `float` | `0.0001` | SVD rank 추정의 양의 허용오차. 기본 0.0001. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `embedding` | `np.ndarray` | (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `supports_transform` | `bool` | 새 자료의 model.transform 지원 여부. False이면 재적합 필요. |
| `classes` | `np.ndarray` | 학습에 사용한 클래스 순서. |
| `explained_variance_ratio` | `np.ndarray` | 판별 축별 분산 설명 비율. |

### umap

비선형 비지도 축소; 새 입력 transform.

파일: [`src/MachineLearning/Decomposition/umap.py`](../src/MachineLearning/Decomposition/umap.py)

Profile: `UMAPProfile` · DTO: `UMAPOutputDTO`
레시피: `umap.json`

```python
run_umap(X: Any, profile: Optional[src.MachineLearning.Decomposition.umap.UMAPProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Decomposition.umap.UMAPOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `random_state` | `int` | `1234` | 재현용 0 이상 정수 시드. 기본 1234. |
| `n_components` | `int` | `2` | 출력 축 수. 알고리즘별 표본·특성·클래스 제약을 만족해야 함. 기본 2. |
| `n_neighbors` | `int` | `15` | 근방 크기. 2 이상 표본 수 미만. 기본 15. |
| `min_dist` | `float` | `0.1` | 저차원 최소 거리. 0 이상 spread 이하. 기본 0.1. |
| `spread` | `float` | `1.0` | 저차원 거리 규모. 유한한 양수. 기본 1.0. |
| `n_epochs` | `int` | `200` | 최적화 epoch 수. 양의 정수. 기본 200. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `embedding` | `np.ndarray` | (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `supports_transform` | `bool` | 새 자료의 model.transform 지원 여부. False이면 재적합 필요. |

### tsne

t-SNE; 시각화용 전 자료 적합; 새 입력 transform 없음.

파일: [`src/MachineLearning/Decomposition/tsne.py`](../src/MachineLearning/Decomposition/tsne.py)

Profile: `TSNEProfile` · DTO: `TSNEOutputDTO`
레시피: `tsne.json`

```python
run_tsne(X: Any, profile: Optional[src.MachineLearning.Decomposition.tsne.TSNEProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Decomposition.tsne.TSNEOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `random_state` | `int` | `1234` | 재현용 0 이상 정수 시드. 기본 1234. |
| `n_components` | `int` | `2` | 출력 축 수. 알고리즘별 표본·특성·클래스 제약을 만족해야 함. 기본 2. |
| `perplexity` | `float` | `30.0` | 유효 근방 크기. 유한한 양수이며 표본 수 미만. 기본 30.0. |
| `learning_rate` | `float` | `200.0` | 최적화 학습률. 유한한 양수. 기본 200.0. |
| `max_iter` | `int` | `1000` | 최대 반복. 250 이상. 기본 1000. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `embedding` | `np.ndarray` | (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `supports_transform` | `bool` | 새 자료의 model.transform 지원 여부. False이면 재적합 필요. |
| `kl_divergence` | `float` | 적합 시 KL 발산. 다른 perplexity의 좌표와 직접 성능 비교 불가. |

### som

온라인 Kohonen SOM; 2차원 BMU 격자·코드북·양자화/위상 오차.

파일: [`src/MachineLearning/Decomposition/som.py`](../src/MachineLearning/Decomposition/som.py)

Profile: `SOMProfile` · DTO: `SOMOutputDTO`
레시피: `som.json`

```python
run_som(X: Any, profile: Optional[src.MachineLearning.Decomposition.som.SOMProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Decomposition.som.SOMOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `random_state` | `int` | `1234` | 재현용 0 이상 정수 시드. 기본 1234. |
| `rows` | `int` | `5` | 격자 행 수. 양의 정수. 기본 5. |
| `columns` | `int` | `5` | 격자 열 수. 양의 정수. 기본 5. |
| `n_iter` | `int` | `1000` | 온라인 갱신 횟수. 양의 정수. 기본 1000. |
| `learning_rate` | `float` | `0.5` | 초기 학습률. 0 초과 1 이하. 기본 0.5. |
| `sigma` | `float` | `1.5` | 초기 이웃 폭. 유한한 양수. 기본 1.5. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `embedding` | `np.ndarray` | (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `supports_transform` | `bool` | 새 자료의 model.transform 지원 여부. False이면 재적합 필요. |
| `weights` | `np.ndarray` | (rows, columns, n_features) 코드북. 스케일링 좌표계. |
| `quantization_error` | `float` | 입력과 BMU 코드북 사이 평균 유클리드 거리. |
| `topographic_error` | `float` | 최상위 두 BMU가 4-이웃으로 인접하지 않은 관측 비율. 1개 노드이면 0. |

### mds

metric/nonmetric SMACOF; 유클리드 입력; 새 입력 transform 없음.

파일: [`src/MachineLearning/Decomposition/mds.py`](../src/MachineLearning/Decomposition/mds.py)

Profile: `MDSProfile` · DTO: `MDSOutputDTO`
레시피: `mds.json`

```python
run_mds(X: Any, profile: Optional[src.MachineLearning.Decomposition.mds.MDSProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.Decomposition.mds.MDSOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `scaling` | `str` | `'standard'` | 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard". |
| `random_state` | `int` | `1234` | 재현용 0 이상 정수 시드. 기본 1234. |
| `n_components` | `int` | `2` | 출력 축 수. 알고리즘별 표본·특성·클래스 제약을 만족해야 함. 기본 2. |
| `metric` | `bool` | `True` | True는 metric SMACOF, False는 nonmetric SMACOF. 기본 True. |
| `n_init` | `int` | `4` | 초기화 반복. 양의 정수. 기본 4. |
| `max_iter` | `int` | `300` | 최대 반복. 양의 정수. 기본 300. |
| `eps` | `float` | `0.001` | 수렴 허용오차. 양수. 기본 0.001. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고. |
| `embedding` | `np.ndarray` | (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표. |
| `feature_names` | `List[str]` | 원 입력 열 순서의 이름. |
| `supports_transform` | `bool` | 새 자료의 model.transform 지원 여부. False이면 재적합 필요. |
| `stress` | `float` | 라이브러리의 raw SMACOF stress. 구버전에서는 최종 갱신 전 값일 수 있으며 nonmetric은 disparity 기준. |
| `n_iter` | `int` | 최적 초기화에서 실제 반복 횟수. |
| `distance_reconstruction_error` | `float` | 최종 좌표와 전처리 입력의 유클리드 거리 제곱 차이 합. nonmetric 목적 함수와는 다름. |

### wrapper-selection

Forward / Backward / CV 점수 기반 양방향 Stepwise; 분류·회귀.

파일: [`src/MachineLearning/FeatureSelection/wrapper.py`](../src/MachineLearning/FeatureSelection/wrapper.py)

Profile: `WrapperSelectionProfile` · DTO: `WrapperSelectionOutputDTO`
레시피: `wrapper-selection-classification.json`, `wrapper-selection-regression.json`

```python
run_wrapper_selection(X: Any, y: Any, profile: Optional[src.MachineLearning.FeatureSelection.wrapper.WrapperSelectionProfile] = None, *, task: Optional[str] = None, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.FeatureSelection.wrapper.WrapperSelectionOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. classification/regression 지원. 기본 'classification'. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'standard'` | 스케일링 방법. none/standard/minmax; 기본 standard. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 후보 평가/CV의 작업 수. 선택용 선형 모델은 별도 스레드 설정을 지정하지 않음. |
| `method` | `str` | `'forward'` | forward/backward/stepwise. Stepwise는 CV 점수 기반 양방향 개선 탐색. 기본 "forward". |
| `n_features_to_select` | `Optional[int]` | `2` | Forward/Backward의 목표 수. None이면 개선 기준 자동 종료. Stepwise에서는 최대 수. 기본 2. |
| `min_improvement` | `float` | `0.0001` | 자동/Stepwise 종료를 위한 최소 CV 점수 개선. 유한한 0 이상. 기본 0.0001. |
| `max_steps` | `int` | `100` | 최대 이동 횟수. 양의 정수. 한도 도달은 history와 stopped_reason 확인. 기본 100. |
| `C` | `float` | `1.0` | 분류 LogisticRegression의 역규제 강도. 양수. 회귀는 OLS 사용. 기본 1.0. |
| `max_iter` | `int` | `2000` | 분류 LogisticRegression 최대 반복. 양의 정수. 기본 2000. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `selector` | `Any` | 적합된 선택 변환기. 원 입력에서 선택 열을 transform. 기본 JSON 제외. |
| `selected_indices` | `np.ndarray` | 원 특성 순서의 선택 열 위치. |
| `selected_features` | `List[str]` | 선택 열 이름. 원 입력 열 순서. |
| `support_mask` | `np.ndarray` | (n_features,) 선택 여부. |
| `transformed_train` | `np.ndarray` | 선택 열의 원 좌표 학습 자료. train_indices 행 순서. |
| `transformed_test` | `np.ndarray` | 선택 열의 원 좌표 보류 자료. test_indices 행 순서. |
| `history` | `List[Dict[str, Any]]` | 선택 이동 기록. 열 위치·action·평균 CV 점수. |
| `selection_cv_score` | `float` | 최종 선택 집합의 학습 자료 내부 CV 점수. 보류 성능 아님. |
| `stopped_reason` | `str` | target/no_improvement/max_steps/no_candidates 중 종료 이유. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. 기본 None. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. 기본 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. 기본 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. 기본 field(default_factory=dict). |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. 기본 None. |

### regularized-selection

L1·L2; 분류 Logistic / 회귀 Lasso·Ridge 계수 기반 선택.

파일: [`src/MachineLearning/FeatureSelection/regularized.py`](../src/MachineLearning/FeatureSelection/regularized.py)

Profile: `RegularizedSelectionProfile` · DTO: `RegularizedSelectionOutputDTO`
레시피: `regularized-selection-classification.json`, `regularized-selection-regression.json`

```python
run_regularized_selection(X: Any, y: Any, profile: Optional[src.MachineLearning.FeatureSelection.regularized.RegularizedSelectionProfile] = None, *, task: Optional[str] = None, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.MachineLearning.FeatureSelection.regularized.RegularizedSelectionOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `task` | `str` | `'classification'` | 분석 목적. classification/regression 지원. 기본 'classification'. |
| `test_size` | `float` | `0.3` | 보류 테스트 비율. 기본 0.3, 0 < 값 < 1. |
| `random_state` | `int` | `1234` | 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수. |
| `scaling` | `str` | `'standard'` | 스케일링 방법. none/standard/minmax; 기본 standard. 학습 폴드 안에서만 적합. |
| `stratify` | `bool` | `True` | 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음. |
| `param_grid` | `Optional[Dict[str, Any]]` | `None` | 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용. |
| `cv` | `int` | `3` | CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요. |
| `scoring` | `Optional[str]` | `None` | sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error. |
| `n_jobs` | `int` | `1` | CV 병렬 작업 수. 기본 1, 0 이외 정수. 후보 평가/CV의 작업 수. 선택용 선형 모델은 별도 스레드 설정을 지정하지 않음. |
| `penalty` | `str` | `'l1'` | l1/l2. L2는 자동 희소화가 아니라 중요도 임계값 선택. 기본 "l1". |
| `alpha` | `float` | `0.01` | 회귀 Lasso/Ridge의 규제 강도. 유한한 양수. 기본 0.01. |
| `C` | `float` | `1.0` | 분류 LogisticRegression 역규제 강도. 유한한 양수. 기본 1.0. |
| `threshold` | `Optional[float]` | `None` | 계수 절대값 평균의 포함 하한. None이면 L1 1e-8, L2 평균 중요도. 양수. 기본 None. |
| `max_iter` | `int` | `5000` | 최대 반복 횟수. 양의 정수. 기본 5000. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model_name` | `str` | 적합된 추정기의 클래스 이름. |
| `task` | `str` | 실행한 분석 목적. classification 또는 regression. |
| `model` | `Any` | 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외. |
| `train_metrics` | `Dict[str, float]` | 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R². |
| `test_metrics` | `Dict[str, float]` | 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가. |
| `y_test` | `np.ndarray` | 테스트 정답 (n_test,). test_indices와 동일한 행 순서. |
| `predictions` | `np.ndarray` | 테스트 예측 (n_test,). y_test와 동일한 행 순서. |
| `train_indices` | `np.ndarray` | 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치. |
| `test_indices` | `np.ndarray` | 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음. |
| `feature_names` | `List[str]` | 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름. |
| `selector` | `Any` | 적합된 선택 변환기. 원 입력에서 선택 열을 transform. 기본 JSON 제외. |
| `selected_indices` | `np.ndarray` | 원 특성 순서의 선택 열 위치. |
| `selected_features` | `List[str]` | 선택 열 이름. 원 입력 열 순서. |
| `support_mask` | `np.ndarray` | (n_features,) 선택 여부. |
| `transformed_train` | `np.ndarray` | 선택 열의 원 좌표 학습 자료. train_indices 행 순서. |
| `transformed_test` | `np.ndarray` | 선택 열의 원 좌표 보류 자료. test_indices 행 순서. |
| `importances` | `np.ndarray` | (n_features,) 전처리 좌표계의 계수 절대값. 다중 클래스는 클래스 평균. |
| `coefficients` | `np.ndarray` | 선택용 모델의 계수. 회귀 (n_features,), 분류 (n_estimators,n_features). |
| `applied_threshold` | `float` | 실제 적용한 중요도 임계값. |
| `classes` | `Optional[numpy.ndarray]` | 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. 기본 None. |
| `probabilities` | `Optional[numpy.ndarray]` | 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. 기본 None. |
| `confusion` | `Optional[numpy.ndarray]` | classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. 기본 None. |
| `best_params` | `Dict[str, Any]` | CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. 기본 field(default_factory=dict). |
| `best_cv_score` | `Optional[float]` | 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음. |
| `cv_results` | `Optional[Dict[str, Any]]` | GridSearchCV 전체 결과. 탐색하지 않았으면 None. 기본 None. |

### apriori

빈발 항목집합; support·confidence·lift·leverage·conviction.

파일: [`src/Association/apriori.py`](../src/Association/apriori.py)

Profile: `AprioriProfile` · DTO: `AprioriOutputDTO`
레시피: `apriori.json`

```python
run_apriori(transactions: Any, profile: Optional[src.Association.apriori.AprioriProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.Association.apriori.AprioriOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `min_support` | `float` | `0.2` | 최소 지지도. 0 초과 1 이하. 기본 0.2. |
| `min_confidence` | `float` | `0.5` | 최소 신뢰도. 0 이상 1 이하. 기본 0.5. |
| `min_lift` | `float` | `1.0` | 최소 향상도. 유한한 0 이상. 기본 1.0. |
| `max_length` | `int` | `3` | 최대 항목집합 길이. 양의 정수. 기본 3. |
| `max_rules` | `int` | `100000` | 연관규칙 출력 한도. 양의 정수. 초과 시 오류. 기본 100000. |
| `max_candidates` | `int` | `100000` | 전체 후보 평가 한도. 초과 시 일부 결과 반환 대신 오류. 기본 100000. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `n_transactions` | `int` | 전체 거래 수. 빈 거래도 분모에 포함. |
| `itemsets` | `pandas.core.frame.DataFrame` | items(tuple), count, support, length 열. 지지도는 거래 비율. |
| `rules` | `pandas.core.frame.DataFrame` | antecedents, consequents, support, confidence, lift, leverage, conviction 열. |

### sna

유향·무향 연결망; 중심성·PageRank·커뮤니티·모듈성.

파일: [`src/Networks/sna.py`](../src/Networks/sna.py)

Profile: `SNAProfile` · DTO: `SNAOutputDTO`
레시피: `sna.json`

```python
run_sna(edges: Any, profile: Optional[src.Networks.sna.SNAProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.Networks.sna.SNAOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `directed` | `bool` | `False` | 유향 그래프 여부. False이면 무향. 기본 False. |
| `nodes` | `Optional[List[str]]` | `None` | 고립 노드를 포함할 선택적 문자열 노드 목록. None이면 연결에서 추출. 기본 None. |
| `pagerank_alpha` | `float` | `0.85` | PageRank 감쇠 계수. 0 초과 1 미만. 기본 0.85. |
| `pagerank_max_iter` | `int` | `1000` | PageRank 반복 한도. 양의 정수. 기본 1000. |
| `pagerank_tol` | `float` | `1e-06` | PageRank 수렴 허용오차. 양수. 기본 1e-6. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `model` | `Any` | 구성된 networkx Graph/DiGraph. 기본 JSON/사전에서 제외. 예측 모델이 아님. |
| `nodes` | `List[str]` | 노드 결과 행 순서. 지정 노드 후 연결의 최초 등장 순서. |
| `node_metrics` | `pandas.core.frame.DataFrame` | 노드별 degree, strength, degree_centrality, betweenness, closeness, pagerank, community. |
| `communities` | `List[List[str]]` | 무향 투영 그래프의 greedy modularity 커뮤니티. 고립점은 단독 커뮤니티. |
| `density` | `float` | 노드 대비 실제 연결 밀도. |
| `n_components` | `int` | 무향 연결 요소 또는 유향 weakly-connected 요소 수. |
| `modularity` | `float` | 무향 투영 그래프의 strength 가중 모듈성. 연결이 없으면 0. |

### oneway-anova

고전·Welch 일원 ANOVA.

파일: [`src/stats/tests/parametric/Average/Multivariate/OneWay.py`](../src/stats/tests/parametric/Average/Multivariate/OneWay.py)

Profile: `OneWayANOVAProfile` · DTO: `OneWayANOVAOutputDTO`
레시피: `oneway-anova.json`

```python
run_oneway_anova(groups: Any, profile: Optional[src.stats.tests.parametric.Average.Multivariate.OneWay.OneWayANOVAProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Average.Multivariate.OneWay.OneWayANOVAOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `equal_var` | `bool` | `True` | 등분산 가정 여부. 기본 True: 고전 ANOVA, False: Welch ANOVA. |
| `alpha` | `float` | `0.05` | 유의수준. 기본 0.05, 유한한 0<값<1. 기각 기준은 p_value<alpha. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `test_type` | `str` | 실행한 고전 또는 Welch 검정 이름. |
| `statistic` | `float` | F 검정 통계량. |
| `p_value` | `float` | 전체 집단 평균 동일성의 p-value. |
| `reject_h0` | `bool` | p_value<alpha 여부. 어느 집단 쌍인지는 사후검정으로 확인. |
| `num_groups` | `int` | 집단 수. 최소 2. |
| `group_means` | `List[float]` | 입력 집단 순서의 평균. |
| `group_stds` | `List[float]` | 입력 집단 순서의 표본 표준편차, ddof=1. |
| `group_ns` | `List[int]` | 입력 집단 순서의 표본 수. |
| `df_between` | `float` | 분자 자유도. k-1. |
| `df_within` | `float` | 분모 자유도. Welch에서는 Satterthwaite 근사. |
| `anova_table` | `pandas.core.frame.DataFrame` | 고전은 SS·df·MS·F·p 표, Welch는 근사 F·df·p만 제공. Welch의 SS·MS는 NaN. |

### twoway-anova

Type I·II·III; 교호작용 포함/제외.

파일: [`src/stats/tests/parametric/Average/Multivariate/TwoWay.py`](../src/stats/tests/parametric/Average/Multivariate/TwoWay.py)

Profile: `TwoWayANOVAProfile` · DTO: `TwoWayANOVAOutputDTO`
레시피: `twoway-anova.json`

```python
run_twoway_anova(y: Any, factor1: Any, factor2: Any, profile: Optional[src.stats.tests.parametric.Average.Multivariate.TwoWay.TwoWayANOVAProfile] = None, factor1_name: str = 'Factor1', factor2_name: str = 'Factor2', response_name: str = 'Response', *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Average.Multivariate.TwoWay.TwoWayANOVAOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `interaction` | `bool` | `True` | 교호작용 포함 여부. 기본 True. False는 가법 주효과 모델. |
| `ss_type` | `int` | `2` | 제곱합 타입 1/2/3. 기본 2. Type I는 요인 1→요인 2의 순차 검정. Sum 코딩 사용. |
| `alpha` | `float` | `0.05` | 유의수준. 기본 0.05, 유한한 0<값<1. 기각 기준은 p_value<alpha. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `test_type` | `str` | 교호작용·SS 타입을 포함한 이원배치 검정 이름. |
| `factor1_name` | `str` | 첫 번째 요인의 표시 이름. |
| `factor2_name` | `str` | 두 번째 요인의 표시 이름. |
| `f_factor1` | `float` | 요인 1 F 통계량. |
| `p_factor1` | `float` | 요인 1 p-value. |
| `reject_h0_factor1` | `bool` | 요인 1 p<alpha 여부. |
| `f_factor2` | `float` | 요인 2 F 통계량. |
| `p_factor2` | `float` | 요인 2 p-value. |
| `reject_h0_factor2` | `bool` | 요인 2 p<alpha 여부. |
| `f_interaction` | `Optional[float]` | 교호작용 F 통계량. interaction=False이면 None. |
| `p_interaction` | `Optional[float]` | 교호작용 p-value. interaction=False이면 None. |
| `reject_h0_interaction` | `Optional[bool]` | 교호작용 p<alpha 여부. interaction=False이면 None. |
| `anova_table` | `pandas.core.frame.DataFrame` | SS·df·F·p 표. 행은 요인·교호작용·잔차. Type III는 절편 검정 포함. |

### manova

일원 요인 MANOVA; Pillai·Wilks·Hotelling-Lawley·Roy.

파일: [`src/stats/tests/parametric/Average/Multivariate/MANOVA.py`](../src/stats/tests/parametric/Average/Multivariate/MANOVA.py)

Profile: `MANOVAProfile` · DTO: `MANOVAOutputDTO`
레시피: `manova.json`

```python
run_manova(Y: Any, groups: Any, profile: Optional[src.stats.tests.parametric.Average.Multivariate.MANOVA.MANOVAProfile] = None, response_names: Optional[List[str]] = None, group_name: str = 'Group', *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Average.Multivariate.MANOVA.MANOVAOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.05` | 유의수준. 기본 0.05, 유한한 0<값<1. 기각 기준은 p_value<alpha. |
| `criterion` | `str` | `'pillai'` | 기각 판단 지표. pillai/wilks/hotelling/roy. 기본 pillai. 모든 지표를 함께 반환. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `test_type` | `str` | MANOVA 검정 이름. |
| `criterion` | `str` | 선택한 기각 판단 지표. |
| `statistic` | `float` | 선택한 다변량 통계량 원 값. F 값과 다름. |
| `f_value` | `float` | 선택한 지표의 근사 F. |
| `p_value` | `float` | 선택한 지표의 전체 집단 공동 평균 동일성 p-value. |
| `reject_h0` | `bool` | 선택한 지표의 p_value<alpha 여부. |
| `df_num` | `float` | 선택한 F의 분자 자유도. |
| `df_den` | `float` | 선택한 F의 분모 자유도. |
| `group_name` | `str` | 요인 표시 이름. |
| `response_names` | `List[str]` | 입력 종속변수 열 순서의 이름. |
| `group_labels` | `List[str]` | 문자열 정렬 순서의 집단 라벨. |
| `group_ns` | `List[int]` | group_labels 순서의 표본 수. |
| `group_means` | `pandas.core.frame.DataFrame` | 행은 group_labels, 열은 response_names의 집단 평균. |
| `test_table` | `pandas.core.frame.DataFrame` | Wilks/Pillai/Hotelling-Lawley/Roy 전체 지표. value·df_num·df_den·statistic(F)·p_value·reject_h0 열. |

### posthoc

Tukey HSD / Student·Welch 쌍별 t + Holm·Bonferroni.

파일: [`src/stats/tests/parametric/Average/Multivariate/PostHoc.py`](../src/stats/tests/parametric/Average/Multivariate/PostHoc.py)

Profile: `PostHocProfile` · DTO: `PostHocOutputDTO`
레시피: `posthoc.json`

```python
run_posthoc(groups: Any, profile: Optional[src.stats.tests.parametric.Average.Multivariate.PostHoc.PostHocProfile] = None, group_names: Optional[List[str]] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Average.Multivariate.PostHoc.PostHocOutputDTO
```

입력 Profile:

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `method` | `str` | `'tukey'` | tukey/holm/bonferroni. 기본 tukey. Holm·Bonferroni는 모든 집단 쌍 t 검정 p-value를 보정. |
| `equal_var` | `bool` | `True` | t 검정 등분산 여부. 기본 True. False는 Welch t; Tukey에는 False 사용 불가. |
| `alpha` | `float` | `0.05` | 유의수준. 기본 0.05, 유한한 0<값<1. 기각 기준은 p_value<alpha. |

결과 Output DTO:

| 필드 | 타입 | 의미·배열 순서 |
| --- | --- | --- |
| `test_type` | `str` | 실행한 독립 집단 평균 사후검정 이름. |
| `method` | `str` | 다중검정 보정 방법. |
| `alpha` | `float` | 전체 쌍 비교 family의 유의수준. |
| `equal_var` | `bool` | 등분산 가정 여부. |
| `group_names` | `List[str]` | 입력 집단 순서의 표시 이름. |
| `group_means` | `List[float]` | 입력 순서의 집단 평균. |
| `group_ns` | `List[int]` | 입력 순서의 집단 표본 수. |
| `comparisons` | `pandas.core.frame.DataFrame` | 모든 쌍의 group1/group2, mean_diff(집단2-집단1), statistic(t 또는 NaN), df(t 자유도 또는 NaN), p_raw(Tukey는 NaN), p_adjusted, ci_lower/ci_upper, reject_h0. Tukey·Bonferroni CI는 family-wise, Holm CI는 NaN. |
| `num_comparisons` | `int` | 비교 쌍 수 k*(k-1)/2. |

## 기존 Z·t 검정 파일

다음 두 파일은 기존 구현을 그대로 유지합니다. Profile과 dataclass Result를 제공하지만 표준 OutputDTO 상속,
JSON 레시피 및 공개 `run_*` 실행기가 아직 없습니다. 통계 결과는 `result.p_value` 등 속성으로 조회합니다.

| 파일 | 기법 | 조건/결과 | 호출 형태 |
| --- | --- | --- | --- |
| `src/stats/tests/parametric/Average/Univariate.py` | 단일 표본 Z·t | `TestProfile` / `TestResult` | `_run_z_test` / `_run_t_test`; 평균·표준편차를 인자로 전달 |
| `src/stats/tests/parametric/Average/Bivariate.py` | 독립 두 표본 Z·Student t·Welch t | `TwoSampleTestProfile` / `TwoSampleTestResult` | `_run_two_sample_z_test` / `_run_two_sample_t_test`; 두 표본 요약값을 인자로 전달 |

Z 검정은 알려진 양의 모표준편차를 지정하고, t 검정은 미지의 모분산을 전제로 합니다.
대립가설은 two-sided/less/greater입니다. 기존 Z·t 계산은 SciPy 참조값과 비교 검증했습니다.
비모수 `onesample.py`, `twosample.py`는 빈 파일이며 미구현입니다.

### Univariate 기존 필드

`TestProfile`:

| 필드 | 타입 | 기본값 |
| --- | --- | --- |
| `mu_0` | `float` | `필수` |
| `known_var` | `bool` | `False` |
| `sigma` | `Optional[float]` | `None` |
| `alpha` | `float` | `0.05` |
| `alternative` | `Literal['two-sided', 'less', 'greater']` | `'two-sided'` |

`TestResult`:

| 필드 | 타입 | 기본값 |
| --- | --- | --- |
| `test_type` | `str` | `필수` |
| `statistic` | `float` | `필수` |
| `p_value` | `float` | `필수` |
| `reject_h0` | `bool` | `필수` |
| `sample_mean` | `float` | `필수` |
| `sample_std` | `float` | `필수` |
| `n` | `int` | `필수` |

### Bivariate 기존 필드

`TwoSampleTestProfile`:

| 필드 | 타입 | 기본값 |
| --- | --- | --- |
| `diff_0` | `float` | `0.0` |
| `known_var` | `bool` | `False` |
| `sigma1` | `Optional[float]` | `None` |
| `sigma2` | `Optional[float]` | `None` |
| `equal_var` | `bool` | `True` |
| `alpha` | `float` | `0.05` |
| `alternative` | `Literal['two-sided', 'less', 'greater']` | `'two-sided'` |

`TwoSampleTestResult`:

| 필드 | 타입 | 기본값 |
| --- | --- | --- |
| `test_type` | `str` | `필수` |
| `statistic` | `float` | `필수` |
| `p_value` | `float` | `필수` |
| `reject_h0` | `bool` | `필수` |
| `diff_mean` | `float` | `필수` |
| `sample_mean1` | `float` | `필수` |
| `sample_mean2` | `float` | `필수` |
| `sample_std1` | `float` | `필수` |
| `sample_std2` | `float` | `필수` |
| `n1` | `int` | `필수` |
| `n2` | `int` | `필수` |

## 실행 환경과 검증

- Python 3.7.17 / 3.12.14 의존성은 각각 `requirements-models-py37.txt`, `requirements-models-py312.txt`에 고정했습니다.
- 3.7 환경의 HDBSCAN 0.8.33은 Cython 0.29.37·wheel·NumPy를 먼저 설치하고 `--no-build-isolation`로 빌드합니다. C/C++ 컴파일러가 필요합니다.
- 3.12 환경은 HDBSCAN 0.8.41, UMAP 0.5.9.post2를 사용합니다. 3.7 환경은 UMAP 0.5.3과 해당 버전의 numba/llvmlite를 사용합니다.
- 새 모듈은 기존 소스의 Python 3.7 호환 문법을 유지합니다. 실제 시험 서버 및 Python 3.7.4에서 직접 검증하지는 않았습니다.
- 일반 분석 환경은 unittest 51개 중 TensorFlow 실제 학습 4개를 선택 의존성으로 건너뛰고 47개를 실행합니다. 일반 예제 33개를 실행하며, 별도 TensorFlow 환경에서는 6개 테스트와 4개 예제를 실행합니다. UMAP은 첫 실행에 Numba 컴파일 시간이 필요할 수 있습니다.
- t-SNE·MDS는 거리/최적화 계산량이 크고 Wrapper는 후보별 CV를 반복합니다. 대규모 자료에서는 표본 수·특성 수·반복 한도를 조절하세요.

```bash
# Python 3.7 환경
python -m pip install numpy==1.21.6 Cython==0.29.37 wheel==0.38.4
python -m pip install --no-build-isolation -r requirements-models-py37.txt

# Python 3.12 환경
python -m pip install -r requirements-models-py312.txt

python -m unittest discover -s tests -v
python -m src.MachineLearning.NaiveBayes.naive_bayes
python -m src.MachineLearning.Clustering.hdbscan
python -m src.MachineLearning.Decomposition.som
python -m src.MachineLearning.FeatureSelection.wrapper
python -m src.Association.apriori
python -m src.Networks.sna
```

공식 API 참고: [scikit-learn 변수 선택](https://scikit-learn.org/stable/modules/feature_selection.html),
[HDBSCAN](https://hdbscan.readthedocs.io/en/latest/api.html),
[UMAP](https://umap-learn.readthedocs.io/en/latest/api.html),
[NetworkX](https://networkx.org/documentation/stable/reference/index.html).


## 2026-10-12 추가 통계 검정·TensorFlow

검정 선택과 가정은 [통계 검정 안내](STATISTICAL_TESTS.md), TensorFlow 학습/타깃 정렬 규약은 [TensorFlow 안내](TENSORFLOW_MODULES.md)를 참고하세요.

### chi-square

- 모듈: `src.stats.tests.nonParametric.chi_square`
- 실행기: `run_chi_square`
- 레시피: `chi-square.json`

#### ChiSquareProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유의수준, 기본 0.05. |
| `ddof` | `<class 'int'>` | `0` | 추정 모수 수에 따른 자유도 감소, 기본 0. |

#### ChiSquareOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `statistic` | `<class 'float'>` | `반환 필드` | 검정 통계량. |
| `p_value` | `<class 'float'>` | `반환 필드` | 귀무가설 하의 p 값. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p_value < alpha. |
| `alpha` | `<class 'float'>` | `반환 필드` | 사용한 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `details` | `Dict[str, Any]` | `반환 필드` | 자유도·기대도수·표본수 등 검정별 진단. |
| `warnings` | `List[str]` | `반환 필드` | 근사 조건과 해석 제약. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### independence

- 모듈: `src.stats.tests.nonParametric.independence`
- 실행기: `run_independence`
- 레시피: `independence.json`

#### IndependenceProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유의수준, 기본 0.05. |
| `correction` | `<class 'bool'>` | `True` | 자유도 1일 때 Yates 보정, 기본 True. |

#### IndependenceOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `statistic` | `<class 'float'>` | `반환 필드` | 검정 통계량. |
| `p_value` | `<class 'float'>` | `반환 필드` | 귀무가설 하의 p 값. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p_value < alpha. |
| `alpha` | `<class 'float'>` | `반환 필드` | 사용한 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `details` | `Dict[str, Any]` | `반환 필드` | 자유도·기대도수·표본수 등 검정별 진단. |
| `warnings` | `List[str]` | `반환 필드` | 근사 조건과 해석 제약. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### mcnemar

- 모듈: `src.stats.tests.nonParametric.mcnemar`
- 실행기: `run_mcnemar`
- 레시피: `mcnemar.json`

#### McNemarProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유의수준, 기본 0.05. |
| `exact` | `<class 'bool'>` | `True` | 이항 정확 검정, 기본 True. |
| `correction` | `<class 'bool'>` | `True` | 근사 검정의 연속성 보정, exact에서는 무시. 기본 True. |

#### McNemarOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `statistic` | `<class 'float'>` | `반환 필드` | 검정 통계량. |
| `p_value` | `<class 'float'>` | `반환 필드` | 귀무가설 하의 p 값. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p_value < alpha. |
| `alpha` | `<class 'float'>` | `반환 필드` | 사용한 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `details` | `Dict[str, Any]` | `반환 필드` | 자유도·기대도수·표본수 등 검정별 진단. |
| `warnings` | `List[str]` | `반환 필드` | 근사 조건과 해석 제약. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### f-test

- 모듈: `src.stats.tests.parametric.Variance.f_test`
- 실행기: `run_f_test`
- 레시피: `f-test.json`

#### FTestProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유의수준, 기본 0.05. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater: var(x)/var(y) 방향. 기본 two-sided. |
| `ratio_0` | `<class 'float'>` | `1.0` | 귀무 분산비, 양수. 기본 1. |

#### FTestOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `statistic` | `<class 'float'>` | `반환 필드` | 검정 통계량. |
| `p_value` | `<class 'float'>` | `반환 필드` | 귀무가설 하의 p 값. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p_value < alpha. |
| `alpha` | `<class 'float'>` | `반환 필드` | 사용한 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `details` | `Dict[str, Any]` | `반환 필드` | 자유도·기대도수·표본수 등 검정별 진단. |
| `warnings` | `List[str]` | `반환 필드` | 근사 조건과 해석 제약. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### normality

- 모듈: `src.stats.tests.diagnostics.normality`
- 실행기: `run_normality`
- 레시피: `normality.json`

#### NormalityProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유의수준, 기본 0.05. |
| `method` | `<class 'str'>` | `'shapiro'` | shapiro/dagostino/jarque_bera. 기본 shapiro. |

#### NormalityOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `statistic` | `<class 'float'>` | `반환 필드` | 검정 통계량. |
| `p_value` | `<class 'float'>` | `반환 필드` | 귀무가설 하의 p 값. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p_value < alpha. |
| `alpha` | `<class 'float'>` | `반환 필드` | 사용한 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `details` | `Dict[str, Any]` | `반환 필드` | 자유도·기대도수·표본수 등 검정별 진단. |
| `warnings` | `List[str]` | `반환 필드` | 근사 조건과 해석 제약. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### homogeneity

- 모듈: `src.stats.tests.diagnostics.homogeneity`
- 실행기: `run_homogeneity`
- 레시피: `homogeneity.json`

#### HomogeneityProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유의수준, 기본 0.05. |
| `method` | `<class 'str'>` | `'levene'` | levene/bartlett/fligner. 기본 levene. |
| `center` | `<class 'str'>` | `'median'` | Levene/Fligner 중심 mean/median/trimmed. 기본 median. |

#### HomogeneityOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `statistic` | `<class 'float'>` | `반환 필드` | 검정 통계량. |
| `p_value` | `<class 'float'>` | `반환 필드` | 귀무가설 하의 p 값. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p_value < alpha. |
| `alpha` | `<class 'float'>` | `반환 필드` | 사용한 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `details` | `Dict[str, Any]` | `반환 필드` | 자유도·기대도수·표본수 등 검정별 진단. |
| `warnings` | `List[str]` | `반환 필드` | 근사 조건과 해석 제약. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### tf-classification

- 모듈: `src.DeepLearning.TensorFlow.classification`
- 실행기: `run_tf_classification`
- 레시피: `tf-classification.json`

#### TFClassificationProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `random_state` | `<class 'int'>` | `1234` | 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234. |
| `epochs` | `<class 'int'>` | `30` | 최대 학습 epoch, 양수. 기본 30. |
| `batch_size` | `<class 'int'>` | `32` | 학습 batch 크기, 양수. 기본 32. |
| `learning_rate` | `<class 'float'>` | `0.001` | Adam 학습률, 양수. 기본 0.001. |
| `patience` | `<class 'int'>` | `5` | validation loss 조기 종료 대기, 0 이상. 기본 5. |
| `test_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 test 비율. 기본 0.2. |
| `validation_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2. |
| `scaling` | `<class 'str'>` | `'standard'` | none/standard; 학습 입력만 적합. 기본 standard. |
| `verbose` | `<class 'int'>` | `0` | Keras 출력 0/1/2. 기본 0. |
| `hidden_layer_sizes` | `Tuple[int, ...]` | `(32, 16)` | 은닉 Dense 폭, 양수 정수 tuple. 빈 tuple은 선형 출력. 기본 (32,16). |

#### TFClassificationOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `task` | `<class 'str'>` | `반환 필드` | classification/regression. 시계열은 regression. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용한 Profile 설정. 레시피 및 overrides 반영. |
| `model` | `Any` | `반환 필드` | 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외. |
| `history` | `Dict[str, List[float]]` | `반환 필드` | epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록. |
| `test_metrics` | `Dict[str, float]` | `반환 필드` | 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위. |
| `predictions` | `<class 'numpy.ndarray'>` | `반환 필드` | test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target). |
| `probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 분류 (n,classes) 확률; 회귀 None. |
| `classes` | `Optional[numpy.ndarray]` | `반환 필드` | 확률 열 순서의 학습 클래스; 회귀 None. |
| `train_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 학습 표본 번호; 시계열은 원 시점 번호. |
| `validation_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 조기 종료 전용 검증 표본/시점 번호. |
| `test_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 보류 평가 표본/시점 번호. |
| `target_indices` | `Optional[numpy.ndarray]` | `반환 필드` | 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None. |
| `y_test` | `<class 'numpy.ndarray'>` | `반환 필드` | predictions와 같은 shape/순서의 실제 test 타깃. |
| `input_shape` | `Tuple[int, ...]` | `반환 필드` | Keras 입력 shape(batch 제외). |
| `epochs_trained` | `<class 'int'>` | `반환 필드` | 실제 실행한 epoch 수. |
| `best_epoch` | `<class 'int'>` | `반환 필드` | 최저 val_loss epoch, 1부터 시작. 최적 weights 복원. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### tf-regression

- 모듈: `src.DeepLearning.TensorFlow.regression`
- 실행기: `run_tf_regression`
- 레시피: `tf-regression.json`

#### TFRegressionProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `random_state` | `<class 'int'>` | `1234` | 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234. |
| `epochs` | `<class 'int'>` | `30` | 최대 학습 epoch, 양수. 기본 30. |
| `batch_size` | `<class 'int'>` | `32` | 학습 batch 크기, 양수. 기본 32. |
| `learning_rate` | `<class 'float'>` | `0.001` | Adam 학습률, 양수. 기본 0.001. |
| `patience` | `<class 'int'>` | `5` | validation loss 조기 종료 대기, 0 이상. 기본 5. |
| `test_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 test 비율. 기본 0.2. |
| `validation_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2. |
| `scaling` | `<class 'str'>` | `'standard'` | none/standard; 학습 입력만 적합. 기본 standard. |
| `verbose` | `<class 'int'>` | `0` | Keras 출력 0/1/2. 기본 0. |
| `hidden_layer_sizes` | `Tuple[int, ...]` | `(32, 16)` | 은닉 Dense 폭, 양수 정수 tuple. 빈 tuple은 선형 출력. 기본 (32,16). |

#### TFRegressionOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `task` | `<class 'str'>` | `반환 필드` | classification/regression. 시계열은 regression. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용한 Profile 설정. 레시피 및 overrides 반영. |
| `model` | `Any` | `반환 필드` | 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외. |
| `history` | `Dict[str, List[float]]` | `반환 필드` | epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록. |
| `test_metrics` | `Dict[str, float]` | `반환 필드` | 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위. |
| `predictions` | `<class 'numpy.ndarray'>` | `반환 필드` | test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target). |
| `probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 분류 (n,classes) 확률; 회귀 None. |
| `classes` | `Optional[numpy.ndarray]` | `반환 필드` | 확률 열 순서의 학습 클래스; 회귀 None. |
| `train_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 학습 표본 번호; 시계열은 원 시점 번호. |
| `validation_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 조기 종료 전용 검증 표본/시점 번호. |
| `test_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 보류 평가 표본/시점 번호. |
| `target_indices` | `Optional[numpy.ndarray]` | `반환 필드` | 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None. |
| `y_test` | `<class 'numpy.ndarray'>` | `반환 필드` | predictions와 같은 shape/순서의 실제 test 타깃. |
| `input_shape` | `Tuple[int, ...]` | `반환 필드` | Keras 입력 shape(batch 제외). |
| `epochs_trained` | `<class 'int'>` | `반환 필드` | 실제 실행한 epoch 수. |
| `best_epoch` | `<class 'int'>` | `반환 필드` | 최저 val_loss epoch, 1부터 시작. 최적 weights 복원. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### tf-rnn

- 모듈: `src.DeepLearning.TensorFlow.rnn`
- 실행기: `run_tf_rnn`
- 레시피: `tf-rnn-classification.json`, `tf-rnn-regression.json`

#### TFRNNProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `random_state` | `<class 'int'>` | `1234` | 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234. |
| `epochs` | `<class 'int'>` | `30` | 최대 학습 epoch, 양수. 기본 30. |
| `batch_size` | `<class 'int'>` | `32` | 학습 batch 크기, 양수. 기본 32. |
| `learning_rate` | `<class 'float'>` | `0.001` | Adam 학습률, 양수. 기본 0.001. |
| `patience` | `<class 'int'>` | `5` | validation loss 조기 종료 대기, 0 이상. 기본 5. |
| `test_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 test 비율. 기본 0.2. |
| `validation_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2. |
| `scaling` | `<class 'str'>` | `'standard'` | none/standard; 학습 입력만 적합. 기본 standard. |
| `verbose` | `<class 'int'>` | `0` | Keras 출력 0/1/2. 기본 0. |
| `task` | `<class 'str'>` | `'regression'` | classification/regression. 기본 regression. |
| `cell` | `<class 'str'>` | `'lstm'` | simple_rnn/lstm/gru 셀 선택. 기본 lstm. |
| `units` | `<class 'int'>` | `16` | 순환 층 폭, 양수. 기본 16. |

#### TFRNNOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `task` | `<class 'str'>` | `반환 필드` | classification/regression. 시계열은 regression. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용한 Profile 설정. 레시피 및 overrides 반영. |
| `model` | `Any` | `반환 필드` | 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외. |
| `history` | `Dict[str, List[float]]` | `반환 필드` | epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록. |
| `test_metrics` | `Dict[str, float]` | `반환 필드` | 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위. |
| `predictions` | `<class 'numpy.ndarray'>` | `반환 필드` | test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target). |
| `probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 분류 (n,classes) 확률; 회귀 None. |
| `classes` | `Optional[numpy.ndarray]` | `반환 필드` | 확률 열 순서의 학습 클래스; 회귀 None. |
| `train_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 학습 표본 번호; 시계열은 원 시점 번호. |
| `validation_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 조기 종료 전용 검증 표본/시점 번호. |
| `test_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 보류 평가 표본/시점 번호. |
| `target_indices` | `Optional[numpy.ndarray]` | `반환 필드` | 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None. |
| `y_test` | `<class 'numpy.ndarray'>` | `반환 필드` | predictions와 같은 shape/순서의 실제 test 타깃. |
| `input_shape` | `Tuple[int, ...]` | `반환 필드` | Keras 입력 shape(batch 제외). |
| `epochs_trained` | `<class 'int'>` | `반환 필드` | 실제 실행한 epoch 수. |
| `best_epoch` | `<class 'int'>` | `반환 필드` | 최저 val_loss epoch, 1부터 시작. 최적 weights 복원. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

### tf-time-series

- 모듈: `src.DeepLearning.TensorFlow.time_series`
- 실행기: `run_tf_time_series`
- 레시피: `tf-time-series.json`

#### TFTimeSeriesProfile

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `random_state` | `<class 'int'>` | `1234` | 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234. |
| `epochs` | `<class 'int'>` | `30` | 최대 학습 epoch, 양수. 기본 30. |
| `batch_size` | `<class 'int'>` | `32` | 학습 batch 크기, 양수. 기본 32. |
| `learning_rate` | `<class 'float'>` | `0.001` | Adam 학습률, 양수. 기본 0.001. |
| `patience` | `<class 'int'>` | `5` | validation loss 조기 종료 대기, 0 이상. 기본 5. |
| `test_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 test 비율. 기본 0.2. |
| `validation_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2. |
| `scaling` | `<class 'str'>` | `'standard'` | none/standard; 학습 입력만 적합. 기본 standard. |
| `verbose` | `<class 'int'>` | `0` | Keras 출력 0/1/2. 기본 0. |
| `cell` | `<class 'str'>` | `'lstm'` | simple_rnn/lstm/gru 셀 선택. 기본 lstm. |
| `units` | `<class 'int'>` | `16` | 순환 층 폭, 양수. 기본 16. |
| `input_width` | `<class 'int'>` | `12` | 관측 입력 창 길이. 기본 12. |
| `horizon` | `<class 'int'>` | `1` | 미래 예측 타임스텝 수. 기본 1. |
| `target_columns` | `Optional[Tuple[int, ...]]` | `None` | 예측할 열 위치; None은 모든 입력 열. 기본 None. |

#### TFTimeSeriesOutputDTO

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `task` | `<class 'str'>` | `반환 필드` | classification/regression. 시계열은 regression. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용한 Profile 설정. 레시피 및 overrides 반영. |
| `model` | `Any` | `반환 필드` | 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외. |
| `history` | `Dict[str, List[float]]` | `반환 필드` | epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록. |
| `test_metrics` | `Dict[str, float]` | `반환 필드` | 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위. |
| `predictions` | `<class 'numpy.ndarray'>` | `반환 필드` | test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target). |
| `probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 분류 (n,classes) 확률; 회귀 None. |
| `classes` | `Optional[numpy.ndarray]` | `반환 필드` | 확률 열 순서의 학습 클래스; 회귀 None. |
| `train_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 학습 표본 번호; 시계열은 원 시점 번호. |
| `validation_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 조기 종료 전용 검증 표본/시점 번호. |
| `test_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 보류 평가 표본/시점 번호. |
| `target_indices` | `Optional[numpy.ndarray]` | `반환 필드` | 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None. |
| `y_test` | `<class 'numpy.ndarray'>` | `반환 필드` | predictions와 같은 shape/순서의 실제 test 타깃. |
| `input_shape` | `Tuple[int, ...]` | `반환 필드` | Keras 입력 shape(batch 제외). |
| `epochs_trained` | `<class 'int'>` | `반환 필드` | 실제 실행한 epoch 수. |
| `best_epoch` | `<class 'int'>` | `반환 필드` | 최저 val_loss epoch, 1부터 시작. 최적 weights 복원. |

공개 실행기의 `profile`, `recipe_path`, `overrides`는 기존 레시피 우선순위를 따릅니다.

## 통계 PDF 전체 확장과 정식 경로

추가42개 공개모듈의 전체 API·필드·레시피·설계·계산경계는 [통계 전체 명세](STATISTICS_COMPLETE.md)에 있습니다.
ANOVA/사후검정은 parametric/Average/anova, MANOVA는 parametric/Multivariate, 카이제곱/McNemar는 nonParametric/Categorical이 정식위치입니다. 이전위치 링크는 호환export로 유효합니다. 기존 Z/t 설명은 이전 호환API이며 새표준실행기는 전체명세를 참고하세요.

## Feature 전처리·상관분석 확장

수치/범주형 Feature의 순서가 있는 전처리, 별도 RoPE 임베딩 회전, 수치/범주 상관분석과
상관행렬 변환을 추가했습니다. [전체 명세](PREPROCESSING_CORRELATION_MODULES.md)를 참고하세요.

## Axes 기반 EDA 시각화

`src/Visualization`의 35개 모듈은 단일 ax에 그려 subplot으로 조합합니다. 회귀 시각화는 제외하고
분포·결측·범주·군집·차원 축소·분류 결정·연관/연결망·시간 패턴을 포함합니다.
[전체 Profile/DTO·의사결정 근거·UCI 조합 예제](EDA_VISUALIZATION_MODULES.md)를 참고하세요.

## 독립 평가 메트릭

추가 평가 모듈 56개의 전체 Profile·Output DTO·레시피·수식·skeleton 경로는 [평가 전체 명세](EVALUATION_METRICS.md)에 있습니다.


## 교차검증 파라미터 최적화 (추가 3종)

| 모듈 | Profile | Output DTO | 실행 함수 |
| --- | --- | --- | --- |
| MachineLearning/Optimization/grid_search | GridSearchProfile | GridSearchOutputDTO | run_grid_search |
| MachineLearning/Optimization/random_search | RandomSearchProfile | RandomSearchOutputDTO | run_random_search |
| MachineLearning/Optimization/optuna_search | OptunaSearchProfile | OptunaSearchOutputDTO | run_optuna_search |

공통 설정은 task/scoring/cv/split/shuffle/random_state/search_space/n_iter/n_jobs/refit/timeout/pruning입니다. 공통 결과는 method/task/scoring/best_params/best_score/cv_results/folds/n_candidates/elapsed_seconds/model/study입니다. 모델·study 객체는 기본 JSON에서 제외합니다. 18개 기본·RF·LightGBM 분류/회귀 레시피 및 방법별 skeleton과 연결합니다. 상세 입력 조건·음수 손실 scorer·그룹/시간 분할은 [최적화 명세](PARAMETER_OPTIMIZATION.md)를 참고하세요.


## 도메인 통합 확장 (2026-10-20)

생존분석·시계열·비선형 회귀·표본 보완 19개 모듈은 [도메인 전체 명세](DOMAIN_ARCHITECTURE.md)에 Profile·DTO 필드와 지원 경계를 정리했습니다. 등록부와 문서 집계는 `tools/exams/audit_catalog.py`로 검사합니다.
