# 단계별 전체 모듈 기능 카탈로그

등록 모듈은 266개, 모델·목적별 JSON 레시피는 293개입니다. 모든 등록 키를 아래 한 단계에 한 번씩 배치했습니다. 단계는 활용 목적에 따른 분류이며, 모든 모듈을 차례로 실행해야 한다는 의미가 아닙니다.

[전체 파이프라인](ANALYSIS_PIPELINE.md)에서 책임과 자료 경계를 먼저 확인하고, [문서 안내](README.md)에서 분야별 상세 명세를 찾으세요. 표의 결과 열은 실제 전용 DTO의 대표 필드이며 전체 필드 목록은 아닙니다. 모듈 키 링크는 해당 기능의 구현·Profile·전체 DTO 계약으로 연결됩니다.

Dataset·FeatureSpec·Recipe 로더·OutputDTO는 공통 기반이므로 등록 모듈 수에 포함하지 않습니다. 이들의 기능은 [기반 계약](ANALYSIS_PIPELINE.md#공통-기반의-역할)에 따로 정리했습니다. 같은 함수가 여러 목적을 지원하더라도 등록 키당 한 번 셉니다. 빈 파일·호환 import·내부 helper는 구현 모듈로 중복 집계하지 않습니다.

| 단계 | 모듈 수 |
| --- | ---: |
| 1. 데이터 이해·EDA·상관분석 | 22 |
| 2. Feature 정의·전처리 | 9 |
| 3. 학습 표본 처리 | 13 |
| 4. Feature 선택 | 4 |
| 5. 모형 적합·예측 | 45 |
| 6. 검증·튜닝·후보 선택 | 5 |
| 7. 추론·가정 진단·확률 계산 | 65 |
| 8. 예측 평가·모형 비교 | 56 |
| 9. 진단·평가·설명 시각화 | 45 |
| 10. 목적함수 기반 의사결정 최적화 | 2 |
| 합계 | 266 |

## 1. 데이터 이해·EDA·상관분석

분포·결측·집단 차이·연관을 탐색합니다. 목표와의 관계로 처리 규칙을 정하면 학습 자료에서만 탐색하고 CV 안에서 검증합니다. 상관·그림만으로 인과나 자동 변수 제거를 결정하지 않습니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`correlation-cramers-v`](../src/stats/correlation/categorical/cramers_v.py) | 범주 Feature 교차표의 Cramér’s V·독립성 p값 | 두 Feature x/y 또는 열 이름 있는 표(행렬 API) | `coefficient` · `p_value` · `reject_h0` · `alpha` |
| [`correlation-kendall`](../src/stats/correlation/numeric/kendall.py) | 수치 Feature 순위의 Kendall tau·정확/근사 p값 | 두 Feature x/y 또는 열 이름 있는 표(행렬 API) | `coefficient` · `p_value` · `reject_h0` · `alpha` |
| [`correlation-matrix-transform`](../src/stats/correlation/matrix_transform.py) | 상관행렬 정리·삼각 추출·거리 변환 | 상관행렬 | `mode` · `transformed_matrix` · `pairs` · `n_features` |
| [`correlation-pearson`](../src/stats/correlation/numeric/pearson.py) | 수치 Feature 쌍의 선형 상관·p값·선택적 다중비교 보정 | 두 Feature x/y 또는 열 이름 있는 표(행렬 API) | `coefficient` · `p_value` · `reject_h0` · `alpha` |
| [`correlation-phi`](../src/stats/correlation/categorical/phi.py) | 이진 범주 쌍의 부호 있는 Phi·독립성 p값 | 두 Feature x/y 또는 열 이름 있는 표(행렬 API) | `coefficient` · `p_value` · `reject_h0` · `alpha` |
| [`correlation-spearman`](../src/stats/correlation/numeric/spearman.py) | 수치 Feature 순위의 단조 연관·p값·선택적 보정 | 두 Feature x/y 또는 열 이름 있는 표(행렬 API) | `coefficient` · `p_value` · `reject_h0` · `alpha` |
| [`plot-categorical-category-rate`](../src/Visualization/categorical/category_rate.py) | 범주별 발생률과 표본 수·불확실성으로 우선 검토 집단을 선택합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-categorical-contingency`](../src/Visualization/categorical/contingency.py) | 범주 조합의 집중·조건부 구성 차이와 희소 셀을 판단합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-categorical-count`](../src/Visualization/categorical/count.py) | 클래스 불균형·희귀 범주·범주 통합 필요성을 판단합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-categorical-grouped-distribution`](../src/Visualization/categorical/grouped_distribution.py) | 집단별 분포·중앙값·산포·소표본 차이를 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-categorical-mosaic`](../src/Visualization/categorical/mosaic.py) | 전체 구성과 범주 간 조건부 연관 패턴을 동시에 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-categorical-pareto`](../src/Visualization/categorical/pareto.py) | 누적 기여를 기준으로 우선 검토할 범주를 선택합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-numeric-ecdf`](../src/Visualization/numeric/ecdf.py) | 특정 기준 이하의 비율과 분위수로 운영 기준을 검토합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-numeric-histogram`](../src/Visualization/numeric/histogram.py) | 왜도·다봉성·희귀 구간을 확인하고 변환/구간화 후보를 검토합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-numeric-kde`](../src/Visualization/numeric/kde.py) | 히스토그램 구간 선택에 따른 착시를 비교하고 분포 구조를 탐색합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-numeric-qq`](../src/Visualization/numeric/qq.py) | 꼬리·분포 가정의 이탈을 점검해 통계 검정과 변환을 선택합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-quality-missing-association`](../src/Visualization/quality/missing_association.py) | 함께 누락되는 Feature를 탐색해 공동 대치와 수집 과정을 검토합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-quality-missing-matrix`](../src/Visualization/quality/missing_matrix.py) | 결측 블록·동시 누락·행 순서 관련 패턴을 확인합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-quality-missingness`](../src/Visualization/quality/missingness.py) | 결측 처리 우선순위와 완전 관측 손실 규모를 확인합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-relationships-correlation`](../src/Visualization/relationships/correlation.py) | 강한 중복 Feature와 보정 유의성 조건을 함께 검토합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-relationships-hexbin`](../src/Visualization/relationships/hexbin.py) | 대량 관측에서 겹침을 줄여 집중 영역과 희소 영역을 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-relationships-scatter`](../src/Visualization/relationships/scatter.py) | 비선형 패턴·군집·이상 관측과 집단별 차이를 확인합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |

## 2. Feature 정의·전처리

대상 열과 처리 순서를 선언합니다. 미적합 factory를 모델 Pipeline에 연결하고 각 학습 폴드에서 fit합니다. 목표 변환과 시퀀스 임베딩은 일반 표 Feature 처리와 분리합니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`categorical-encoding`](../src/Preprocessing/categorical/encoding.py) | one-hot·ordinal·frequency 인코딩, 학습 vocabulary와 미지 범주 정책 | 학습 Feature·선택적 새 Feature | `preprocessed_feature` · `new_preprocessed_feature` · `input_features` · `output_features` |
| [`categorical-imputation`](../src/Preprocessing/categorical/imputation.py) | 범주형 최빈값·상수 결측 대치 | 학습 Feature·선택적 새 Feature | `preprocessed_feature` · `new_preprocessed_feature` · `input_features` · `output_features` |
| [`feature-preprocessing`](../src/Preprocessing/feature_pipeline.py) | FeatureSpec의 열·처리 순서를 ColumnTransformer로 결합 | 열 이름 있는 X_train·FeatureSpec 목록·선택적 X_new | `preprocessed_feature` · `new_preprocessed_feature` · `input_features` · `output_features` |
| [`numeric-imputation`](../src/Preprocessing/numeric/imputation.py) | 평균·중앙값·최빈값·상수 결측 대치와 선택적 결측 indicator | 학습 Feature·선택적 새 Feature | `preprocessed_feature` · `new_preprocessed_feature` · `input_features` · `output_features` |
| [`numeric-power`](../src/Preprocessing/numeric/power.py) | Yeo–Johnson·Box–Cox와 선택적 표준화 | 학습 Feature·선택적 새 Feature | `preprocessed_feature` · `new_preprocessed_feature` · `input_features` · `output_features` |
| [`numeric-scaling`](../src/Preprocessing/numeric/scaling.py) | standard·minmax·robust·maxabs·행 normalize 또는 항등 변환 | 학습 Feature·선택적 새 Feature | `preprocessed_feature` · `new_preprocessed_feature` · `input_features` · `output_features` |
| [`regression-polynomial`](../src/Preprocessing/regression/polynomial.py) | 학습 전용 polynomial·상호작용 Feature 설계 | 학습 X·선택적 X_new | `preprocessed_feature` · `new_preprocessed_feature` · `input_features` · `output_features` |
| [`regression-target-transform`](../src/Preprocessing/regression/target_transform.py) | 회귀 반응변수의 train-only 변환·역변환 | 학습 수치 y·선택적 새 y | `transformed_target` · `new_transformed_target` · `transformer` |
| [`rope`](../src/Preprocessing/positional/rope.py) | 시퀀스 임베딩의 회전 위치 인코딩; 표 범주 인코딩과 별도 | 시퀀스 임베딩·위치 정보 | `preprocessed_feature` · `positions` · `inverse_frequencies` · `transformer` |

## 3. 학습 표본 처리

X와 y의 행을 함께 처리합니다. 학습 폴드에만 fit_resample하고 검증·최종 평가 행과 발생률은 보존합니다. Cook 기준 제거는 기본 동작이 아닌 명시적 학습 정책입니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`regression-influence-filter`](../src/Preprocessing/regression/influence_filter.py) | Cook 관측 검토·명시적 학습 행 처리; Pipeline 행 변환과 분리 | 학습 X/y·적합 OLS 진단/처리 정책 | `cleaned_features` · `cleaned_target` · `kept_positions` · `removed_positions` |
| [`resampling-adasyn`](../src/Preprocessing/resampling/adasyn.py) | ADASYN 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-borderline-smote`](../src/Preprocessing/resampling/borderline_smote.py) | BorderlineSMOTE 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-cluster-centroids`](../src/Preprocessing/resampling/cluster_centroids.py) | ClusterCentroids 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-edited-nearest-neighbours`](../src/Preprocessing/resampling/edited_nearest_neighbours.py) | EditedNearestNeighbours 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-near-miss`](../src/Preprocessing/resampling/near_miss.py) | NearMiss 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-random-over`](../src/Preprocessing/resampling/random_over.py) | RandomOverSampler 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-random-under`](../src/Preprocessing/resampling/random_under.py) | RandomUnderSampler 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-smote`](../src/Preprocessing/resampling/smote.py) | SMOTE 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-smoten`](../src/Preprocessing/resampling/smoten.py) | SMOTEN 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-smotenc`](../src/Preprocessing/resampling/smotenc.py) | SMOTENC 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-svm-smote`](../src/Preprocessing/resampling/svm_smote.py) | SVMSMOTE 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |
| [`resampling-tomek-links`](../src/Preprocessing/resampling/tomek_links.py) | TomekLinks 학습 표본 보완 | 학습 X/y·자료형/이웃 조건에 맞는 설정 | `X_resampled` · `y_resampled` · `class_counts_before` · `class_counts_after` |

## 4. Feature 선택

열 선택 역시 학습/CV 안에서 수행합니다. VIF는 공선성, AIC/BIC는 동일 반응·동일 관측의 우도와 복잡도, wrapper는 CV 예측 점수를 사용하므로 기준을 혼용하지 않습니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`regression-information-selection`](../src/stats/regression/information_selection.py) | 동일 반응 OLS AIC/BIC 후진 변수 선택 | 유한 full-rank X·동일 행의 원 반응 y | `selector` · `model` · `selected_features` · `history` |
| [`regression-vif-selection`](../src/Preprocessing/regression/vif_selection.py) | 공선성의 train-only 순차 VIF Feature 선택 | 학습 수치 Feature X | `preprocessed_feature` · `new_preprocessed_feature` · `input_features` · `output_features` |
| [`regularized-selection`](../src/MachineLearning/FeatureSelection/regularized.py) | L1·L2 규제 기반 변수 선택 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`wrapper-selection`](../src/MachineLearning/FeatureSelection/wrapper.py) | CV 기반 Wrapper 변수 선택 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |

## 5. 모형 적합·예측

입력 형태·타깃 정의역·분할 방식은 기법별로 다릅니다. IID 내부 분할 runner와 외부 Pipeline용 미적합 estimator를 구분합니다. 범주 입력·새 관측 예측 지원 여부는 상세 명세에서 확인합니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`adaboost`](../src/MachineLearning/Trees/adaboost.py) | AdaBoost 회귀·분류 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`adaline`](../src/MachineLearning/NeuralNetworks/adaline.py) | 아달린(ADALINE) 이진 분류 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`apriori`](../src/Association/apriori.py) | Apriori 빈발 항목집합·연관규칙 | 거래별 항목 목록 | `n_transactions` · `itemsets` · `rules` |
| [`catboost`](../src/MachineLearning/Trees/catboost.py) | CatBoost 회귀·분류 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`dbscan`](../src/MachineLearning/Clustering/dbscan.py) | DBSCAN 밀도 군집화 | 유한 수치 Feature X | `model` · `labels` · `feature_names` · `n_clusters` |
| [`eclat`](../src/Association/eclat.py) | Eclat 빈발 항목집합·연관규칙 | 거래별 항목 목록 | `algorithm` · `n_transactions` · `itemsets` · `rules` |
| [`fp-growth`](../src/Association/fp_growth.py) | FPGrowth 빈발 항목집합·연관규칙 | 거래별 항목 목록 | `algorithm` · `n_transactions` · `itemsets` · `rules` |
| [`glm-gamma`](../src/MachineLearning/GeneralizedLinearModels/gamma.py) | gamma 전용 GLM 계수·우도·조건별 잔차·영향도 | 수치 설계 X·양의 연속 y·선택적 X_new | `model` · `coefficients` · `fit_metrics` · `diagnostics` |
| [`glm-logistic`](../src/MachineLearning/GeneralizedLinearModels/logistic.py) | logistic 전용 GLM 계수·우도·조건별 잔차·영향도 | 수치 설계 X·0/1 y·선택적 X_new | `model` · `coefficients` · `fit_metrics` · `diagnostics` |
| [`glm-poisson`](../src/MachineLearning/GeneralizedLinearModels/poisson.py) | poisson 전용 GLM 계수·우도·조건별 잔차·영향도 | 수치 설계 X·비음수 정수 y·선택적 log 노출 offset | `model` · `coefficients` · `fit_metrics` · `diagnostics` |
| [`hdbscan`](../src/MachineLearning/Clustering/hdbscan.py) | HDBSCAN 계층 밀도 군집화 | 유한 수치 Feature X | `model` · `labels` · `feature_names` · `n_clusters` |
| [`k-distance`](../src/MachineLearning/Clustering/k_distance.py) | DBSCAN eps 선택용 k-distance 진단 | 유한 수치 Feature X | `model` · `feature_names` · `distances` · `sorted_indices` |
| [`kmeans`](../src/MachineLearning/Clustering/kmeans.py) | K-means 군집화 | 유한 수치 Feature X | `model` · `labels` · `feature_names` · `n_clusters` |
| [`lda`](../src/MachineLearning/Decomposition/lda.py) | LDA 선형 판별 차원 축소 | 수치 X·LDA에는 클래스 y도 필요 | `model` · `embedding` · `feature_names` · `supports_transform` |
| [`lightgbm`](../src/MachineLearning/Trees/lightgbm.py) | LightGBM 회귀·분류 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`linear-regression`](../src/MachineLearning/LinearModels/linear_regression.py) | 선형회귀(단순·다중) | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`mds`](../src/MachineLearning/Decomposition/mds.py) | MDS 다차원 척도법 | 수치 X·LDA에는 클래스 y도 필요 | `model` · `embedding` · `feature_names` · `supports_transform` |
| [`mlp`](../src/MachineLearning/NeuralNetworks/mlp.py) | MLP 다층 퍼셉트론 회귀·분류 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`naive-bayes`](../src/MachineLearning/NaiveBayes/naive_bayes.py) | 나이브 베이즈 분류 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`naive-bayes-bernoulli`](../src/MachineLearning/NaiveBayes/bernoulli.py) | bernoulli 분포 전용 Naive Bayes | 0/1 X 또는 명시 임계값·클래스 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`naive-bayes-binomial`](../src/MachineLearning/NaiveBayes/binomial.py) | Beta 평활화 성공 확률의 독립 Binomial Naive Bayes | 0..n_trials 정수 X·클래스 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`naive-bayes-categorical`](../src/MachineLearning/NaiveBayes/categorical.py) | 원 명목 범주·unknown/missing을 분리한 Categorical Naive Bayes | 원 범주 Feature X·클래스 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`naive-bayes-complement`](../src/MachineLearning/NaiveBayes/complement.py) | complement 분포 전용 Naive Bayes | 비음수 빈도 X·클래스 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`naive-bayes-gaussian`](../src/MachineLearning/NaiveBayes/gaussian.py) | gaussian 분포 전용 Naive Bayes | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`naive-bayes-multinomial`](../src/MachineLearning/NaiveBayes/multinomial.py) | multinomial 분포 전용 Naive Bayes | 비음수 빈도/가중 빈도 X·클래스 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`naive-bayes-poisson`](../src/MachineLearning/NaiveBayes/poisson.py) | Gamma 평활화 rate의 독립 Poisson Naive Bayes | 동일 노출의 비음수 정수 X·클래스 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`nonlinear-regression`](../src/MachineLearning/NonlinearModels/nonlinear_regression.py) | 명명된 비선형 평균모형 회귀 | 설명변수 1개·수치 y·모형/초기값/경계 | `model_name` · `parameter_names` · `coefficients` · `covariance` |
| [`pca`](../src/MachineLearning/Decomposition/pca.py) | PCA 주성분 분석 | 수치 X·LDA에는 클래스 y도 필요 | `model` · `embedding` · `feature_names` · `supports_transform` |
| [`perceptron`](../src/MachineLearning/NeuralNetworks/perceptron.py) | 단층 Perceptron 분류 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |
| [`regression-weighted`](../src/MachineLearning/LinearModels/weighted_regression.py) | 학습 분산 추정을 포함하는 실행가능 가중 회귀 | 수치 X/y·선택적 X_new | `model` · `coefficients` · `weights` · `weighted_residuals` |
| [`sna`](../src/Networks/sna.py) | SNA 사회연결망 분석 | 노드·가중 연결 목록 | `model` · `nodes` · `node_metrics` · `communities` |
| [`som`](../src/MachineLearning/Decomposition/som.py) | SOM 자기조직화 지도 | 수치 X·LDA에는 클래스 y도 필요 | `model` · `embedding` · `feature_names` · `supports_transform` |
| [`tf-classification`](../src/DeepLearning/TensorFlow/classification.py) | TensorFlow Dense 분류 | 수치 X/y; Dense 2D, CNN 3D/4D, RNN 3D | `model` · `history` · `test_metrics` · `predictions` |
| [`tf-cnn`](../src/DeepLearning/TensorFlow/cnn.py) | TensorFlow CNN 1D·2D 전용 분류·회귀 | 수치 X/y; Dense 2D, CNN 3D/4D, RNN 3D | `model` · `history` · `test_metrics` · `predictions` |
| [`tf-gru`](../src/DeepLearning/TensorFlow/gru.py) | TensorFlow GRU 전용 분류·회귀 | 수치 X/y; Dense 2D, CNN 3D/4D, RNN 3D | `model` · `history` · `test_metrics` · `predictions` |
| [`tf-lstm`](../src/DeepLearning/TensorFlow/lstm.py) | TensorFlow LSTM 전용 분류·회귀 | 수치 X/y; Dense 2D, CNN 3D/4D, RNN 3D | `model` · `history` · `test_metrics` · `predictions` |
| [`tf-regression`](../src/DeepLearning/TensorFlow/regression.py) | TensorFlow Dense 회귀 | 수치 X/y; Dense 2D, CNN 3D/4D, RNN 3D | `model` · `history` · `test_metrics` · `predictions` |
| [`tf-rnn`](../src/DeepLearning/TensorFlow/rnn.py) | TensorFlow RNN 시퀀스 분류·회귀 | 수치 X/y; Dense 2D, CNN 3D/4D, RNN 3D | `model` · `history` · `test_metrics` · `predictions` |
| [`tf-simple-rnn`](../src/DeepLearning/TensorFlow/simple_rnn.py) | TensorFlow SimpleRNN 전용 분류·회귀 | 수치 X/y; Dense 2D, CNN 3D/4D, RNN 3D | `model` · `history` · `test_metrics` · `predictions` |
| [`tf-time-series`](../src/DeepLearning/TensorFlow/time_series.py) | TensorFlow LSTM 시계열 예측 | 시간순 정규 간격 시계열·선택적 외생변수 | `model` · `history` · `test_metrics` · `predictions` |
| [`timeseries-arima`](../src/TimeSeries/arima.py) | ARIMA 학습·차수 선택·고정 horizon 예측 | 시간순 정규 간격 시계열·선택적 외생변수 | `selected_order` · `seasonal_order` · `candidates` · `predictions` |
| [`timeseries-sarima`](../src/TimeSeries/sarima.py) | SARIMA 학습·차수 선택·고정 horizon 예측 | 시간순 정규 간격 시계열·선택적 외생변수 | `selected_order` · `seasonal_order` · `candidates` · `predictions` |
| [`tsne`](../src/MachineLearning/Decomposition/tsne.py) | t-SNE 시각화용 차원 축소 | 수치 X·LDA에는 클래스 y도 필요 | `model` · `embedding` · `feature_names` · `supports_transform` |
| [`umap`](../src/MachineLearning/Decomposition/umap.py) | UMAP 비선형 차원 축소 | 수치 X·LDA에는 클래스 y도 필요 | `model` · `embedding` · `feature_names` · `supports_transform` |
| [`xgboost`](../src/MachineLearning/Trees/xgboost.py) | XGBoost 회귀·분류 | 수치 Feature X·타깃 y | `model_name` · `model` · `train_metrics` · `test_metrics` |

## 6. 검증·튜닝·후보 선택

최종 평가 자료를 전달하지 않습니다. IID·시간·그룹 설계를 정하고 같은 fold/scorer로 비교합니다. 선택된 CV 점수와 최종 보류 성능은 별도 결과입니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`optimization-grid-search`](../src/MachineLearning/Optimization/grid_search.py) | GridSearchCV 격자 탐색 | 학습 X/y·미적합 estimator/Pipeline·후보 공간 | `scoring` · `best_params` · `best_score` · `cv_results` |
| [`optimization-optuna-search`](../src/MachineLearning/Optimization/optuna_search.py) | Optuna TPE 교차검증 탐색 | 학습 X/y·미적합 estimator/Pipeline·후보 공간 | `scoring` · `best_params` · `best_score` · `cv_results` |
| [`optimization-random-search`](../src/MachineLearning/Optimization/random_search.py) | RandomizedSearchCV 확률 탐색 | 학습 X/y·미적합 estimator/Pipeline·후보 공간 | `scoring` · `best_params` · `best_score` · `cv_results` |
| [`regression-workflow`](../src/MachineLearning/LinearModels/regression_workflow.py) | 회귀 전처리 후보의 CV 선택·원 단위 보류 비교·조건부 추론 | 수치 Feature X·타깃 y | `model` · `baseline_model` · `comparison` · `selected_name` |
| [`timeseries-rolling-selection`](../src/TimeSeries/rolling_selection.py) | 시간순 rolling 검증 차수 선택과 미래 보류 평가 | 시간순 정규 간격 시계열·선택적 외생변수 | `model` · `comparison` · `folds` · `selected_order` |

## 7. 추론·가정 진단·확률 계산

관측 설계와 입력 단위를 먼저 정합니다. 가설 검정·구간 추정·확률 계산을 구별하며, 비기각은 가정의 증명이 아닙니다. 진단 후 변경한 모형은 학습/CV 단계로 돌아가 검증합니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`adf`](../src/stats/tests/TimeSeries/adf.py) | 시간순 자료의 단위근 ADF 검정·차수 선택 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`anderson-darling`](../src/stats/tests/diagnostics/anderson_darling.py) | 지정 분포 적합 후 Anderson–Darling 임계값 판단; p값 없음 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`bowker`](../src/stats/tests/nonParametric/Categorical/bowker.py) | 같은 대상의 전후 다범주 표 대칭성 Bowker 검정 | 전후 같은 범주 k×k 대응표 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`box-m`](../src/stats/tests/diagnostics/box_m.py) | 독립 다변량 집단의 공분산 동질성 Box M 근사 검정 | 여러 집단의 (표본,변수) 행렬 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`chi-square`](../src/stats/tests/nonParametric/Categorical/chi_square.py) | 카이제곱 적합도 검정 | 관측 도수·선택적 기대도수 | `statistic` · `p_value` · `reject_h0` · `alpha` |
| [`chi-square-homogeneity`](../src/stats/tests/nonParametric/Categorical/homogeneity.py) | 독립 모집단의 범주 분포 동질성 Pearson 검정 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`cmh`](../src/stats/tests/Association/cmh.py) | 층화 2×2 표의 공통 OR·CMH 검정 | 층×2×2 도수표 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`cochran-q`](../src/stats/tests/nonParametric/Categorical/cochran_q.py) | 같은 대상의 여러 이진 조건 성공확률 Cochran Q 검정 | 피험자×조건의 0/1 행렬 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`distribution-bernoulli`](../src/stats/distributions/bernoulli.py) | 베르누이 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-beta`](../src/stats/distributions/beta.py) | 베타 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-binomial`](../src/stats/distributions/binomial.py) | 이항 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-cauchy`](../src/stats/distributions/cauchy.py) | 코시 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-chi-square`](../src/stats/distributions/chi_square.py) | 카이제곱 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-exponential`](../src/stats/distributions/exponential.py) | 지수 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-f`](../src/stats/distributions/f.py) | F 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-gamma`](../src/stats/distributions/gamma.py) | 감마 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-geometric`](../src/stats/distributions/geometric.py) | 기하 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-hypergeometric`](../src/stats/distributions/hypergeometric.py) | 초기하 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-lognormal`](../src/stats/distributions/lognormal.py) | 로그정규 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-negative-binomial`](../src/stats/distributions/negative_binomial.py) | 음이항 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-normal`](../src/stats/distributions/normal.py) | 정규 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-poisson`](../src/stats/distributions/poisson.py) | 포아송 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-student-t`](../src/stats/distributions/student_t.py) | Student t 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`distribution-weibull`](../src/stats/distributions/weibull.py) | 와이블 분포 확률·분위수·모멘트 | 분포 모수·선택적 x/확률 q/구간·난수 개수 | `distribution_name` · `distribution_kind` · `evaluation_points` · `pmf` |
| [`f-test`](../src/stats/tests/parametric/Variance/f_test.py) | 독립 두 표본 분산비 F 검정 | 독립 정규 수치 표본 x/y | `statistic` · `p_value` · `reject_h0` · `alpha` |
| [`fisher-exact`](../src/stats/tests/nonParametric/Categorical/fisher_exact.py) | 고정 주변합 Fisher exact·RxC 열거/Monte Carlo | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`friedman`](../src/stats/tests/nonParametric/Location/friedman.py) | 완전 대응 블록×조건의 Friedman 순위 검정 | 행=블록, 열=조건의 완전 대응 행렬 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`homogeneity`](../src/stats/tests/diagnostics/homogeneity.py) | 등분산 검정 | 독립 수치 집단들 | `statistic` · `p_value` · `reject_h0` · `alpha` |
| [`hotelling`](../src/stats/tests/parametric/Multivariate/hotelling.py) | 독립 두 다변량 평균 벡터의 Hotelling T² 검정 | 독립 두 (표본,변수) 행렬 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`independence`](../src/stats/tests/nonParametric/Categorical/independence.py) | 카이제곱 독립성 검정 | 독립 관측 교차 빈도표 | `statistic` · `p_value` · `reject_h0` · `alpha` |
| [`kolmogorov-smirnov`](../src/stats/tests/diagnostics/kolmogorov_smirnov.py) | 고정 모수 KS 또는 정규/지수 추정 모수 Lilliefors | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`kruskal-wallis`](../src/stats/tests/nonParametric/Location/kruskal_wallis.py) | 독립 여러 집단의 순위 Kruskal–Wallis 검정 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`mann-whitney`](../src/stats/tests/nonParametric/Location/mann_whitney.py) | 독립 두 집단의 Mann–Whitney U 정확/점근 검정 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`manova`](../src/stats/tests/parametric/Multivariate/manova.py) | 일원배치 MANOVA | 다변량 반응·집단 라벨 | `test_type` · `criterion` · `statistic` · `f_value` |
| [`mcnemar`](../src/stats/tests/nonParametric/Categorical/mcnemar.py) | McNemar 대응 이진 검정 | 대응 이진 2×2 표 | `statistic` · `p_value` · `reject_h0` · `alpha` |
| [`mood-median`](../src/stats/tests/nonParametric/Location/mood_median.py) | 공동 중앙값 위/아래 빈도를 이용한 독립 집단 비교 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`normality`](../src/stats/tests/diagnostics/normality.py) | 정규성 검정 | 비상수 수치 표본 | `statistic` · `p_value` · `reject_h0` · `alpha` |
| [`odds-ratio`](../src/stats/tests/Association/odds_ratio.py) | 독립 2×2 표의 OR·log-Wald 검정·구간 | 노출/비노출×사건/비사건 2×2 표 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`one-sample-mean`](../src/stats/tests/parametric/Average/one_sample.py) | 단일 평균의 Z/t 검정·평균 신뢰구간 | 수치 표본 x·귀무 평균·선택적 알려진 sigma | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`one-sample-proportion`](../src/stats/tests/parametric/Proportion/one_sample.py) | 단일 비율 exact/score 검정·Clopper–Pearson/Wilson 구간 | 성공 수·시행 수·귀무 비율 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`one-sample-variance`](../src/stats/tests/parametric/Variance/one_sample.py) | 정규 모집단의 모분산 카이제곱 검정·구간 | 수치 표본·귀무 분산 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`oneway-anova`](../src/stats/tests/parametric/Average/anova/one_way.py) | 일원배치 ANOVA·Welch ANOVA | 독립 수치 집단들 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`paired-t`](../src/stats/tests/parametric/Average/paired.py) | 대응 관측 차이의 평균 t 검정·차이 구간 | 같은 대상의 동일 길이 x/y | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`posthoc`](../src/stats/tests/parametric/Average/anova/posthoc.py) | 독립 집단 평균 사후검정 | 독립 집단들·선택적 집단 이름 | `test_type` · `alpha` · `equal_var` · `group_names` |
| [`regression-collinearity`](../src/stats/regression/collinearity.py) | VIF·tolerance·표준화 조건수·설계 rank | 수치 설계 X | `table` · `condition_number` · `rank` |
| [`regression-heteroscedasticity`](../src/stats/regression/heteroscedasticity.py) | Breusch–Pagan/Koenker·White 등분산성 진단 | 적합 OLS DTO·필요시 제약/축소 모형 | `tests` · `details` |
| [`regression-independence`](../src/stats/regression/independence.py) | 순서 자료의 BG·Ljung–Box 및 DW 기술 지표 | 적합 OLS DTO·필요시 제약/축소 모형 | `tests` · `details` |
| [`regression-influence`](../src/stats/regression/influence.py) | Cook·leverage·studentized 잔차·DFFITS·DFBETAS | 적합 OLS DTO·필요시 제약/축소 모형 | `table` · `dfbetas` · `thresholds` |
| [`regression-joint-f`](../src/stats/regression/joint_f.py) | 계수 묶음과 사용자 선형 가설의 F 검정 | 적합 OLS DTO·필요시 제약/축소 모형 | `tests` · `details` |
| [`regression-linearity`](../src/stats/regression/linearity.py) | RESET·Rainbow 선형 사양 진단 | 적합 OLS DTO·필요시 제약/축소 모형 | `tests` · `details` |
| [`regression-nested-f`](../src/stats/regression/nested_f.py) | 중첩 회귀모형의 추가 설명력 F 검정 | 적합 OLS DTO·필요시 제약/축소 모형 | `tests` · `details` |
| [`regression-normality`](../src/stats/regression/normality.py) | 회귀 잔차의 정규성 진단 | 적합 OLS DTO·필요시 제약/축소 모형 | `tests` · `details` |
| [`regression-ols`](../src/stats/regression/ols.py) | OLS 계수 t·전체 F·R²·평균/관측 예측구간 | 수치 설계 X/y·선택적 X_new | `model` · `coefficients` · `fit_metrics` · `design` |
| [`repeated-measures-anova`](../src/stats/tests/parametric/Average/anova/repeated_measures.py) | 완전 균형 일원 반복측정 ANOVA·선택적 GG 보정 | 행=피험자, 열=조건의 완전 대응 행렬 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`risk-ratio`](../src/stats/tests/Association/risk_ratio.py) | cohort 2×2 표의 RR·log-Wald 검정·구간 | cohort 노출/비노출×사건/비사건 표 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`runs`](../src/stats/tests/nonParametric/Sequence/runs.py) | 순서 이진열의 runs 무작위성 정규근사 검정 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`sign`](../src/stats/tests/nonParametric/Location/sign.py) | 단일 기준·대응 차이의 부호 정확 이항 검정 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`survival-cox`](../src/stats/survival/cox.py) | Cox 비례위험 회귀·생존예측·PH 진단 | 시간·사건 지시자·집단/공변량(기법별) | `coefficients` · `ph_diagnostics` · `predictions` · `events` |
| [`survival-kaplan-meier`](../src/stats/survival/kaplan_meier.py) | 우측 중도절단 자료의 Kaplan-Meier 생존함수 추정 | 시간·사건 지시자·집단/공변량(기법별) | `times` · `at_risk` · `events` · `censored` |
| [`survival-logrank`](../src/stats/survival/logrank.py) | 다집단 로그순위 검정과 Holm 쌍별 비교 | 시간·사건 지시자·집단/공변량(기법별) | `statistic` · `p_value` · `reject_h0` · `alpha` |
| [`survival-rmst`](../src/stats/survival/rmst.py) | 제한평균생존시간 추정 | 시간·사건 지시자·집단/공변량(기법별) | `tau` · `estimate` |
| [`two-sample-mean`](../src/stats/tests/parametric/Average/two_sample.py) | 독립 두 평균의 Student/Welch/Z 검정·차이 구간 | 독립 수치 표본 x/y·귀무 차이 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`two-sample-proportion`](../src/stats/tests/parametric/Proportion/two_sample.py) | 독립 비율의 pooled Z·Newcombe 차이 구간 | 두 집단의 성공/시행 수 | `test_type` · `statistic` · `p_value` · `reject_h0` |
| [`twoway-anova`](../src/stats/tests/parametric/Average/anova/two_way.py) | 이원배치 ANOVA | 수치 반응·두 범주 요인 | `test_type` · `factor1_name` · `factor2_name` · `f_factor1` |
| [`wilcoxon`](../src/stats/tests/nonParametric/Location/wilcoxon.py) | 단일·대응 차이의 signed-rank 정확/점근 검정 | 기법별 표본/집단/순서열(상세 명세 확인) | `test_type` · `statistic` · `p_value` · `reject_h0` |

## 8. 예측 평가·모형 비교

적합·샘플링 없이 같은 행 순서의 정답과 예측을 평가합니다. 확률 열은 명시한 클래스 순서와 대응해야 합니다. 최종 평가 결과로 후보를 계속 고르지 않습니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`metric-classification-accuracy`](../src/Evaluation/classification/accuracy.py) | 전체 정분류 비율 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-average-precision`](../src/Evaluation/classification/average_precision.py) | recall 증가 가중 평균 precision(AP) | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-balanced-accuracy`](../src/Evaluation/classification/balanced_accuracy.py) | 클래스별 recall 평균 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-brier-score`](../src/Evaluation/classification/brier_score.py) | 이진 양성 확률의 제곱오차 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-calibration`](../src/Evaluation/classification/calibration.py) | 확률 bin별 평균 예측·실제 발생률; 보정기 적합 없음 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-cohen-kappa`](../src/Evaluation/classification/cohen_kappa.py) | 우연 일치 보정 분류 일치도 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-confusion-matrix`](../src/Evaluation/classification/confusion_matrix.py) | 지정 클래스 순서의 오분류 빈도표 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-f1`](../src/Evaluation/classification/f1.py) | precision·recall 조화평균 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-fbeta`](../src/Evaluation/classification/fbeta.py) | 재현율 가중 F-beta | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-gain`](../src/Evaluation/classification/gain.py) | 선별 구간의 전체 양성 포착 비율 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-gain-curve`](../src/Evaluation/classification/gain_curve.py) | 선별 비율별 cumulative gain 곡선 수치 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-gini`](../src/Evaluation/classification/gini.py) | ROC AUC 기반 순위 Gini | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-ks`](../src/Evaluation/classification/ks.py) | 누적 양성·음성 분리의 최대 차이 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-lift`](../src/Evaluation/classification/lift.py) | 선별 구간 양성률/전체 양성률 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-lift-curve`](../src/Evaluation/classification/lift_curve.py) | 선별 비율별 lift 곡선 수치 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-log-loss`](../src/Evaluation/classification/log_loss.py) | 실제 클래스 확률의 음의 로그손실 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-mcc`](../src/Evaluation/classification/mcc.py) | 혼동행렬 기반 Matthews 상관 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-metrics`](../src/Evaluation/classification/metrics.py) | classification 분야 지표 묶음 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-pr-auc`](../src/Evaluation/classification/pr_auc.py) | PR 곡선 사다리꼴 면적; AP와 구분 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-precision`](../src/Evaluation/classification/precision.py) | 예측 양성의 정답 비율 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-precision-recall-curve`](../src/Evaluation/classification/precision_recall_curve.py) | 임계값별 precision·recall 배열 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-recall`](../src/Evaluation/classification/recall.py) | 실제 양성의 검출 비율 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-roc-auc`](../src/Evaluation/classification/roc_auc.py) | 양성 점수 순위의 ROC 면적 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-roc-curve`](../src/Evaluation/classification/roc_curve.py) | 임계값별 FPR·TPR·threshold 배열 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-classification-specificity`](../src/Evaluation/classification/specificity.py) | 실제 음성의 정분류 비율 | 같은 행 정답·예측 라벨/점수/확률 | `metric_name` · `value` · `details` |
| [`metric-clustering-ami`](../src/Evaluation/clustering/ami.py) | 우연 보정 상호정보량 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-ari`](../src/Evaluation/clustering/ari.py) | 정답 군집과 우연 보정 Rand 일치도 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-calinski-harabasz`](../src/Evaluation/clustering/calinski_harabasz.py) | 군집 간/내 산포비 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-completeness`](../src/Evaluation/clustering/completeness.py) | 동일 정답 클래스의 군집 결집 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-davies-bouldin`](../src/Evaluation/clustering/davies_bouldin.py) | 군집 내부 산포 대비 군집 간 분리 비율 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-homogeneity`](../src/Evaluation/clustering/homogeneity.py) | 예측 군집 안의 정답 클래스 순도 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-metrics`](../src/Evaluation/clustering/metrics.py) | clustering 분야 지표 묶음 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-nmi`](../src/Evaluation/clustering/nmi.py) | 정규화 상호정보량 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-silhouette`](../src/Evaluation/clustering/silhouette.py) | 같은 군집·다른 군집의 거리 기반 분리도 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-clustering-v-measure`](../src/Evaluation/clustering/v_measure.py) | homogeneity·completeness 조화평균 | 군집 라벨·좌표 X 또는 참 라벨 | `metric_name` · `value` · `details` |
| [`metric-comparison`](../src/Evaluation/comparison.py) | 동일 정답·행 순서의 여러 모델 평가표 | 동일 정답·모형별 정렬된 예측/점수 | `table` · `directions` |
| [`metric-cv-summary`](../src/Evaluation/cv_summary.py) | 폴드별 평가값의 평균·표준편차·표본 수 | fold별 수치 평가값·선택적 크기 | `table` |
| [`metric-regression-adjusted-r2`](../src/Evaluation/regression/adjusted_r2.py) | 표본 수·설명변수 수 보정 R² | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-explained-variance`](../src/Evaluation/regression/explained_variance.py) | 잔차 분산 기준 설명력 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-mae`](../src/Evaluation/regression/mae.py) | 절대오차 평균 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-mape`](../src/Evaluation/regression/mape.py) | 실제값 기준 절대 백분율 오차 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-max-error`](../src/Evaluation/regression/max_error.py) | 최대 절대오차 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-median-absolute-error`](../src/Evaluation/regression/median_absolute_error.py) | 절대오차 중앙값 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-metrics`](../src/Evaluation/regression/metrics.py) | regression 분야 지표 묶음 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-mse`](../src/Evaluation/regression/mse.py) | 제곱오차 평균 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-msle`](../src/Evaluation/regression/msle.py) | 비음수 실제·예측의 log1p 제곱오차 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-pinball-loss`](../src/Evaluation/regression/pinball_loss.py) | 지정 분위수 예측의 pinball 손실 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-r2`](../src/Evaluation/regression/r2.py) | 평가 정답 평균 대비 오차 감소 비율 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-rmse`](../src/Evaluation/regression/rmse.py) | 제곱오차 평균의 제곱근 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-rmsle`](../src/Evaluation/regression/rmsle.py) | MSLE 제곱근 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-smape`](../src/Evaluation/regression/smape.py) | 실제·예측 크기를 함께 쓰는 대칭 백분율 오차 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-regression-wape`](../src/Evaluation/regression/wape.py) | 절대오차 합/실제 절댓값 합 | 같은 행·같은 단위 y_true/y_pred | `metric_name` · `value` · `details` |
| [`metric-time-series-horizon-errors`](../src/Evaluation/time_series/horizon_errors.py) | 예측 horizon별 오차 집계 | 시간/horizon 정렬 정답·예측·학습 기준열 | `metric_name` · `value` · `details` |
| [`metric-time-series-mase`](../src/Evaluation/time_series/mase.py) | 학습 naive 절대 차분으로 나눈 평가 MAE | 시간/horizon 정렬 정답·예측·학습 기준열 | `metric_name` · `value` · `details` |
| [`metric-time-series-metrics`](../src/Evaluation/time_series/metrics.py) | time-series 분야 지표 묶음 | 시간/horizon 정렬 정답·예측·학습 기준열 | `metric_name` · `value` · `details` |
| [`metric-time-series-rmsse`](../src/Evaluation/time_series/rmsse.py) | 학습 naive 제곱 차분으로 나눈 평가 RMSE | 시간/horizon 정렬 정답·예측·학습 기준열 | `metric_name` · `value` · `details` |

## 9. 진단·평가·설명 시각화

호출자가 만든 ax에만 그립니다. DTO의 수치/표와 실행 객체를 구분하고 subplots 조합·저장은 호출자가 수행합니다. 데이터나 이미 계산한 지표를 사용하며 그림 자체가 모형을 교정하지 않습니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`plot-association-rules`](../src/Visualization/association/rules.py) | 빈도·조건부 확률·독립 대비 lift를 함께 비교해 규칙 검토 우선순위를 정합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-classification-calibration`](../src/Visualization/classification/calibration.py) | 확률의 과신·과소신과 실제 발생률의 일치를 점검합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-classification-confusion`](../src/Visualization/classification/confusion.py) | 어느 클래스 사이에서 오분류가 발생하는지 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-classification-precision-recall`](../src/Visualization/classification/precision_recall.py) | 불균형 분류에서 재현율과 양성 예측 신뢰도를 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-classification-roc`](../src/Visualization/classification/roc.py) | 분류 순위 성능과 위양성/민감도의 균형을 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-classification-threshold`](../src/Visualization/classification/threshold.py) | 비용과 precision/recall trade-off를 근거로 검증 자료의 임계값 후보를 선택합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-clustering-dendrogram`](../src/Visualization/clustering/dendrogram.py) | 계층 병합 거리와 절단 후보로 군집 구조를 검토합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-clustering-elbow`](../src/Visualization/clustering/elbow.py) | 추가 군집 수에 따른 개선 규모를 비교합니다. 모델을 다시 적합하지 않습니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-clustering-k-distance`](../src/Visualization/clustering/k_distance.py) | 밀도 군집 eps 후보와 거리의 급격한 변화를 검토합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-clustering-silhouette`](../src/Visualization/clustering/silhouette.py) | 음의 silhouette·군집별 산포와 평균으로 군집 분리를 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-decomposition-explained-variance`](../src/Visualization/decomposition/explained_variance.py) | 목표 누적분산에 필요한 성분 수와 차원 축소 손실을 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-decomposition-loadings`](../src/Visualization/decomposition/loadings.py) | 성분을 구성하는 원 Feature와 부호·기여 패턴을 해석합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-networks-graph`](../src/Visualization/networks/graph.py) | 연결 구조·분리 요소·허브 후보를 검토합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-optimization-convergence`](../src/Visualization/optimization/convergence.py) | 유전 알고리즘 실행가능 최우수 목적값의 세대별 변화 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-bernoulli`](../src/Visualization/qq/bernoulli.py) | bernoulli 이산 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-beta`](../src/Visualization/qq/beta.py) | beta 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-binomial`](../src/Visualization/qq/binomial.py) | binomial 이산 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-chi-square`](../src/Visualization/qq/chi_square.py) | chi_square 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-exponential`](../src/Visualization/qq/exponential.py) | exponential 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-gamma`](../src/Visualization/qq/gamma.py) | gamma 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-logistic`](../src/Visualization/qq/logistic.py) | logistic 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-lognormal`](../src/Visualization/qq/lognormal.py) | lognormal 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-normal`](../src/Visualization/qq/normal.py) | normal 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-poisson`](../src/Visualization/qq/poisson.py) | poisson 이산 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-student-t`](../src/Visualization/qq/student_t.py) | student_t 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-uniform`](../src/Visualization/qq/uniform.py) | uniform 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-qq-weibull`](../src/Visualization/qq/weibull.py) | weibull 분포 전용 Q-Q 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-coefficient-intervals`](../src/Visualization/regression/coefficient_intervals.py) | 회귀계수·t·p·신뢰구간 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-cooks-distance`](../src/Visualization/regression/cooks_distance.py) | Cook 관측 영향력과 명시 기준 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-glm-residuals`](../src/Visualization/regression/glm_residuals.py) | 분포별 GLM 적합평균과 Pearson/deviance 잔차 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-leverage`](../src/Visualization/regression/leverage.py) | leverage·studentized 잔차·Cook 등고선 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-model-comparison`](../src/Visualization/regression/model_comparison.py) | 기준·전처리·대안 회귀의 검증 점수 비교 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-observed-predicted`](../src/Visualization/regression/observed_predicted.py) | 원 단위 관측–예측과 R²·RMSE | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-partial-residual`](../src/Visualization/regression/partial_residual.py) | Feature별 component-plus-residual 선형성 탐색 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-prediction-interval`](../src/Visualization/regression/prediction_interval.py) | 평균 CI와 새 관측 PI의 분리 표시 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-residual-fitted`](../src/Visualization/regression/residual_fitted.py) | 잔차–적합값의 곡률·등분산 탐색 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-residual-order`](../src/Visualization/regression/residual_order.py) | 잔차 관측/시간 순서 탐색 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-scale-location`](../src/Visualization/regression/scale_location.py) | 표준화 잔차 산포의 scale-location | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-regression-vif`](../src/Visualization/regression/vif.py) | 공선성 VIF와 표시 상한 | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-selection-feature-importance`](../src/Visualization/selection/feature_importance.py) | 동일 평가 기준의 Feature 중요도와 반복 변동을 비교해 검토 우선순위를 정합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-selection-validation-curve`](../src/Visualization/selection/validation_curve.py) | 학습량/분류 설정 변화의 train–validation 차이로 과적합과 개선 여지를 검토합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-time-series-autocorrelation`](../src/Visualization/time_series/autocorrelation.py) | 시간 의존성과 계절 지연 후보를 탐색합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-time-series-lag`](../src/Visualization/time_series/lag.py) | 지연 관계·비선형 반복 패턴을 확인합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-time-series-seasonal`](../src/Visualization/time_series/seasonal.py) | 요일·월·시간별 반복 구조와 수집 불균형을 비교합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |
| [`plot-time-series-timeline`](../src/Visualization/time_series/timeline.py) | 수준 변화·추세·수집 공백·관측 급변을 탐색합니다. | 기법별 원자료/계산 결과·호출자 ax | `figure` · `axes` · `artists` · `data` |

## 10. 목적함수 기반 의사결정 최적화

예측모형의 하이퍼파라미터 탐색과 별개입니다. 목적함수·제약·변수 범위를 입력하고 해의 실행가능성과 최적성 근거를 구분합니다.

| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |
| --- | --- | --- | --- |
| [`genetic-algorithm`](../src/Optimization/genetic_algorithm.py) | 실수·정수 혼합 유전 알고리즘: 실행가능성 우선 선택 | 목적함수·변수 경계·정수열·제약함수 | `best_solution` · `best_objective` · `best_candidate` · `best_violation` |
| [`production-allocation`](../src/Optimization/production_allocation.py) | 직렬 생산라인 정수 설비 배분과 독립 전수검증 | 설비 비용/처리량·예산·가용 수·GA 설정 | `ga_result` · `process_table` · `annual_production` · `total_cost` |

## 기능 명세와 API 필드 참조의 구분

이 목록은 기능·자료·결과를 선택하기 위한 명세입니다. [핵심 API 필드](reference/CORE_API.md)와 분야별 문서는 파라미터의 정확한 타입·기본값·배열 대응·실패 조건을 보충합니다. 현재 기능 범위는 등록부와 소스 기준이며 제안 단계의 기능을 포함하지 않습니다.

생성: `python tools/docs/build_module_catalog.py`; 누락·중복·생성 차이 검사: `python tools/docs/build_module_catalog.py --check`. 이 작업은 모형을 적합하거나 선택 패키지를 로드하지 않습니다.
