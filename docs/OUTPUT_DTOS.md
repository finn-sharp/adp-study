# 분석 Output DTO 계약

모든 공개 분석 실행기 `run_*`와 샘플링/거리/연결행렬/영어분석/LIME 분석 함수는
명시적인 **dataclass Output DTO**를 반환한다. 입력 조건은 Profile, 결과는 OutputDTO다.
코드 내부의 Attributes/Args/Returns/Raises 명세가 계약의 기준이다.

## 활용

```python
result = run_multiple_regression(X, y)
print(result.test_metrics)                  # DTO 필드 접근
prediction = result.model.predict(new_X)   # 적합 모델 재사용
native = result.to_dict()                  # 배열/표는 그대로 유지, 모델은 기본 제외
with_model = result.to_dict(include_runtime=True)
json_text = result.to_json(indent=2)        # 저장 가능한 데이터만 JSON 변환
```

모든 DTO는 `src.output.OutputDTO`를 상속한다. 모델별로 전용 DTO 클래스를 두면서
공통 지도학습 필드(모델, 지표, 예측, 분할 위치, 클래스/확률, CV)를 상속한다.
같은 알고리즘의 회귀/분류는 동일 전용 DTO이며 task/classes/optional 필드로 구분한다.
다중·다항·비선형 회귀에는 각각 별도 DTO를 둔다.

## 사전·JSON 규칙

- `to_dict()`는 실행 객체를 제외하고 NumPy/pandas 데이터를 native 형태로 유지한다.
  배열·표·추정기를 deep copy하지 않는다.
- `include_runtime=True`는 모델·인코더·다항 변환기·벡터라이저·선택기·sampler·그림·원 외부 결과 객체도 포함한다.
- `to_json()` 또는 `to_dict(json_safe=True)`는 저장 가능한 결과 데이터만 변환한다.
- DataFrame은 columns/index/data, Series는 name/index/data를 보존한다.
- 희소행렬은 COO row/col/data/shape로 표현하여 dense 확장을 피한다.
- 배열은 목록, 날짜는 ISO 문자열, 집합은 안정적인 순서의 목록, 비유한 수치는 null이다.
- JSON 사전 키는 문자열이며 변환 후 중복되면 오류를 낸다. 임의 Python 객체는 암묵적으로
  문자열화하지 않고 TypeError를 반환한다.
- JSON은 분석 결과 교환 형식이며 모델/그림 복원이나 원 pandas dtype의 정확한 왕복을
  보장하는 저장 형식이 아니다. 대용량 예측/표에는 별도의 데이터 파일 저장이 필요할 수 있다.

## 기존 호출 호환

- 기존 `*Result` 이름은 실제 `*OutputDTO` 클래스의 별칭으로 유지한다.
- 이전 dict 반환 함수는 DTO 속성을 사용하며 `result['field']`/`get`/`dict(result)`도 지원한다.
- ANOVA는 `ANOVAOutputDTO(method, anova_table, analysis)`이며 방법별 결과는 analysis DTO다.
  기존 statistic/p_value/f_factor1 등은 analysis로 위임해 속성 접근을 유지한다.
- 거리/연결행렬은 `.matrix`가 원 배열이고 `np.asarray(result)`와 `.shape`도 지원한다.
- Pairplot은 `.grid`가 원 seaborn 객체이며 `.figure`/`.fig`로 그림 접근을 유지한다.
- 시계열 분해의 원 객체는 runtime `decomposition`, 저장 가능한 성분은
  `decomposition_components`의 observed/trend/seasonal/resid 시계열로 제공한다.
- 영어 분석은 `.mode`/`.items`, LIME 설명은 `.classes`/`.probabilities`/`.contributions`/
  `.intercepts`를 사용한다. 원 외부 객체는 `.explanation`이다.

## 분석 결과와 지원 함수

DTO는 분석 결과를 반환하는 실행 API에 적용한다. `validate_features`, `numeric_sample`,
`validate_series` 같은 입력 검증, `make_scaler`/`make_sampler`/`build_sampling_pipeline`과
토크나이저 생성 같은 구성 함수는 원래 추정기/배열/콜백 타입을 유지한다.
`clean_text`는 전처리 문자열, `set_matplotlib_font`는 설정 부수효과 함수다.
이 지원 함수들은 분석 결과 DTO로 감싸지 않는다.

## 모듈별 반환 목록

분석 API 55개, 전용 반환 DTO 53종. 기계 판독 목록은 [output_dtos.json](output_dtos.json)이다.

| 모듈 | 분석 함수 | Output DTO |
|---|---|---|
| `src/Association/apriori.py` | `run_association` | `AssociationOutputDTO` |
| `src/MachineLearning/Clustering/clustering.py` | `run_clustering` | `ClusteringOutputDTO` |
| `src/MachineLearning/Clustering/clustering.py` | `distance_matrix` | `DistanceMatrixOutputDTO` |
| `src/MachineLearning/Clustering/clustering.py` | `make_linkage_matrix` | `LinkageMatrixOutputDTO` |
| `src/MachineLearning/Decomposition/manifold.py` | `run_embedding` | `EmbeddingOutputDTO` |
| `src/MachineLearning/Decomposition/pca.py` | `run_decomposition` | `DecompositionOutputDTO` |
| `src/MachineLearning/Decomposition/pca.py` | `run_pca_classification` | `PCAClassificationOutputDTO` |
| `src/MachineLearning/Linear/lasso.py` | `run_lasso` | `LassoOutputDTO` |
| `src/MachineLearning/Linear/least_squares.py` | `run_least_squares` | `LeastSquaresOutputDTO` |
| `src/MachineLearning/Linear/logistic.py` | `run_logistic` | `LogisticOutputDTO` |
| `src/MachineLearning/Linear/numerical.py` | `run_numerical_regression` | `NumericalOutputDTO` |
| `src/MachineLearning/Linear/ridge.py` | `run_ridge` | `RidgeOutputDTO` |
| `src/MachineLearning/NaiveBayes/naive_bayes.py` | `run_naive_bayes` | `NaiveBayesOutputDTO` |
| `src/MachineLearning/Neighbors/knn.py` | `run_knn` | `KNNOutputDTO` |
| `src/MachineLearning/NeuralNetworks/mlp.py` | `run_mlp` | `MLPOutputDTO` |
| `src/MachineLearning/Regression/regression.py` | `run_multiple_regression` | `MultipleRegressionOutputDTO` |
| `src/MachineLearning/Regression/regression.py` | `run_polynomial_regression` | `PolynomialRegressionOutputDTO` |
| `src/MachineLearning/Regression/regression.py` | `run_nonlinear_regression` | `NonlinearRegressionOutputDTO` |
| `src/MachineLearning/SVM/svm.py` | `run_svm` | `SVMOutputDTO` |
| `src/MachineLearning/Trees/adaboost.py` | `run_adaboost` | `AdaBoostOutputDTO` |
| `src/MachineLearning/Trees/decision_tree.py` | `run_decision_tree` | `DecisionTreeOutputDTO` |
| `src/MachineLearning/Trees/gradient_boosting.py` | `run_gradient_boosting` | `GradientBoostingOutputDTO` |
| `src/MachineLearning/Trees/lightgbm.py` | `run_lightgbm` | `LightGBMOutputDTO` |
| `src/MachineLearning/Trees/random_forest.py` | `run_random_forest` | `RandomForestOutputDTO` |
| `src/MachineLearning/Trees/xgboost.py` | `run_xgboost` | `XGBoostOutputDTO` |
| `src/MachineLearning/common.py` | `run_supervised` | `SupervisedOutputDTO` |
| `src/MachineLearning/feature_selection.py` | `run_feature_selection` | `FeatureSelectionOutputDTO` |
| `src/MachineLearning/feature_selection.py` | `run_permutation_importance` | `PermutationImportanceOutputDTO` |
| `src/Preprocessing/sampling.py` | `resample_training_data` | `SamplingOutputDTO` |
| `src/TextMining/text.py` | `run_vectorization` | `VectorizationOutputDTO` |
| `src/TextMining/text.py` | `run_text_classification` | `TextClassificationOutputDTO` |
| `src/TextMining/text.py` | `run_topic_model` | `TopicOutputDTO` |
| `src/TextMining/text.py` | `analyze_english` | `EnglishAnalysisOutputDTO` |
| `src/TextMining/text.py` | `explain_text` | `TextExplanationOutputDTO` |
| `src/TimeSeries/arima.py` | `run_arima` | `ForecastOutputDTO` |
| `src/TimeSeries/arima.py` | `run_sarima` | `ForecastOutputDTO` |
| `src/TimeSeries/forecasting.py` | `run_forecast` | `ForecastOutputDTO` |
| `src/TimeSeries/forecasting.py` | `run_series_diagnostics` | `SeriesDiagnosticsOutputDTO` |
| `src/TimeSeries/forecasting.py` | `run_multivariate_forecast` | `MultivariateForecastOutputDTO` |
| `src/Visualization/plots.py` | `run_plot` | `PlotOutputDTO` |
| `src/Visualization/plots.py` | `run_pairplot` | `PairPlotOutputDTO` |
| `src/stats/analysis.py` | `run_descriptive` | `DescriptiveOutputDTO` |
| `src/stats/analysis.py` | `run_statistical_test` | `StatisticalOutputDTO` |
| `src/stats/anova.py` | `run_anova` | `ANOVAOutputDTO` |
| `src/stats/manova.py` | `run_manova` | `MANOVAOutputDTO` |
| `src/stats/posthoc.py` | `run_posthoc` | `PostHocOutputDTO` |
| `src/stats/regression.py` | `run_regression_inference` | `RegressionOutputDTO` |
| `src/stats/regression.py` | `run_variable_selection` | `SelectionOutputDTO` |
| `src/stats/repeated_measures.py` | `run_repeated_anova` | `RepeatedANOVAOutputDTO` |
| `src/stats/tests/nonParametric/onesample.py` | `run_one_sample_rank_test` | `OneSampleRankOutputDTO` |
| `src/stats/tests/nonParametric/twosample.py` | `run_two_sample_rank_test` | `TwoSampleRankOutputDTO` |
| `src/stats/tests/parametric/Average/Bivariate.py` | `run_two_sample_test` | `TwoSampleTestOutputDTO` |
| `src/stats/tests/parametric/Average/Multivariate/OneWay.py` | `run_oneway_anova` | `OneWayANOVAOutputDTO` |
| `src/stats/tests/parametric/Average/Multivariate/TwoWay.py` | `run_twoway_anova` | `TwoWayANOVAOutputDTO` |
| `src/stats/tests/parametric/Average/Univariate.py` | `run_one_sample_test` | `TestOutputDTO` |
