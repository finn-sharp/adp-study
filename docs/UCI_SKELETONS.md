# UCI 모델 스켈레톤 설계 및 실행

## 참고 코드 분석

참고 파일은 `src/MachineLearning/skeleton/xgboost.py`입니다. 사용자가 제공한 참고 원본은
`docs/reference-skeleton/xgboost.py`에 보존했습니다.

원본의 핵심은 `load_data → load_recipe → build_pipeline → main` 순서입니다.
Wine Quality를 읽고 `quality`를 타깃으로 분리한 다음, `FeatureSpec`에 특성 열과
`StandardScaler`를 지정합니다. `BasePreprocessor`를 모델과 `Basepipeline`으로 연결하고,
학습·테스트를 먼저 나눈 뒤 학습 자료에서만 적합합니다. 마지막에는 예측값으로 RMSE와 R²를
출력하고 전처리·모델 구성을 확인합니다. 새 스켈레톤도 이 흐름을 유지합니다.

현재 구현과 연결할 때 다음을 교정했습니다.

1. 참고 코드가 import하는 `XGBoostRegressionModel`은 현재 모델 파일에 없습니다.
   새 지도 스켈레톤은 기존 `_create_model(Profile)`을 연결하여 모델 정의를 복제하지 않습니다.
2. 원 JSON은 라이브러리 인자와 공통 실행 설정을 함께 담고 있으며 일부 과거 별칭도 있습니다.
   `load_recipe` 단계에서 기존 `load_profile`을 사용해 모델별 JSON·별칭·목적을 검증합니다.
3. 원 `BasePreprocessor` wrapper는 sklearn clone 계약을 제공하지 않습니다.
   해당 wrapper의 `get_preprocessor()`가 반환하는 `ColumnTransformer`를 연결하여
   GridSearchCV가 전처리를 폴드 안에서 다시 적합할 수 있게 했습니다. GitHub에는 참고 base 파일이 없으므로 같은 계약의 기반 클래스 3개를 함께 제공합니다.
4. 회귀 예제를 분류로 확장할 때는 학습 타깃에서만 LabelEncoder를 적합합니다.
   평가·결과·`result.predict`에는 원 라벨을 복원합니다. Adaline은 두 Iris 클래스만 사용합니다.
5. Wrapper 변수 선택의 CV보다 먼저 전체 학습 자료를 스케일링하면 누수가 생깁니다.
   바깥 FeatureSpec은 passthrough로 두고, 기존 선택기의 내부 CV와 선택 후 스케일러를 사용합니다.
6. TensorFlow에 외부 전체 자료 스케일러를 덧붙이지 않습니다. 기존 runner의 학습·검증·테스트
   분할과 학습 구간 전처리·조기 종료·DTO를 사용합니다. Pandas 분류 라벨은 명시적인 문자열 배열로 전달합니다.
7. 정식 GitHub 경로인 `src.MachineLearning.Trees`를 사용해 기존 레시피 registry와 일치시킵니다.

## 파일과 모델 범위

33개 실행 파일을 참고 코드와 같은 `src/MachineLearning/skeleton/`에 둡니다.
각 파일에는 작성 정보·변경 이력·메타데이터와 함수별 Args/Returns/Raises가 있습니다.
공통 연결·UCI 로딩만 `_common.py`, `_datasets.py`로 분리했습니다.
import만 해서는 데이터 다운로드나 학습이 시작되지 않습니다.

| 계열 | 실행 파일 | 목적 / 데이터 |
| --- | --- | --- |
| 트리 | xgboost, catboost, lightgbm, adaboost | 기본 Wine 회귀; Iris 분류도 지원 |
| 신경망 | mlp | Wine 회귀 / Iris 분류 |
| 선형·신경망 | linear_regression, perceptron, adaline | Wine 회귀 / Iris 분류 / 이진 Iris 분류 |
| Naive Bayes | naive_bayes | Iris 분류; Gaussian, Multinomial, Bernoulli, Complement |
| 변수 선택 | wrapper_selection, regularized_selection | Iris 분류 / Wine 회귀 |
| 군집 | kmeans, dbscan, hdbscan | Iris 수치 특성만 적합 |
| 이웃 거리 | k_distance | Iris의 k번째 이웃 거리 |
| 차원 축소 | pca, lda, umap, tsne, mds, som | Iris; LDA만 타깃 라벨을 적합에 사용 |
| TensorFlow Dense | tf_classification, tf_regression | Iris / Wine |
| TensorFlow RNN | tf_rnn | Libras의 독립 손동작 시퀀스 분류 |
| TensorFlow 예측 | tf_time_series | Bike Sharing 일별 대여 수 예측 |
| 연관규칙·네트워크 | apriori, sna | Retail 영수증 장바구니 / 상품 공동구매 edge |
| 전용 딥러닝 | tf_cnn, tf_simple_rnn, tf_lstm, tf_gru | Optdigits 공식 분할 / Libras 독립 시퀀스 |
| 추가 연관규칙 | fp_growth, eclat | Retail 영수증 장바구니 |

이 스켈레톤 범위는 구현된 학습·분석 모델입니다. 통계 가설검정·확률분포는
학습 모델의 파이프라인을 사용하지 않으므로 기존 전용 예제를 유지합니다.

## UCI 원본과 자료 변환

공식 UCI ZIP에서 지정한 파일만 읽으며 캐시는 저장소의 `.uci-cache/`에 둡니다.
`UCI_SKELETON_CACHE` 환경 변수 또는 각 `main(cache_dir=...)`로 위치를 바꿀 수 있습니다.
첫 실행에는 네트워크가 필요하고, 이후에는 캐시를 재사용합니다. ZIP을 임의 경로로 풀지 않습니다.
실패를 합성 자료로 조용히 대체하지 않습니다.

| 자료 | 실제 선택 자료 | 출처·라이선스 |
| --- | --- | --- |
| Wine Quality | red wine 원본 1,599행·11특성; quality 회귀 | Cortez 등(2009), [UCI 186](https://archive.ics.uci.edu/dataset/186/wine+quality), CC BY 4.0 |
| Iris | 150행·4특성·3종류; Adaline만 두 종류 100행 | Fisher(1936), [UCI 53](https://archive.ics.uci.edu/dataset/53/iris), CC BY 4.0 |
| Bike Sharing | day.csv의 731일, 날짜 순 cnt 한 열 | Fanaee-T(2013), [UCI 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), CC BY 4.0 |
| Libras Movement | 360동작, 프레임마다 x/y 좌표가 교대로 나오는 90특성을 (45,2)로 복원 | Dias·Peres·Bíscaro(2009), [UCI 181](https://archive.ics.uci.edu/dataset/181/libras+movement), CC BY 4.0 |
| Online Retail | 취소·비양수 수량/가격·우편요금 등 비상품 코드를 제외한 영수증 | Chen(2015), [UCI 352](https://archive.ics.uci.edu/dataset/352/online+retail), CC BY 4.0 |

Bike Sharing에서는 `cnt = casual + registered`인 두 구성 성분을 예측 특성에 넣지 않습니다.
원 날짜를 먼저 분리하고 각 구간 안에서만 입력/타깃 창을 만듭니다. 중복·누락 날짜를 거부합니다.
Libras는 서로 다른 동작 표본을 분리하며 tabular 열을 임의로 시간축으로 바꾸지 않습니다.
원본은 여러 동작을 같은 수행자에게서 수집했으므로 이 표본 분할은 새 수행자에 대한 검증이 아닙니다.

Retail은 원본 전체를 읽고 원 시간 순 영수증 1,000개를 **그룹 전체**로 선택합니다.
선택한 영수증에서 거래 빈도가 높은 30개 상품으로 제한하고, 고유 상품이 둘 이상인 영수증을 사용합니다.
취소·반품·수량 가중 여부와 상품 제한은 분석 정의에 영향을 줍니다.
Apriori support의 분모는 이 정제된 장바구니 수이며, 전체 Retail 거래 수가 아닙니다.
UCI 부분 자료에서 규칙을 확인할 수 있도록 Apriori만 기본 `min_support=0.05`를 명시적으로 적용합니다.
나머지 값은 기존 JSON을 읽고, 사용자 overrides가 이 스켈레톤 기본값보다 우선합니다. 기존 JSON 파일은 수정하지 않습니다.
SNA edge의 가중치는 같은 상품 쌍이 등장한 영수증 수입니다. 이는 상품 공동구매 관계입니다.

## 학습과 결과 계약

지도 학습은 Profile의 `test_size`, `random_state`, `stratify`를 사용합니다.
`param_grid`가 있으면 학습 자료의 CV에서만 모델을 선택합니다.
원본의 0.2/42를 전 모델에 고정하지 않고 모델별 JSON 값을 사용합니다.
전처리 방법도 JSON의 `scaling`을 따르므로 트리의 기본값은 none일 수 있습니다.
Naive Bayes의 Multinomial/Complement 예제는 Iris의 원 비음수 특성을 보존합니다.

지도·일반 분석 스켈레톤은 `SkeletonOutputDTO`를 반환합니다.
`metrics`, `details`, `predictions`, `train_indices`, `test_indices`를 수치/자료로 확인할 수 있습니다.
`model`은 Basepipeline 객체이고 `encoder`는 학습 라벨 인코더이며 둘 다 JSON에서 제외됩니다.
지도 결과의 `result.predict(X_new)`는 전처리와 원 라벨 복원을 함께 수행합니다.
`result.to_json(indent=2)`는 수치 결과 저장용입니다.
TensorFlow·Apriori·SNA는 기존 구현의 전용 Output DTO를 반환합니다.

군집·축소·이웃 거리의 `metrics`는 전체 자료의 탐색/기술 지표입니다.
보류 일반화 성능처럼 해석하지 않습니다. t-SNE/MDS의 학습 좌표를 새 입력의 좌표로 반환하지 않습니다.
PCA/LDA/UMAP/SOM만 기존 구현이 지원하는 transform을 재사용합니다.

## 실행

저장소 루트에서 기존 모델 환경과 추가 Excel 엔진을 준비합니다.

```bash
python -m pip install -r requirements-uci-skeletons.txt
python -m src.MachineLearning.skeleton.xgboost
python -m src.MachineLearning.skeleton.pca
python -m src.MachineLearning.skeleton.apriori
```

목적이나 레시피를 바꾸는 예:

```python
from src.MachineLearning.skeleton.xgboost import main, load_data

result = main(task="classification", overrides={"n_estimators": 50})
data = load_data(task="classification")
print(result.predict(data.drop(columns="target").iloc[:3]))
print(result.to_json(indent=2))
```

```python
from src.MachineLearning.skeleton.naive_bayes import main
result = main(overrides={"variant": "multinomial"})

from src.MachineLearning.skeleton.tf_rnn import main
result = main(overrides={"cell": "gru", "epochs": 5})

from src.MachineLearning.skeleton.tf_time_series import main
result = main(overrides={"input_width": 14, "horizon": 3, "epochs": 5})
```

TensorFlow는 기존 별도 requirements 환경에서 실행합니다.
macOS Apple Silicon에서는 동일 버전의 `tensorflow==2.16.2`를 사용해 확인했고,
Linux용 기존 파일은 `tensorflow-cpu==2.16.2`를 지정합니다. 일반 분석 환경의 NumPy 2 핀과 섞지 않습니다.

## 검증

- 실제 UCI 원본 5종에서 일반 분석 스켈레톤 23개를 기본 설정으로 실행했습니다.
- 분류/회귀의 추가 목적 7개와 Naive Bayes 추가 variant 3개도 실제 자료에서 실행했습니다.
- TensorFlow 4개는 별도 환경에서 실제 자료로 2 epoch 학습·보류 예측·DTO 직렬화를 확인했습니다.
  이 검사는 짧은 실행 검증이며 모델 품질이나 기본 30 epoch의 성능을 주장하지 않습니다.
- 오프라인 계약 검사: import 시 학습/다운로드 없음, 33개 함수 흐름·메타데이터·docstring,
  전처리의 학습 행 평균·CV, 날짜/시퀀스 순서, 완전한 영수증 그룹·취소 제거, 미지원 transform 거부.
- 생성한 스켈레톤에는 시각화 import·그래프 생성·그림 파일 저장이 없습니다.
- Python 3.7 구문도 검사합니다. 이번 실행 환경은 Apple Silicon Python 3.12.14이며,
  이번 스켈레톤을 실제 Python 3.7 또는 ADP 시험 서버에서 실행하지는 않았습니다.

모든 모델과 Profile의 상세 명세는 기존 모델별 문서와 구현 파일을 따릅니다.

## 별도 딥러닝·연관규칙 확장

CNN 1D/2D·전용 SimpleRNN/LSTM/GRU·FP-Growth/Eclat을 각각 별도 구현/레시피/스켈레톤 파일로 추가했습니다.
[전체 필드·아키텍처·UCI 입력·검증 명세](DEEP_ASSOCIATION_MODULES.md)를 참고하세요.
