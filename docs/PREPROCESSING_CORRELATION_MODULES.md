# Feature 기반 전처리·상관분석 명세

전처리 기법 정의와 Feature 선택/처리 순서를 분리합니다. 각 기법은 미적합 변환기를
생성하고, 기존 `FeatureSpec`은 대상 컬럼과 순서가 있는 sklearn `Pipeline`을 선언합니다.
`feature_pipeline.py`의 `ColumnTransformer`는 그룹을 결합합니다. 모델 학습에는
이를 모델 앞에 붙인 Pipeline 전체를 CV에 전달해야 합니다. 변환된 전체 학습 자료를
먼저 만든 뒤 그 자료에서 CV를 수행하면 전처리 누수가 발생합니다.

## Feature 처리 계약

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.Preprocessing.skeleton.feature_pipeline import load_data, build_pipeline

X_train, X_test, y_train, y_test = load_data()
features, unfitted = build_pipeline()
model = Pipeline([('features', unfitted), ('model', LogisticRegression(max_iter=1000))])
search = GridSearchCV(model, {'model__C': [0.1, 1.0]}, cv=3).fit(X_train, y_train)
predictions = search.predict(X_test)
```

- 수치형 Feature: 결측치 대치 → 스케일링; 범주형 Feature: 결측치 대치 → 인코딩.
- `make_feature_spec(name, columns, steps)`의 steps는 `(단계명, 미적합 변환기)` 순서 목록입니다.
- `build_feature_preprocessor`는 적합하지 않습니다. `run_feature_preprocessing`은 전달받은
  X_train만 적합하고 X_new에는 transform만 적용합니다. 이 runner는 개별 학습/보류 변환용입니다.
- 타깃은 먼저 X에서 분리합니다. `target_columns`에 선언한 열의 Feature 할당을 거부하고,
  타깃 이름이 선언되면 remainder=passthrough도 거부합니다. 미선언 타깃은 자동 추측하지 않습니다.
- 그룹 이름/원 컬럼 이름은 고유해야 합니다. 그룹 간 컬럼 중복은 기본 거부하고
  명시적 allow_overlap=True에서만 허용합니다. 출력 열 이름은 그룹 접두사를 포함합니다.
- 결측 처리는 학습 값에서만 계산합니다. 비상수 대치는 학습 열 전체가 결측이면 거부합니다.
  상수 대치는 전체 결측 열도 보존합니다. indicator는 학습 시 결측이 관측된 열에 대해서만 생성합니다.
- 수치 변환의 무한대·복소수, Box–Cox의 비양수, power 학습의 상수 열은 거부합니다.
- 범주형은 결측을 통일한 뒤 문자열로 정규화합니다. 한 입력 내 서로 다른 범주가 동일
  문자열로 충돌하면 거부합니다. 결측 대치가 필요한 인코딩 입력은 먼저 imputation을 적용합니다.
- 미지 범주는 onehot에서 0 벡터, ordinal에서 음수 unknown_value, frequency에서 지정 빈도;
  handle_unknown=error이면 거부합니다. drop_first와 미지 범주를 함께 쓰면 기준 범주와 0 벡터가 겹칠 수 있습니다.
- ordinal의 자동 문자열 정렬은 의미 있는 서열을 뜻하지 않습니다. 서열 변수는 category_orders를
  원 열 순서대로 명시합니다. frequency는 학습 빈도이며 타깃을 사용하지 않습니다.

## 위치 임베딩 RoPE

`src/Preprocessing/positional/rope.py`는 NumPy 기반 임베딩 회전이며 표 범주형 인코더가 아닙니다.
표 FeatureSpec 연결에서는 RoPE를 거부하고 시퀀스 임베딩 단계에 별도로 적용합니다.
입력은 `(..., sequence, embedding)`이며 마지막 축의 선두 rotary_dim(양의 짝수)만 회전합니다.
`theta_j = base^(-2j/rotary_dim)`, `angle = position/position_scale * theta_j`입니다.
interleaved는 인접 짝, split_half는 앞/뒤 절반을 짝짓습니다. 이 layout은 모델의 가중치 배치와
일치해야 합니다. positions는 배치/헤드가 공유하는 1차원 위치이며 생략하면 arange+offset입니다.
위치 회전에는 적합할 통계가 없으며 inverse=True는 동일 위치/설정에서 역회전합니다.
출력은 부동소수 입력 dtype을 유지합니다. 전체 Transformer/attention 학습 또는 TensorFlow/torch
연산 그래프를 구현하는 모듈은 아닙니다. Libras skeleton은 고정 seeded 투영으로 16차원
임베딩을 만들어 회전·노름 보존을 확인합니다. 성능 비교용 학습 모델로 해석하지 않습니다.
수식의 근거는 [RoFormer 원 논문](https://arxiv.org/abs/2104.09864)입니다.

## 수치형·범주형 상관분석

수치형 파일은 각각 Pearson(선형), Spearman(순위 단조), Kendall tau-b(동률 보정)를 구현합니다.
단일 쌍과 행렬 API를 제공합니다. Series 쌍은 인덱스가 같아야 하고, omit은 대응 결측 쌍을
함께 제거합니다. 상수·표본 수 부족 쌍은 단일 API에서 오류, 행렬에서는 NaN과 경고입니다.
행렬은 각 쌍의 n_matrix, 원 p, 보정 p와 보정 유의성 행렬을 함께 반환합니다. 대각선 p는
NaN이며 상삼각의 고유 검정만 Holm/Bonferroni/BH 또는 none으로 보정합니다.
쌍별 결측 제거의 표본이 다르면 계수 행렬의 양의 준정부호(PSD)를 보장하지 않습니다.

Pearson의 p는 정규 독립 관측에서 영 상관 귀무가설에 대한 beta 분포 검정입니다.
신뢰구간은 n>3에서 양측 Fisher 근사이며 이는 내부 추론 절차입니다.
요청한 공개 상관계수 변환 API는 Fisher z가 아니라 아래 행렬 정리 API입니다.
Spearman의 기본 p는 점근 근사로, n≤500에서 경고합니다. permutation은 고정 random_state로
두 번째 순위의 무작위 순열을 만들고 +1 보정합니다. 표본의 교환 가능성이 필요하며
전수 exact 검정이 아닙니다. Kendall exact는 동률을 허용하지 않습니다. 구버전 SciPy가
방향별 exact를 제공하지 않으면 오류를 반환하며 임의로 양측 p의 보수로 바꾸지 않습니다.

범주형은 원 라벨로 분할표를 만듭니다. Cramér V는 명목 범주의 비음수 연관 강도이며
기본 편향 보정은 계수에만 적용합니다. p는 원 분할표의 무보정 Pearson χ² 검정입니다.
Phi는 2×2 전용 부호 있는 계수입니다. 0/1 외 라벨은 positive_x/positive_y를 명시해야 하며
한 축의 positive를 뒤집으면 부호가 바뀝니다. p는 χ² 또는 양측 Fisher입니다.
기대빈도 5 미만이면 χ² 근사 경고를 제공합니다. 관측 독립성·자료 설계·다중비교를 확인해야
하며 상관을 인과 또는 모델 일반화 성능으로 해석하지 않습니다.

## 상관행렬 변환·정리

`run_correlation_matrix_transform`은 정방 대칭 [-1,1] 계수, 대각선 1 및 행/열 이름/순서를
검증합니다. tidy는 고유 상삼각 쌍을 절댓값 내림차순으로 정리합니다. upper/lower는 사용하지
않는 칸을 NaN으로 마스킹합니다. min_abs_coefficient와 선택 max_p_value로 필터링합니다.
거리 변환은 signed `sqrt(2*(1-r))`, absolute `sqrt(2*(1-|r|))`입니다. 항상 대각선 0입니다.
입력 PSD를 점검하며 비PSD·결측 또는 필터링 때문에 완전한 거리 공리를 보장하지 못하는
경우 경고합니다. 절대값 거리의 metric 조건도 자동 보장하지 않습니다.
미정의 계수를 0 상관으로 바꾸지 않습니다. allow_missing=False가 기본이고 True에서는
미정의 쌍을 보존/제외합니다. p 필터에는 동일 이름/순서/shape의 p 행렬이 필요합니다.

## 경로·레시피·UCI skeleton

13개 모듈은 각각 전용 Profile/Output DTO/JSON과 기존 7개 섹션, Attributes,
Args/Returns/Raises를 유지합니다. Profile 생략 시 JSON을 매번 읽고 overrides를 적용합니다.
명시적 Profile과 레시피 옵션은 동시 지정하지 않습니다. 적합 객체는 DTO의 transformer로
접근하고 사전/JSON에서는 제외합니다. DataFrame/배열 수치 결과는 DTO 직렬화를 지원합니다.

| 위치 | 모듈 |
| --- | --- |
| src/Preprocessing/numeric | scaling, imputation, power |
| src/Preprocessing/categorical | imputation, encoding (onehot/ordinal/frequency) |
| src/Preprocessing/positional | rope |
| src/Preprocessing | feature_pipeline |
| src/stats/correlation/numeric | pearson, spearman, kendall |
| src/stats/correlation/categorical | cramers_v, phi |
| src/stats/correlation | matrix_transform |
| src/Preprocessing/skeleton | 수치 3, 범주 2, RoPE 1, Feature 통합 1 |
| src/stats/skeleton | 상관 5, 행렬 변환 1; 기존 분포 예제도 유지 |

수치 전처리는 Wine 원 특성에서 타깃을 제외하고 먼저 80/20 분할합니다.
범주 및 통합 Feature는 [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult)의
공식 train/test에서 각각 첫 1,000행을 사용합니다. `?`는 결측으로 읽고 income의 test 말미
마침표를 제거합니다. fnlwgt는 이 간단한 Feature 예제에서 제외합니다. 탐색 상관은 Iris 원
수치 4특성 또는 Adult train의 원 범주를 사용합니다. 이 부분 자료의 통계 검정은 모집단
표본 설계/가중을 반영한 추론이 아닙니다. Adult 원본은 CC BY 4.0입니다.
공유 로더는 import에서 다운로드하지 않으며 .uci-cache 또는 UCI_SKELETON_CACHE를 사용합니다.

```bash
python -m src.Preprocessing.skeleton.feature_pipeline
python -m src.Preprocessing.skeleton.categorical_encoding
python -m src.Preprocessing.skeleton.rope
python -m src.stats.skeleton.pearson
python -m src.stats.skeleton.phi
python -m src.stats.skeleton.correlation_matrix_transform
```

행렬 skeleton 기본은 tidy이며 `main(overrides={'mode':'distance'})` 등으로 변경합니다.
범주형 encoding의 방법 재정의는 전용 skeleton의 main(overrides={'method':'frequency'})로
전달합니다. Feature 통합 skeleton의 overrides는 조합 Profile 전용이며 개별 기법 설정은
build_pipeline에서 해당 변환기 Profile로 명시합니다.

기법·호환 근거: [scikit-learn 전처리](https://scikit-learn.org/stable/modules/preprocessing.html),
[SciPy Pearson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html),
[Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html),
[Kendall](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html),
[분할표 association](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html).

## 전체 Profile·DTO 필드

아래는 각 모듈 자체의 전체 필드 명세와 기본값입니다. runtime transformer는 JSON 제외됩니다.

### numeric-scaling — `src.Preprocessing.numeric.scaling`

#### NumericScalingOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `preprocessed_feature` | `Any` | `반환 결과` | 학습 수치 Feature 변환 행렬. |
| `new_preprocessed_feature` | `Any` | `반환 결과` | 새 자료 변환 또는 None; 재적합하지 않음. |
| `input_features` | `List[str]` | `반환 결과` | 원 열 이름과 순서. |
| `output_features` | `List[str]` | `반환 결과` | 출력 열 이름과 순서. |
| `parameters` | `Dict[str, Any]` | `반환 결과` | 적용한 Profile. |
| `transformer` | `Any` | `반환 결과` | 적합 변환기; JSON에서 제외. |
| `n_samples` | `int` | `반환 결과` | 학습 행 수. |
| `warnings` | `List[str]` | `반환 결과` | 해석 정보. |

#### NumericScalingProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `'standard'` | none/standard/minmax/robust/maxabs/normalize; 기본 'standard'. |
| `center` | `bool` | `True` | Standard/Robust 중심화; 기본 True. |
| `scale` | `bool` | `True` | Standard/Robust 척도 정규화; 기본 True. |
| `clip` | `bool` | `False` | MinMax 새 입력 clipping; 기본 False. |
| `norm` | `str` | `'l2'` | Normalizer l1/l2/max; 기본 'l2'. |

### numeric-imputation — `src.Preprocessing.numeric.imputation`

#### NumericImputationOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `preprocessed_feature` | `Any` | `반환 결과` | 학습 수치 Feature 변환 행렬. |
| `new_preprocessed_feature` | `Any` | `반환 결과` | 새 자료 변환 또는 None; 재적합하지 않음. |
| `input_features` | `List[str]` | `반환 결과` | 원 열 이름과 순서. |
| `output_features` | `List[str]` | `반환 결과` | 출력 열 이름과 순서. |
| `parameters` | `Dict[str, Any]` | `반환 결과` | 적용한 Profile. |
| `transformer` | `Any` | `반환 결과` | 적합 변환기; JSON에서 제외. |
| `n_samples` | `int` | `반환 결과` | 학습 행 수. |
| `warnings` | `List[str]` | `반환 결과` | 해석 정보. |

#### NumericImputationProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `strategy` | `str` | `'median'` | mean/median/most_frequent/constant; 기본 'median'. |
| `fill_value` | `float` | `0` | constant의 유한 대치 값; 기본 0. |
| `add_indicator` | `bool` | `False` | 학습 결측 열 indicator 추가; 기본 False. |

### numeric-power — `src.Preprocessing.numeric.power`

#### NumericPowerOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `preprocessed_feature` | `Any` | `반환 결과` | 학습 수치 Feature 변환 행렬. |
| `new_preprocessed_feature` | `Any` | `반환 결과` | 새 자료 변환 또는 None; 재적합하지 않음. |
| `input_features` | `List[str]` | `반환 결과` | 원 열 이름과 순서. |
| `output_features` | `List[str]` | `반환 결과` | 출력 열 이름과 순서. |
| `parameters` | `Dict[str, Any]` | `반환 결과` | 적용한 Profile. |
| `transformer` | `Any` | `반환 결과` | 적합 변환기; JSON에서 제외. |
| `n_samples` | `int` | `반환 결과` | 학습 행 수. |
| `warnings` | `List[str]` | `반환 결과` | 해석 정보. |

#### NumericPowerProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `'yeo-johnson'` | yeo-johnson/box-cox; Box-Cox는 양수; 기본 'yeo-johnson'. |
| `standardize` | `bool` | `True` | 변환 후 학습 기준 표준화; 기본 True. |

### categorical-imputation — `src.Preprocessing.categorical.imputation`

#### CategoricalImputationOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `preprocessed_feature` | `Any` | `반환 결과` | 학습 자료 변환 행렬; OneHot는 sparse일 수 있음. |
| `new_preprocessed_feature` | `Any` | `반환 결과` | 새 자료 변환 또는 None; 재적합하지 않음. |
| `input_features` | `List[str]` | `반환 결과` | 원 열 이름과 순서. |
| `output_features` | `List[str]` | `반환 결과` | 출력 열 이름과 순서. |
| `parameters` | `Dict[str, Any]` | `반환 결과` | 적용한 Profile. |
| `transformer` | `Any` | `반환 결과` | 적합 변환기; JSON에서 제외. |
| `n_samples` | `int` | `반환 결과` | 학습 행 수. |
| `warnings` | `List[str]` | `반환 결과` | 해석 정보. |

#### CategoricalImputationProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `strategy` | `str` | `'most_frequent'` | most_frequent/constant; 기본 'most_frequent'. |
| `fill_value` | `str` | `'(missing)'` | constant의 명시적 대치 범주; 기본 '(missing)'. |

### categorical-encoding — `src.Preprocessing.categorical.encoding`

#### CategoricalEncodingOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `preprocessed_feature` | `Any` | `반환 결과` | 학습 자료 변환 행렬; OneHot는 sparse일 수 있음. |
| `new_preprocessed_feature` | `Any` | `반환 결과` | 새 자료 변환 또는 None; 재적합하지 않음. |
| `input_features` | `List[str]` | `반환 결과` | 원 열 이름과 순서. |
| `output_features` | `List[str]` | `반환 결과` | 출력 열 이름과 순서. |
| `parameters` | `Dict[str, Any]` | `반환 결과` | 적용한 Profile. |
| `transformer` | `Any` | `반환 결과` | 적합 변환기; JSON에서 제외. |
| `n_samples` | `int` | `반환 결과` | 학습 행 수. |
| `warnings` | `List[str]` | `반환 결과` | 해석 정보. |

#### CategoricalEncodingProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `'onehot'` | onehot/ordinal/frequency; 기본 'onehot'. |
| `handle_unknown` | `str` | `'ignore'` | ignore/error; 기본 'ignore'. |
| `drop_first` | `bool` | `False` | OneHot 첫 학습 범주 제거; 기본 False. |
| `sparse_output` | `bool` | `False` | OneHot sparse 출력; 기본 False. |
| `unknown_value` | `int` | `-1` | Ordinal 미지 범주 값; 음수; 기본 -1. |
| `category_orders` | `Optional[List[List[str]]]` | `None` | 원 열 순서의 명시적 범주 목록; None이면 문자열 학습 순서; 기본 None. |
| `unknown_frequency` | `float` | `0` | Frequency 미지 범주 값 [0,1]; 기본 0. |

### rope — `src.Preprocessing.positional.rope`

#### RoPEOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `preprocessed_feature` | `numpy.ndarray` | `반환 결과` | 회전한 임베딩; 원 shape 유지. |
| `positions` | `numpy.ndarray` | `반환 결과` | 적용한 1차원 위치; 배치/헤드 간 공유. |
| `inverse_frequencies` | `numpy.ndarray` | `반환 결과` | 회전 쌍별 base^(-2i/rotary_dim). |
| `parameters` | `Dict[str, Any]` | `반환 결과` | 적용 Profile. |
| `transformer` | `Any` | `반환 결과` | 재사용 sklearn 변환기; JSON 제외. |
| `warnings` | `List[str]` | `반환 결과` | 임베딩 해석 정보. |

#### RoPEProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `rotary_dim` | `Optional[int]` | `None` | 회전할 선두 짝수 차원; None이면 전체 embedding. |
| `base` | `float` | `10000.0` | 주파수 밑, 유한한 1 초과; 기본 10000. |
| `layout` | `str` | `'interleaved'` | interleaved 또는 split_half; 기본 interleaved. |
| `offset` | `int` | `0` | 기본 위치 시작값, 0 이상; 기본 0. |
| `position_scale` | `float` | `1.0` | 위치를 나눌 양수 척도; 기본 1. |
| `inverse` | `bool` | `False` | 음의 각도로 역회전할지 여부; 기본 False. |

### feature-preprocessing — `src.Preprocessing.feature_pipeline`

#### FeaturePreprocessingOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `preprocessed_feature` | `Any` | `반환 결과` | 학습 Feature 변환 행렬. |
| `new_preprocessed_feature` | `Any` | `반환 결과` | 새 Feature 변환 또는 None. |
| `input_features` | `List[str]` | `반환 결과` | 원 열 이름과 순서. |
| `output_features` | `List[str]` | `반환 결과` | 그룹/변환 출력 열 이름과 순서. |
| `feature_specs` | `List[Dict[str, Any]]` | `반환 결과` | 그룹별 원 컬럼과 처리 단계 설명. |
| `parameters` | `Dict[str, Any]` | `반환 결과` | Feature 조합 Profile. |
| `transformer` | `Any` | `반환 결과` | 학습 자료에 적합한 ColumnTransformer; JSON 제외. |
| `n_samples` | `int` | `반환 결과` | 학습 행 수. |

#### FeaturePreprocessingProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `remainder` | `str` | `'drop'` | 지정하지 않은 열 drop/passthrough; 기본 drop. |
| `sparse_threshold` | `float` | `0.3` | ColumnTransformer sparse 결합 임계값 [0,1]; 기본 0.3. |
| `allow_overlap` | `bool` | `False` | 서로 다른 그룹의 열 중복 허용; 기본 False. |
| `target_columns` | `Optional[List[str]]` | `None` | Feature에서 제외할 타깃 이름; 기본 None. |

### correlation-pearson — `src.stats.correlation.numeric.pearson`

#### PearsonMatrixOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `반환 결과` | 기법 이름. |
| `correlation_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 계수 행렬; 미정의 값 NaN. |
| `p_value_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 원 p 행렬; 대각선 NaN. |
| `adjusted_p_value_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 지정 방법으로 보정한 p 행렬. |
| `n_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 각 쌍의 유효 표본 수. |
| `significance_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 보정 p < alpha, 대각선 False. |
| `feature_names` | `List[str]` | `반환 결과` | 원 열 순서. |
| `alpha` | `float` | `반환 결과` | 유의수준. |
| `warnings` | `List[str]` | `반환 결과` | 미정의 쌍·결측·추론 경고. |

#### PearsonOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `반환 결과` | 적용한 기법/추론 방법. |
| `coefficient` | `float` | `반환 결과` | 상관계수 [-1,1]. |
| `p_value` | `float` | `반환 결과` | 선택 대립가설의 p 값. |
| `reject_h0` | `bool` | `반환 결과` | p_value < alpha. |
| `alpha` | `float` | `반환 결과` | 적용 유의수준. |
| `alternative` | `str` | `반환 결과` | 대립가설 방향. |
| `n_samples` | `int` | `반환 결과` | 쌍별 유효 표본 수. |
| `confidence_interval` | `Optional[Tuple[float, float]]` | `반환 결과` | Pearson Fisher 근사 양측 구간; 다른 기법 None. |
| `warnings` | `List[str]` | `반환 결과` | 제거·추론 가정·수치 경고. |

#### PearsonProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.05` | 유의수준 (0,1); 기본 0.05. |
| `alternative` | `str` | `'two-sided'` | two-sided/less/greater; 기본 'two-sided'. |
| `nan_policy` | `str` | `'raise'` | raise/omit; omit은 쌍별 제거; 기본 'raise'. |
| `min_samples` | `int` | `3` | 쌍별 최소 유효 표본, 3 이상; 기본 3. |
| `max_pairs` | `int` | `10000` | 행렬의 상삼각 쌍 한도; 기본 10000. |
| `p_adjust` | `str` | `'holm'` | 행렬 다중비교 none/bonferroni/holm/fdr_bh; 기본 'holm'. |
| `confidence_interval` | `bool` | `True` | n>3에서 Fisher 근사 신뢰구간; 기본 True. |

### correlation-spearman — `src.stats.correlation.numeric.spearman`

#### SpearmanMatrixOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `반환 결과` | 기법 이름. |
| `correlation_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 계수 행렬; 미정의 값 NaN. |
| `p_value_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 원 p 행렬; 대각선 NaN. |
| `adjusted_p_value_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 지정 방법으로 보정한 p 행렬. |
| `n_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 각 쌍의 유효 표본 수. |
| `significance_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 보정 p < alpha, 대각선 False. |
| `feature_names` | `List[str]` | `반환 결과` | 원 열 순서. |
| `alpha` | `float` | `반환 결과` | 유의수준. |
| `warnings` | `List[str]` | `반환 결과` | 미정의 쌍·결측·추론 경고. |

#### SpearmanOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `반환 결과` | 적용한 기법/추론 방법. |
| `coefficient` | `float` | `반환 결과` | 상관계수 [-1,1]. |
| `p_value` | `float` | `반환 결과` | 선택 대립가설의 p 값. |
| `reject_h0` | `bool` | `반환 결과` | p_value < alpha. |
| `alpha` | `float` | `반환 결과` | 적용 유의수준. |
| `alternative` | `str` | `반환 결과` | 대립가설 방향. |
| `n_samples` | `int` | `반환 결과` | 쌍별 유효 표본 수. |
| `confidence_interval` | `Optional[Tuple[float, float]]` | `반환 결과` | Pearson Fisher 근사 양측 구간; 다른 기법 None. |
| `warnings` | `List[str]` | `반환 결과` | 제거·추론 가정·수치 경고. |

#### SpearmanProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.05` | 유의수준 (0,1); 기본 0.05. |
| `alternative` | `str` | `'two-sided'` | two-sided/less/greater; 기본 'two-sided'. |
| `nan_policy` | `str` | `'raise'` | raise/omit; omit은 쌍별 제거; 기본 'raise'. |
| `min_samples` | `int` | `3` | 쌍별 최소 유효 표본, 3 이상; 기본 3. |
| `max_pairs` | `int` | `10000` | 행렬의 상삼각 쌍 한도; 기본 10000. |
| `p_adjust` | `str` | `'holm'` | 행렬 다중비교 none/bonferroni/holm/fdr_bh; 기본 'holm'. |
| `p_value_method` | `str` | `'asymptotic'` | asymptotic/permutation; 기본 'asymptotic'. |
| `n_resamples` | `int` | `9999` | permutation 무작위 표본 수, 양수; 기본 9999. |
| `random_state` | `int` | `1234` | permutation 시드, 0 이상; 기본 1234. |

### correlation-kendall — `src.stats.correlation.numeric.kendall`

#### KendallMatrixOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `반환 결과` | 기법 이름. |
| `correlation_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 계수 행렬; 미정의 값 NaN. |
| `p_value_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 원 p 행렬; 대각선 NaN. |
| `adjusted_p_value_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 지정 방법으로 보정한 p 행렬. |
| `n_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 각 쌍의 유효 표본 수. |
| `significance_matrix` | `pandas.core.frame.DataFrame` | `반환 결과` | 보정 p < alpha, 대각선 False. |
| `feature_names` | `List[str]` | `반환 결과` | 원 열 순서. |
| `alpha` | `float` | `반환 결과` | 유의수준. |
| `warnings` | `List[str]` | `반환 결과` | 미정의 쌍·결측·추론 경고. |

#### KendallOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `반환 결과` | 적용한 기법/추론 방법. |
| `coefficient` | `float` | `반환 결과` | 상관계수 [-1,1]. |
| `p_value` | `float` | `반환 결과` | 선택 대립가설의 p 값. |
| `reject_h0` | `bool` | `반환 결과` | p_value < alpha. |
| `alpha` | `float` | `반환 결과` | 적용 유의수준. |
| `alternative` | `str` | `반환 결과` | 대립가설 방향. |
| `n_samples` | `int` | `반환 결과` | 쌍별 유효 표본 수. |
| `confidence_interval` | `Optional[Tuple[float, float]]` | `반환 결과` | Pearson Fisher 근사 양측 구간; 다른 기법 None. |
| `warnings` | `List[str]` | `반환 결과` | 제거·추론 가정·수치 경고. |

#### KendallProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.05` | 유의수준 (0,1); 기본 0.05. |
| `alternative` | `str` | `'two-sided'` | two-sided/less/greater; 기본 'two-sided'. |
| `nan_policy` | `str` | `'raise'` | raise/omit; omit은 쌍별 제거; 기본 'raise'. |
| `min_samples` | `int` | `3` | 쌍별 최소 유효 표본, 3 이상; 기본 3. |
| `max_pairs` | `int` | `10000` | 행렬의 상삼각 쌍 한도; 기본 10000. |
| `p_adjust` | `str` | `'holm'` | 행렬 다중비교 none/bonferroni/holm/fdr_bh; 기본 'holm'. |
| `method` | `str` | `'auto'` | auto/exact/asymptotic; exact는 동률 없음; 기본 'auto'. |

### correlation-cramers-v — `src.stats.correlation.categorical.cramers_v`

#### CramersVOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `반환 결과` | 계수/검정 계산 방법. |
| `coefficient` | `float` | `반환 결과` | Cramers V [0,1] 또는 양성 수준 기준 signed Phi [-1,1]. |
| `p_value` | `float` | `반환 결과` | 독립성 chi-square 또는 Phi Fisher 양측 p. |
| `reject_h0` | `bool` | `반환 결과` | p < alpha. |
| `alpha` | `float` | `반환 결과` | 적용 유의수준. |
| `n_samples` | `int` | `반환 결과` | 대응 유효 표본 수. |
| `observed` | `pandas.core.frame.DataFrame` | `반환 결과` | 원 범주 레벨 순서의 분할표. |
| `expected` | `pandas.core.frame.DataFrame` | `반환 결과` | 독립성 귀무 기대 빈도. |
| `statistic` | `float` | `반환 결과` | 원 Pearson chi-square 통계량. |
| `df` | `int` | `반환 결과` | 독립성 자유도. |
| `levels_x` | `List[Any]` | `반환 결과` | 행 범주 수준. |
| `levels_y` | `List[Any]` | `반환 결과` | 열 범주 수준. |
| `warnings` | `List[str]` | `반환 결과` | 희소 표·결측·해석 경고. |

#### CramersVProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.05` | 유의수준 (0,1); 기본 0.05. |
| `nan_policy` | `str` | `'raise'` | raise/omit; 대응 쌍 함께 제거; 기본 'raise'. |
| `bias_correction` | `bool` | `True` | 작은 표본 편향 보정; p 값은 원 Pearson chi-square; 기본 True. |

### correlation-phi — `src.stats.correlation.categorical.phi`

#### PhiOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `method` | `str` | `반환 결과` | 계수/검정 계산 방법. |
| `coefficient` | `float` | `반환 결과` | Cramers V [0,1] 또는 양성 수준 기준 signed Phi [-1,1]. |
| `p_value` | `float` | `반환 결과` | 독립성 chi-square 또는 Phi Fisher 양측 p. |
| `reject_h0` | `bool` | `반환 결과` | p < alpha. |
| `alpha` | `float` | `반환 결과` | 적용 유의수준. |
| `n_samples` | `int` | `반환 결과` | 대응 유효 표본 수. |
| `observed` | `pandas.core.frame.DataFrame` | `반환 결과` | 원 범주 레벨 순서의 분할표. |
| `expected` | `pandas.core.frame.DataFrame` | `반환 결과` | 독립성 귀무 기대 빈도. |
| `statistic` | `float` | `반환 결과` | 원 Pearson chi-square 통계량. |
| `df` | `int` | `반환 결과` | 독립성 자유도. |
| `levels_x` | `List[Any]` | `반환 결과` | 행 범주 수준. |
| `levels_y` | `List[Any]` | `반환 결과` | 열 범주 수준. |
| `warnings` | `List[str]` | `반환 결과` | 희소 표·결측·해석 경고. |

#### PhiProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.05` | 유의수준 (0,1); 기본 0.05. |
| `nan_policy` | `str` | `'raise'` | raise/omit; 대응 쌍 함께 제거; 기본 'raise'. |
| `positive_x` | `Any` | `None` | 첫 이진 변수의 양성 수준; None은 0/1 자료만; 기본 None. |
| `positive_y` | `Any` | `None` | 둘째 이진 변수의 양성 수준; None은 0/1 자료만; 기본 None. |
| `p_value_method` | `str` | `'chi_square'` | chi_square/fisher; Fisher는 양측 정확 p; 기본 'chi_square'. |

### correlation-matrix-transform — `src.stats.correlation.matrix_transform`

#### CorrelationMatrixTransformOutputDTO

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `mode` | `str` | `반환 결과` | 적용한 변환. |
| `transformed_matrix` | `Optional[pandas.core.frame.DataFrame]` | `반환 결과` | 삼각/거리 행렬; tidy는 None. |
| `pairs` | `pandas.core.frame.DataFrame` | `반환 결과` | feature_x/feature_y/coefficient/abs_coefficient/p_value, |r| 내림차순. |
| `n_features` | `int` | `반환 결과` | 원 변수 수. |
| `parameters` | `Dict[str, Any]` | `반환 결과` | 적용 Profile. |
| `warnings` | `List[str]` | `반환 결과` | PSD·결측·거리 해석 경고. |

#### CorrelationMatrixTransformProfile

| 필드 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `mode` | `str` | `'tidy'` | tidy/upper/lower/distance; 기본 tidy. |
| `distance_kind` | `str` | `'signed'` | signed는 sqrt(2(1-r)), absolute는 sqrt(2(1-|r|)); 기본 signed. |
| `min_abs_coefficient` | `float` | `0.0` | 보존할 최소 |r| [0,1]; 기본 0. |
| `max_p_value` | `Optional[float]` | `None` | p 필터 [0,1] 또는 None; 지정 시 p 행렬 필요. |
| `include_diagonal` | `bool` | `False` | tidy/upper/lower에서 자기 쌍 포함; distance 대각은 항상0. 기본 False. |
| `allow_missing` | `bool` | `False` | NaN 허용 여부; 허용 시 정리 표에서 제외, 행렬에서는 NaN 유지. 기본 False. |
| `tolerance` | `float` | `1e-08` | 대칭/대각/r 범위 오차 허용, 0 이상; 기본 1e-8. |
| `max_pairs` | `int` | `10000` | 중복 없는 쌍 수 한도; 기본 10000. |
