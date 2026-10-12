# Axes 기반 EDA·의사결정 시각화

35개 단일 pane 모듈은 `Profile → JSON → plot 함수 → 전용 Output DTO`를 사용합니다.
각 모듈은 기존 메타데이터·7개 섹션·전체 Attributes·Args/Returns/Raises 명세를 유지합니다.
한 호출은 하나의 사용자가 전달한 `ax`에만 그립니다. 숫자 계산과 그림을 분리해 확인할 수
있도록 data·diagnostics·parameters·warnings를 반환합니다. 회귀선/예측/잔차 시각화는 이번 범위에 없습니다.
QQ의 분위수 기준선, 시간 지연 scatter, 범주별 원 수치 분포는 EDA입니다.

## subplot 조합 계약

```python
from matplotlib import pyplot as plt
from src.Visualization.numeric.histogram import plot_histogram
from src.Visualization.categorical.grouped_distribution import plot_grouped_distribution
from src.Visualization.clustering.dendrogram import plot_dendrogram
from src.Visualization.relationships.correlation import plot_correlation
from src.stats.correlation.numeric.pearson import run_pearson_matrix

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
plot_histogram(X['age'], ax=axes[0, 0])
plot_grouped_distribution(X['age'], X['category'], ax=axes[0, 1])
plot_dendrogram(X_numeric_scaled, ax=axes[1, 0],
                overrides={'cut_height': 5.0, 'truncate_last': 20})
analysis = run_pearson_matrix(X_numeric)
result = plot_correlation(analysis.correlation_matrix,
                          analysis.adjusted_p_value_matrix,
                          ax=axes[1, 1], overrides={'max_p_value': .05})
fig.tight_layout()
# 저장·표시·닫기는 호출자가 결정합니다.
# fig.savefig('eda.png'); plt.show(); plt.close(fig)
```

- ax는 필수 기존 2D Matplotlib Axes입니다. plot 함수는 subplots/figure/gca/sca/show/save를 호출하지 않습니다.
  기존 내용을 자동으로 지우지 않으므로 재그리기는 호출자가 ax.clear() 여부를 결정합니다.
- heatmap/hexbin/rules의 색상 막대는 같은 figure의 별도 `cax`를 명시할 때만 그립니다.
  cax 생략 시 추가 축이 없으며 cax=ax 또는 다른 figure의 축은 거부합니다.
- 레이아웃·공유 축·GridSpec·폰트·스타일은 호출자가 소유합니다. 전역 rcParams를 바꾸지 않습니다.
  범주·수치 단위가 다른 차트에는 의미 없이 sharex/sharey를 적용하지 마세요.
- 반환 result.axes는 입력 ax와 동일하며 figure/axes/artists는 JSON 제외 runtime입니다.
  data는 그린 값 또는 분석 요약 전체이며 diagnostics는 분모·표본 수·표시 제한·기준을 명시합니다.
  일부 차트의 표시 한도는 계산/반환 전체를 줄이지 않습니다. 각 명세를 따르세요.
- 입력 DataFrame/Series는 변경하지 않습니다. Series 쌍의 인덱스는 일치해야 합니다.
  대부분 대응 결측은 함께 제거하고 제거 수를 보고합니다. 시간 lag/ACF/PACF는 결측을 거부하여
  삭제로 시간축을 압축하지 않습니다. 무한대·복소수·범주 문자열 충돌은 거부합니다.
- Profile 생략 시 모델별 plot-* JSON을 읽습니다. 명시적 Profile과 overrides/recipe_path 동시 지정은 오류입니다.
  결정 기준/집계 알고리즘은 각 파일에 있고 공통 코드는 축·입력·그림/DTO 계약만 다룹니다.

## 판단별 선택

| 판단 | 모듈 | 반환 근거 / 주의 |
| --- | --- | --- |
| 분포·꼬리·분위수 | numeric histogram/kde/ecdf/qq | 도수/밀도 격자·누적 비율·분위수 기준선; KDE bandwidth/경계 확인 |
| 범주 불균형·우선순위 | categorical count/pareto | 전체 유효 관측 분모, 미표시 관측 수, 누적 coverage |
| 집단 분포·산포 | categorical grouped_distribution | box/violin/strip; 각 집단 n·분위수·평균, box 이상치는 자동 삭제 기준 아님 |
| 범주 조합 | categorical contingency/mosaic | row/column/all/count 분모·원 빈도; mosaic 면적=공동 비율 |
| 집단별 발생률 | categorical category_rate | 양성 의미·n·Wilson 구간; 다중비교/인과를 자동 주장하지 않음 |
| 수집 품질·공동 결측 | quality missingness/missing_matrix/missing_association | 결측률/행 패턴/결측 지표 상관; 상수 지표 NaN, MCAR/MAR/MNAR 판정 없음 |
| 변수 관계·중복 | relationships scatter/hexbin/correlation | 그룹·밀도·기존 계수와 선택 p 필터; 미정의 셀 마스킹 |
| 계층 구조·군집 수 | clustering dendrogram/silhouette/k_distance/elbow | linkage·절단 군집·개별/평균 silhouette·이웃 거리·inertia 개선 |
| 차원 수·성분 해석 | decomposition explained_variance/loadings | 목표 누적분산 도달 여부·원 Feature 성분 계수 |
| 분류 평가·운영 임계값 | classification confusion/roc/precision_recall/threshold/calibration | 원 빈도·AUC/AP·모든 후보 비용·이항 빈 구간 n·Brier/ECE |
| Feature/학습량 검토 | selection feature_importance/validation_curve | 동일 scoring 기준 중요도·반복 변동·train–validation 차이 |
| 규칙 우선 검토 | association rules | support/confidence/lift 필터, lift 색·support 크기; 희소성 확인 |
| 연결 구조 | networks graph | 명시 Graph·degree 크기·재현 가능한 배치; 공간 거리는 관계 강도 아님 |
| 시간 패턴 | time_series timeline/lag/autocorrelation/seasonal | 후행 요약·동일 행 지연·ACF/PACF 근사 구간·요일/월/시간 분포 |

## 의사결정 해석과 계산 경계

### 군집·차원 축소

덴드로그램 입력을 observations/distance/linkage로 명시합니다. 사전 거리의 정방·대칭·비음수·영
대각을 검증합니다. ward/centroid/median은 euclidean 관측으로만 새 계산하며 임의 사전 거리로
이 방식을 수행하지 않습니다. 사전 linkage는 SciPy validity를 검사합니다. max_samples는 O(n²)
계층 계산/표시 제한입니다. truncate_last는 표시만 축소합니다. cut_height는 사용자 지정 후보이며
최적 k를 추정하지 않습니다. 결과 cluster_labels는 원 잎 순서의 fcluster distance 절단입니다.

silhouette은 2≤군집 수<n이고 소속 순서를 유지합니다. 기본 noise_label=-1을 제외한 뒤 계산하며,
precomputed에서는 동일 행/열을 함께 선택합니다. 모든 군집/표본 거리 기준을 동일하게 비교해야 합니다.
k_distance는 자기 자신 포함 k번째 이웃입니다. DBSCAN min_samples 관례와 맞춰 k를 선택하세요.
elbow는 이미 계산한 같은 자료의 inertia만 표시합니다. 입력 k를 정렬하지만 모델을 적합하지 않습니다.
설명분산의 목표가 주어진 성분으로 도달하지 않으면 components_to_target=None입니다.
성분 전체 부호는 임의이며 loadings 입력 components_와 sqrt(eigenvalue) 등 다른 정의를 구별하세요.

### 비율·상관·분류

범주 발생률은 명시 positive_label의 빈도와 Wilson 개별 이항 구간입니다. 관측 독립성·표본 설계를
확인하고 작은 n의 불확실성을 보세요. count/pareto 표시 한도 이후 범주도 전체 분모에 포함합니다.
contingency는 원 counts를 diagnostics에 반환합니다. 결측 상관은 값의 상관이 아닌 누락 지표의
상관이며 원인 추론이 아닙니다. correlation은 이미 계산한 대칭 계수와 p를 받아 마스킹합니다.
보정 p 사용 여부는 호출자의 명시적 선택입니다. NaN을 0 상관으로 채우지 않습니다.

ROC/PR/calibration/threshold는 이진 전용입니다. 큰 score가 양성임을 지정하고 원 라벨을 보존합니다.
ROC는 순위 AUC, PR은 sklearn average_precision이며 사다리꼴 면적이 아닙니다. PR 끝점에는
threshold가 없어 NaN을 반환합니다. ROC의 초기 임계값은 라이브러리 버전에 따라 inf일 수 있고
엄격한 DTO JSON에서는 null입니다. calibration 확률은 [0,1]이어야 합니다. 동일 폭 n_bins에서
빈 구간은 제외하고 구간별 n을 반환합니다. ECE는 구간 선택에 의존합니다.

threshold는 `score >= threshold` 규칙이며 모든 고유 점수와 양끝 후보를 계산합니다. max_thresholds는
표시 격자만 제한합니다. 비용=(FP×FP비용+FN×FN비용)/n이며 최소 비용 후보의 동률은 오름차순
첫 후보를 선택합니다. precision은 예측 양성이 없으면 NaN입니다. 비용은 지표 y축에 혼합하지
않고 전체 후보 data/diagnostics에서 확인합니다. 임계값/Feature 선택에는 검증 자료를 쓰고 최종
테스트는 분리하세요. Feature importance와 validation_curve는 사전 계산 결과만 그리며 모델을
재학습하지 않습니다. 입력 deviations의 의미를 명시하고 자동 CI로 해석하지 않습니다.

### 시간·레이아웃

시간 lag/ACF/PACF는 원 자료가 시간 순·동일 간격이라고 가정하며 결측은 거부합니다. timeline은
시간을 오름차순·고유로 요구하고 trailing mean은 이전/현재 행만 사용합니다. 날짜 누락을 자동
채우거나 단위를 재표본화하지 않습니다. seasonal은 미관측 기간을 만들지 않는 원 분포 요약입니다.
ACF/PACF confidence는 점별 근사 구간이며 그림의 band는 0 중심, data는 추정계수 중심 구간입니다.
다중 지연 검정 또는 모델 차수를 자동 확정하지 않습니다.

## UCI subplot skeleton 11개

`src/Visualization/skeleton/{numeric,categorical,quality,relationships,clustering,decomposition,
classification,selection,association,networks,time_series}.py`는
load_data → load_recipe → build_pipeline → main 흐름을 유지합니다.
`main(axes=existing_axes)`는 기존 축을 사용하며 추가/미사용 subplot은 변경하지 않습니다.
axes 생략 시 **예제 계층에서만** 레이아웃을 생성하고 미사용 칸을 숨깁니다. show/save는 호출자가 합니다.

```python
from matplotlib import pyplot as plt
from src.Visualization.skeleton.clustering import main
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
result = main(axes=axes, overrides={
    'dendrogram': {'cut_height': 7.0, 'truncate_last': 20},
    'k_distance': {'k': 5, 'eps': 0.6},
})
print(result.panels['silhouette'].diagnostics)
print(result.to_json(indent=2))
```

Wine: 원 수치 분포, Adult 공식 train 첫 1,000행: 범주/결측, Iris: 관계·군집·PCA·분류/선택,
Retail: 규칙/상품 연결망, Bike 일별: 시간 패턴을 사용합니다. 출처와 CC BY 4.0은
[기존 UCI 명세](UCI_SKELETONS.md) 및 [Adult 포함 명세](PREPROCESSING_CORRELATION_MODULES.md)를 따릅니다.
군집의 StandardScaler는 전체 자료 탐색 공간 정의이며 보류 일반화 검증이 아닙니다.
분류/선택 예제는 원 Iris를 train60%/validation20%/final-test20%로 나누고 전처리는 train에만 적합합니다.
그림·임계값/중요도 검토는 validation에서 수행하며 final-test는 사용하지 않습니다.
selection의 learning curve는 train 내부 CV에 전처리 포함 Pipeline을 전달합니다.
원 UCI 기존 모델 skeleton 46개와 시각화 조합 예제 11개는 별도이며 기존 모델 파일에 그림을 추가하지 않습니다.

```bash
python -m examples.eda_visualization_examples  # 고정 예제 자료, 네트워크 없는 35개 pane 실행
python -m src.Visualization.skeleton.categorical  # 실제 UCI, 첫 실행 다운로드
python -m src.Visualization.skeleton.clustering
python -m src.Visualization.skeleton.classification
```

## 전체 모듈·Profile·표시 자료 명세

모든 DTO는 figure/axes/artists/data/diagnostics/parameters/warnings 7필드를 가집니다.
figure·axes·artists는 runtime으로 JSON 제외하고, 나머지는 엄격한 JSON으로 직렬화합니다.
아래 data 컬럼·diagnostics는 각 모듈의 기본 계산 계약입니다. 행/열 이름을 가진 행렬은 이름을 보존합니다.

### numeric.histogram

왜도·다봉성·희귀 구간을 확인하고 변환/구간화 후보를 검토합니다.

`plot_histogram(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-numeric-histogram.json`. Profile: `NumericHistogramProfile`, DTO: `NumericHistogramOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `bins` | `int` | `20` | 양의 histogram 구간 수. 기본 20. |
| `density` | `bool` | `False` | True는 면적 1 밀도, False는 도수. 기본 False. |

표시/반환 data: `left`, `right`, `height`.

진단: `n`, `missing_removed`, `mean`, `median`.

단변량 결측 제거/표본 수와 입력 정의역을 결과에서 확인합니다.

### numeric.kde

히스토그램 구간 선택에 따른 착시를 비교하고 분포 구조를 탐색합니다.

`plot_kde(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-numeric-kde.json`. Profile: `NumericKdeProfile`, DTO: `NumericKdeOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `points` | `int` | `200` | 밀도 평가 격자 수, 2 이상. 기본 200. |
| `bandwidth` | `float` | `1.0` | Scott bandwidth에 곱할 양수 배율. 기본 1.0. |

표시/반환 data: `value`, `density`.

진단: `n`, `missing_removed`, `bandwidth_factor`.

KDE는 경계 밖에도 밀도를 배정하며 bandwidth에 따라 모양이 달라집니다.

### numeric.ecdf

특정 기준 이하의 비율과 분위수로 운영 기준을 검토합니다.

`plot_ecdf(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-numeric-ecdf.json`. Profile: `NumericEcdfProfile`, DTO: `NumericEcdfOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |

표시/반환 data: `value`, `cumulative_fraction`, `count`.

진단: `n`, `missing_removed`, `q25_q50_q75`.

단변량 결측 제거/표본 수와 입력 정의역을 결과에서 확인합니다.

### numeric.qq

꼬리·분포 가정의 이탈을 점검해 통계 검정과 변환을 선택합니다.

`plot_qq(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-numeric-qq.json`. Profile: `NumericQqProfile`, DTO: `NumericQqOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `distribution` | `str` | `'norm'` | SciPy 연속분포 이름; 기본 norm. 기본 'norm'. |

표시/반환 data: `theoretical`, `observed`.

진단: `n`, `missing_removed`, `reference_slope`, `reference_intercept`, `quantile_r`.

기준선은 분위수 비교용이며 회귀 모델 결과 또는 정규성 검정 p 값이 아닙니다.

### categorical.count

클래스 불균형·희귀 범주·범주 통합 필요성을 판단합니다.

`plot_count(categories, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-categorical-count.json`. Profile: `CategoricalCountProfile`, DTO: `CategoricalCountOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `max_categories` | `int` | `20` | 표시할 최대 범주 수; 초과 범주는 진단에 보고. 기본 20. |
| `proportion` | `bool` | `False` | 전체 유효 범주 표본 대비 비율. 기본 False. |

표시/반환 data: `category`, `count`, `fraction`, `displayed`.

진단: `n`, `missing_removed`, `omitted_rows`, `n_categories`.

단변량 결측 제거/표본 수와 입력 정의역을 결과에서 확인합니다.

### categorical.pareto

누적 기여를 기준으로 우선 검토할 범주를 선택합니다.

`plot_pareto(categories, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-categorical-pareto.json`. Profile: `CategoricalParetoProfile`, DTO: `CategoricalParetoOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `coverage` | `float` | `0.8` | 주요 범주 누적 비율 기준 (0,1]. 기본 0.8. |
| `max_categories` | `int` | `20` | 표시 한도; 누적 분모는 전체 유효 관측. 기본 20. |

표시/반환 data: `category`, `count`, `cumulative_fraction`.

진단: `n`, `missing_removed`, `categories_to_coverage`, `omitted_rows`.

단변량 결측 제거/표본 수와 입력 정의역을 결과에서 확인합니다.

### categorical.grouped_distribution

집단별 분포·중앙값·산포·소표본 차이를 비교합니다.

`plot_grouped_distribution(values, categories, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-categorical-grouped-distribution.json`. Profile: `CategoricalGroupedDistributionProfile`, DTO: `CategoricalGroupedDistributionOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `kind` | `str` | `'box'` | box/violin/strip; 같은 Axes에 범주별 수치 분포. 기본 'box'. |
| `max_categories` | `int` | `20` | 최대 범주; 초과하면 오류로 명시적 선택 요구. 기본 20. |
| `whisker` | `float` | `1.5` | box의 IQR whisker 배율. 기본 1.5. |
| `random_state` | `int` | `1234` | strip jitter seed. 기본 1234. |

표시/반환 data: `category`, `n`, `median`, `q25`, `q75`, `mean`.

진단: `n`, `missing_removed`, `outliers_are_not_errors`.

그룹 표본 수를 함께 확인하세요. box 이상치는 자동 삭제 기준이 아닙니다.

### categorical.contingency

범주 조합의 집중·조건부 구성 차이와 희소 셀을 판단합니다.

`plot_contingency(x, y, *, ax, profile=None, recipe_path=None, overrides=None, cax=None)`

레시피: `plot-categorical-contingency.json`. Profile: `CategoricalContingencyProfile`, DTO: `CategoricalContingencyOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `normalization` | `str` | `'row'` | count/row/column/all 분모. 기본 'row'. |
| `annotate` | `bool` | `True` | 셀 값 표시. 기본 True. |
| `max_categories` | `int` | `20` | 각 축의 최대 범주 수. 기본 20. |

표시/반환 data: `0`, `1`.

진단: `n`, `missing_removed`, `counts`, `normalization`.

비율과 원 빈도를 함께 보세요. 색 차이 자체는 유의성 검정이 아닙니다.

### categorical.mosaic

전체 구성과 범주 간 조건부 연관 패턴을 동시에 비교합니다.

`plot_mosaic(x, y, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-categorical-mosaic.json`. Profile: `CategoricalMosaicProfile`, DTO: `CategoricalMosaicOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `max_categories` | `int` | `12` | 각 축의 최대 범주 수. 기본 12. |
| `annotate` | `bool` | `True` | 셀 원 빈도를 표시. 기본 True. |

표시/반환 data: `0`, `1`.

진단: `n`, `missing_removed`, `area_denominator`.

면적은 공동 비율, 높이는 X 안에서의 Y 비율입니다.

### categorical.category_rate

범주별 발생률과 표본 수·불확실성으로 우선 검토 집단을 선택합니다.

`plot_category_rate(categories, outcome, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-categorical-category-rate.json`. Profile: `CategoricalCategoryRateProfile`, DTO: `CategoricalCategoryRateOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `positive_label` | `Any` | `1` | 양성 outcome 라벨; 원 의미를 명시. 기본 1. |
| `confidence` | `float` | `0.95` | Wilson 이항 신뢰수준 (0,1). 기본 0.95. |
| `max_categories` | `int` | `20` | 최대 범주 수. 기본 20. |

표시/반환 data: `category`, `positive_count`, `n`, `rate`, `lower`, `upper`.

진단: `n`, `missing_removed`, `positive_label`, `confidence`.

독립 이항 관측의 개별 구간이며 다중비교 보정·인과·일반화 평가를 뜻하지 않습니다.

### quality.missingness

결측 처리 우선순위와 완전 관측 손실 규모를 확인합니다.

`plot_missingness(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-quality-missingness.json`. Profile: `QualityMissingnessProfile`, DTO: `QualityMissingnessOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |

표시/반환 data: `feature`, `missing_count`, `missing_fraction`.

진단: `n_rows`, `complete_rows`.

결측률만으로 컬럼 삭제 또는 MCAR/MAR/MNAR를 결정하지 않습니다.

### quality.missing_matrix

결측 블록·동시 누락·행 순서 관련 패턴을 확인합니다.

`plot_missing_matrix(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-quality-missing-matrix.json`. Profile: `QualityMissingMatrixProfile`, DTO: `QualityMissingMatrixOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `max_rows` | `int` | `500` | 원 순서에서 표시할 최대 행 수. 기본 500. |

표시/반환 data: `a`, `b`, `c`, `d`.

진단: `n_rows`, `displayed_rows`, `missing_cells`.

원 행 순서를 보존합니다. 표시 한도 이후 행도 반환 data에 포함됩니다.

### quality.missing_association

함께 누락되는 Feature를 탐색해 공동 대치와 수집 과정을 검토합니다.

`plot_missing_association(values, *, ax, profile=None, recipe_path=None, overrides=None, cax=None)`

레시피: `plot-quality-missing-association.json`. Profile: `QualityMissingAssociationProfile`, DTO: `QualityMissingAssociationOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `annotate` | `bool` | `True` | 결측 지표 간 상관을 셀에 표시. 기본 True. |

표시/반환 data: `a`, `b`, `c`, `d`.

진단: `n_rows`, `undefined_features`, `missing_counts`.

결측 여부 지표의 상관입니다. 항상/절대 결측 열은 NaN이고 결측 발생 원인은 식별하지 않습니다.

### relationships.scatter

비선형 패턴·군집·이상 관측과 집단별 차이를 확인합니다.

`plot_scatter(x, y, groups=None, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-relationships-scatter.json`. Profile: `RelationshipsScatterProfile`, DTO: `RelationshipsScatterOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `max_points` | `int` | `5000` | 표시할 최대 대응 행, 원 순서. 기본 5000. |
| `size` | `float` | `16.0` | 양수 marker 크기. 기본 16.0. |

표시/반환 data: `x`, `y`, `group`.

진단: `n`, `displayed_rows`, `missing_removed`.

회귀선·예측·잔차를 생성하지 않는 원 변수 관계 탐색입니다.

### relationships.hexbin

대량 관측에서 겹침을 줄여 집중 영역과 희소 영역을 비교합니다.

`plot_hexbin(x, y, *, ax, profile=None, recipe_path=None, overrides=None, cax=None)`

레시피: `plot-relationships-hexbin.json`. Profile: `RelationshipsHexbinProfile`, DTO: `RelationshipsHexbinOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `gridsize` | `int` | `30` | 양의 육각 격자 수. 기본 30. |
| `log_counts` | `bool` | `False` | 색을 로그 빈도로 표시. 기본 False. |

표시/반환 data: `x_center`, `y_center`, `count`.

진단: `n`, `missing_removed`.

단변량 결측 제거/표본 수와 입력 정의역을 결과에서 확인합니다.

### relationships.correlation

강한 중복 Feature와 보정 유의성 조건을 함께 검토합니다.

`plot_correlation(matrix, p_values=None, *, ax, profile=None, recipe_path=None, overrides=None, cax=None)`

레시피: `plot-relationships-correlation.json`. Profile: `RelationshipsCorrelationProfile`, DTO: `RelationshipsCorrelationOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `annotate` | `bool` | `True` | 계수 숫자를 셀에 표시. 기본 True. |
| `mask_upper` | `bool` | `True` | 중복 상삼각을 숨김. 기본 True. |
| `max_p_value` | `Optional[float]` | `None` | 선택 p 필터; 보정 p 사용 권장. 기본 None. |

표시/반환 data: `a`, `b`, `c`, `d`.

진단: `n_features`, `p_filter`.

입력 계수와 p를 사용합니다. 시각화가 상관을 재추정하거나 추가 검정을 수행하지 않습니다.

### clustering.dendrogram

계층 병합 거리와 절단 후보로 군집 구조를 검토합니다.

`plot_dendrogram(values, labels=None, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-clustering-dendrogram.json`. Profile: `ClusteringDendrogramProfile`, DTO: `ClusteringDendrogramOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `method` | `str` | `'ward'` | single/complete/average/weighted/centroid/median/ward. 기본 'ward'. |
| `metric` | `str` | `'euclidean'` | SciPy pdist 거리; ward/centroid/median은 euclidean. 기본 'euclidean'. |
| `input_kind` | `str` | `'observations'` | observations/distance/linkage; 거리 행렬 명시. 기본 'observations'. |
| `cut_height` | `Optional[float]` | `None` | 선택 비음수 절단 높이. 기본 None. |
| `max_samples` | `int` | `1000` | O(n²) 계층 계산/표시의 최대 표본 수. 기본 1000. |
| `truncate_last` | `int` | `30` | 마지막 병합 단위 표시 수; 0이면 원 잎 전체. 기본 30. |

표시/반환 data: `left`, `right`, `distance`, `n_leaves`.

진단: `n_samples`, `displayed_leaves`, `leaf_order`, `cluster_labels`, `method`, `metric`.

거리/스케일/연결 기준에 민감합니다. 절단 높이는 탐색 기준이며 최적 군집 수를 자동 결정하지 않습니다.

### clustering.silhouette

음의 silhouette·군집별 산포와 평균으로 군집 분리를 비교합니다.

`plot_silhouette(values, labels, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-clustering-silhouette.json`. Profile: `ClusteringSilhouetteProfile`, DTO: `ClusteringSilhouetteOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `metric` | `str` | `'euclidean'` | sklearn 거리 또는 precomputed. 기본 'euclidean'. |
| `noise_label` | `Any` | `-1` | 제외할 밀도 군집 noise 라벨; None은 미제외. 기본 -1. |
| `max_samples` | `int` | `5000` | 전체 쌍 거리 계산의 최대 유효 표본. 기본 5000. |

표시/반환 data: `original_position`, `cluster`, `silhouette`.

진단: `mean_silhouette`, `negative_fraction`, `noise_excluded`, `n`.

noise 제외 후 군집 구조를 평가합니다. 서로 다른 거리/Feature 공간의 점수를 직접 비교하지 않습니다.

### clustering.k_distance

밀도 군집 eps 후보와 거리의 급격한 변화를 검토합니다.

`plot_k_distance(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-clustering-k-distance.json`. Profile: `ClusteringKDistanceProfile`, DTO: `ClusteringKDistanceOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `k` | `int` | `5` | 자기 자신 포함 k번째 이웃; 2 이상. 기본 5. |
| `metric` | `str` | `'euclidean'` | NearestNeighbors 거리. 기본 'euclidean'. |
| `eps` | `Optional[float]` | `None` | 사용자가 검토하는 비음수 DBSCAN eps. 기본 None. |

표시/반환 data: `rank`, `distance`.

진단: `k`, `self_included`, `fraction_within_eps`.

무릎은 eps 후보의 시각적 기준이며 자동 최적값이 아닙니다. scaling과 metric을 먼저 결정하세요.

### clustering.elbow

추가 군집 수에 따른 개선 규모를 비교합니다. 모델을 다시 적합하지 않습니다.

`plot_elbow(k_values, inertias, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-clustering-elbow.json`. Profile: `ClusteringElbowProfile`, DTO: `ClusteringElbowOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |

표시/반환 data: `k`, `inertia`.

진단: `relative_improvement`.

같은 학습 Feature·스케일·표본에서 계산한 inertia만 비교하세요. 무릎을 자동 최적 k로 표시하지 않습니다.

### decomposition.explained_variance

목표 누적분산에 필요한 성분 수와 차원 축소 손실을 비교합니다.

`plot_explained_variance(ratios, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-decomposition-explained-variance.json`. Profile: `DecompositionExplainedVarianceProfile`, DTO: `DecompositionExplainedVarianceOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `target_fraction` | `float` | `0.9` | 누적 설명분산 선택 기준 (0,1]. 기본 0.9. |

표시/반환 data: `component`, `ratio`, `cumulative_ratio`.

진단: `components_to_target`, `covered_fraction`.

미제공 성분의 분산은 추정하지 않습니다. 설명분산 자체는 예측 품질이 아닙니다.

### decomposition.loadings

성분을 구성하는 원 Feature와 부호·기여 패턴을 해석합니다.

`plot_loadings(components, feature_names=None, *, ax, profile=None, recipe_path=None, overrides=None, cax=None)`

레시피: `plot-decomposition-loadings.json`. Profile: `DecompositionLoadingsProfile`, DTO: `DecompositionLoadingsOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `annotate` | `bool` | `False` | 각 성분 계수 표시. 기본 False. |

표시/반환 data: `a`, `b`, `c`, `d`.

진단: `n_components`, `n_features`.

입력은 components_/명시 loading 행렬입니다. 성분의 전체 부호는 임의이며 서로 다른 적합 결과의 부호를 직접 비교하지 않습니다.

### classification.confusion

어느 클래스 사이에서 오분류가 발생하는지 비교합니다.

`plot_confusion(y_true, y_pred, labels=None, *, ax, profile=None, recipe_path=None, overrides=None, cax=None)`

레시피: `plot-classification-confusion.json`. Profile: `ClassificationConfusionProfile`, DTO: `ClassificationConfusionOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `normalization` | `str` | `'true'` | count/true/pred/all 분모. 기본 'true'. |
| `annotate` | `bool` | `True` | 셀 숫자 표시. 기본 True. |

표시/반환 data: `0`, `1`.

진단: `counts`, `n`, `missing_removed`, `accuracy`, `normalization`.

보류/외부 예측으로 평가하세요. 정규화 분모와 클래스 표본 수를 함께 확인하세요.

### classification.roc

분류 순위 성능과 위양성/민감도의 균형을 비교합니다.

`plot_roc(y_true, scores, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-classification-roc.json`. Profile: `ClassificationRocProfile`, DTO: `ClassificationRocOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `positive_label` | `Any` | `1` | 큰 score가 의미하는 양성 라벨. 기본 1. |

표시/반환 data: `fpr`, `tpr`, `threshold`.

진단: `roc_auc`, `n`, `positive_fraction`, `missing_removed`.

이진 score 순위 평가입니다. 보류/외부 점수를 사용하고 비용에 맞는 임계값은 별도로 선택하세요.

### classification.precision_recall

불균형 분류에서 재현율과 양성 예측 신뢰도를 비교합니다.

`plot_precision_recall(y_true, scores, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-classification-precision-recall.json`. Profile: `ClassificationPrecisionRecallProfile`, DTO: `ClassificationPrecisionRecallOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `positive_label` | `Any` | `1` | 큰 score가 의미하는 양성 라벨. 기본 1. |

표시/반환 data: `precision`, `recall`, `threshold`.

진단: `average_precision`, `n`, `positive_fraction`, `missing_removed`.

AP는 사다리꼴 PR 면적과 다릅니다. 불균형 표본의 양성 비율을 함께 확인하세요.

### classification.threshold

비용과 precision/recall trade-off를 근거로 검증 자료의 임계값 후보를 선택합니다.

`plot_threshold(y_true, scores, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-classification-threshold.json`. Profile: `ClassificationThresholdProfile`, DTO: `ClassificationThresholdOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `positive_label` | `Any` | `1` | 큰 score가 의미하는 양성 라벨. 기본 1. |
| `false_positive_cost` | `float` | `1.0` | 비음수 위양성 한 건 비용. 기본 1.0. |
| `false_negative_cost` | `float` | `1.0` | 비음수 위음성 한 건 비용. 기본 1.0. |
| `max_thresholds` | `int` | `200` | 전체 점수 후보의 표시 격자 한도, 2 이상. 기본 200. |

표시/반환 data: `threshold`, `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, `f1`, `cost_per_sample`.

진단: `n`, `missing_removed`, `minimum_cost_threshold`, `minimum_cost_per_sample`, `tie_rule`, `n_candidates`, `displayed_candidates`.

이 자료에서 임계값을 선택하면 같은 자료의 성능은 선택 편향을 포함합니다. 선택용 검증 자료와 최종 테스트를 분리하세요. 비용은 평균 비용이며 통계 신뢰구간이 아닙니다.

### classification.calibration

확률의 과신·과소신과 실제 발생률의 일치를 점검합니다.

`plot_calibration(y_true, probabilities, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-classification-calibration.json`. Profile: `ClassificationCalibrationProfile`, DTO: `ClassificationCalibrationOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `positive_label` | `Any` | `1` | probabilities가 의미하는 양성 라벨. 기본 1. |
| `n_bins` | `int` | `10` | 양의 동일 폭 확률 구간 수. 기본 10. |

표시/반환 data: `bin`, `left`, `right`, `n`, `mean_probability`, `positive_fraction`.

진단: `brier_score`, `ece`, `n`, `missing_removed`.

빈 구간은 표시하지 않으며 n을 반환합니다. ECE는 구간 정의에 의존하고 보류 확률을 사용해야 합니다.

### association.rules

빈도·조건부 확률·독립 대비 lift를 함께 비교해 규칙 검토 우선순위를 정합니다.

`plot_rules(rules, *, ax, profile=None, recipe_path=None, overrides=None, cax=None)`

레시피: `plot-association-rules.json`. Profile: `AssociationRulesProfile`, DTO: `AssociationRulesOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `min_support` | `float` | `0.0` | 표시할 최소 support [0,1]. 기본 0.0. |
| `min_confidence` | `float` | `0.0` | 최소 confidence [0,1]. 기본 0.0. |
| `min_lift` | `float` | `0.0` | 비음수 최소 lift. 기본 0.0. |
| `max_rules` | `int` | `200` | lift 내림차순 표시 한도. 기본 200. |

표시/반환 data: `support`, `confidence`, `lift`.

진단: `input_rules`, `eligible_rules`, `displayed_rules`, `color`, `size`.

높은 lift의 낮은 support 규칙은 희소성을 확인하세요. 규칙은 인과·수익성을 보장하지 않습니다.

### networks.graph

연결 구조·분리 요소·허브 후보를 검토합니다.

`plot_graph(graph, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-networks-graph.json`. Profile: `NetworksGraphProfile`, DTO: `NetworksGraphOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `max_nodes` | `int` | `100` | 표시 가능한 최대 노드; 자동 부분 그래프 생성 안 함. 기본 100. |
| `random_state` | `int` | `1234` | spring 배치 재현성 seed. 기본 1234. |
| `show_labels` | `bool` | `True` | 노드 label 표시. 기본 True. |

표시/반환 data: `node`, `x`, `y`, `degree`.

진단: `n_nodes`, `n_edges`, `directed`, `isolates`.

배치 거리와 node 크기는 탐색 표현입니다. degree 크기는 영향력·인과를 뜻하지 않습니다.

### time_series.timeline

수준 변화·추세·수집 공백·관측 급변을 탐색합니다.

`plot_timeline(times, values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-time-series-timeline.json`. Profile: `TimeSeriesTimelineProfile`, DTO: `TimeSeriesTimelineOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `rolling_window` | `int` | `7` | 이전/현재 행만 쓰는 후행 이동평균; 0이면 미표시. 기본 7. |

표시/반환 data: `time`, `value`, `trailing_mean`.

진단: `n`, `missing_removed`, `rolling_window_in_rows`.

이동평균은 후행 관측 요약이며 예측값이 아닙니다. 행 간격이 일정한지 별도로 확인하세요.

### time_series.lag

지연 관계·비선형 반복 패턴을 확인합니다.

`plot_lag(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-time-series-lag.json`. Profile: `TimeSeriesLagProfile`, DTO: `TimeSeriesLagOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `lag` | `int` | `1` | 양의 행 지연; 날짜 간격은 호출자가 검증. 기본 1. |

표시/반환 data: `previous`, `current`.

진단: `lag_rows`, `n_pairs`.

입력 행은 시간 순이며 일정 간격이어야 합니다. 결측을 제거해 시간축을 압축하지 않습니다.

### time_series.autocorrelation

시간 의존성과 계절 지연 후보를 탐색합니다.

`plot_autocorrelation(values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-time-series-autocorrelation.json`. Profile: `TimeSeriesAutocorrelationProfile`, DTO: `TimeSeriesAutocorrelationOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `nlags` | `int` | `20` | 양의 최대 지연, 표본 수 미만. 기본 20. |
| `kind` | `str` | `'acf'` | acf/pacf. 기본 'acf'. |
| `confidence` | `float` | `0.95` | 근사 신뢰수준 (0,1). 기본 0.95. |

표시/반환 data: `lag`, `coefficient`, `lower`, `upper`.

진단: `n`, `confidence`, `band_center`.

정상성·동일 시간 간격을 확인하세요. 근사 개별 구간으로 다중 지연 검정/모델 차수를 자동 확정하지 않습니다.

### time_series.seasonal

요일·월·시간별 반복 구조와 수집 불균형을 비교합니다.

`plot_seasonal(times, values, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-time-series-seasonal.json`. Profile: `TimeSeriesSeasonalProfile`, DTO: `TimeSeriesSeasonalOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `period` | `str` | `'weekday'` | weekday/month/hour의 원 관측 분포. 기본 'weekday'. |

표시/반환 data: `count`, `mean`, `median`, `std`.

진단: `n`, `missing_removed`, `period`.

집계 분포는 추세·표본 구성·노출량 차이를 보정하지 않습니다. 미관측 기간은 생성하지 않습니다.

### selection.feature_importance

동일 평가 기준의 Feature 중요도와 반복 변동을 비교해 검토 우선순위를 정합니다.

`plot_feature_importance(feature_names, importances, deviations=None, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-selection-feature-importance.json`. Profile: `SelectionFeatureImportanceProfile`, DTO: `SelectionFeatureImportanceOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `max_features` | `int` | `20` | 표시할 Feature 수; 큰 값 순. 기본 20. |
| `error_multiplier` | `float` | `1.0` | 표시 오차=입력 deviations×배율; 통계 CI를 자동 의미하지 않음. 기본 1.0. |

표시/반환 data: `feature`, `importance`, `deviation`.

진단: `n_features`, `displayed_features`, `error_multiplier`.

입력 중요도의 정의·평가 split·오차 의미를 확인하세요. 상관 Feature의 중요도는 분산되며 크기는 인과 기여가 아닙니다.

### selection.validation_curve

학습량/분류 설정 변화의 train–validation 차이로 과적합과 개선 여지를 검토합니다.

`plot_validation_curve(x_values, train_scores, validation_scores, train_deviations=None, validation_deviations=None, *, ax, profile=None, recipe_path=None, overrides=None)`

레시피: `plot-selection-validation-curve.json`. Profile: `SelectionValidationCurveProfile`, DTO: `SelectionValidationCurveOutputDTO`.

| 필드 | 타입 | 기본값 | 의미 |
| --- | --- | --- | --- |
| `title` | `str` | `''` | 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''. |
| `color` | `str` | `'C0'` | Matplotlib 기본 선/표식 색. 기본 'C0'. |
| `alpha` | `float` | `0.8` | artist 투명도 [0,1]. 기본 0.8. |
| `x_label` | `str` | `'Training samples / model setting'` | 표시할 x축 의미. 기본 'Training samples / model setting'. |
| `score_label` | `str` | `'Classification score (higher is better)'` | 동일한 분류 평가 점수 이름. 기본 'Classification score (higher is better)'. |
| `error_multiplier` | `float` | `1.0` | 입력 표준편차 등의 표시 배율; CI를 자동 추정하지 않음. 기본 1.0. |

표시/반환 data: `x`, `train`, `validation`, `train_deviation`, `validation_deviation`.

진단: `train_validation_gap`, `x_spacing`.

같은 split/CV·평가 지표의 사전 계산 값만 비교합니다. 표시 오차는 자동 CI가 아니며 테스트 자료로 설정을 선택하지 마세요.

## 검증·호환 근거

35개 기본 pane·명시 cax·다른 subplot 불변·새 figure/show 없음·DTO JSON·기존 입력 불변,
ECDF/분모/표시 한도·Wilson 극단 비율·SciPy linkage/sklearn silhouette/AUC/AP/threshold 비용,
이웃 self 포함·PACF/후행 평균·Feature 오차·원 numeric x 간격을 검증합니다.
실제 UCI 11개 조합(35개 pane)을 실행하고 PNG 레이아웃·숫자 대비·레이블을 확인했습니다.
Python 3.12 로컬 전체 검사 113개 중 선택 TensorFlow 학습 7개는 별도 환경 검사이며 나머지 통과입니다.
Python 3.7/3.12 Matplotlib/SciPy/sklearn 조합은 기존 CI에서 검증합니다.

축 소유권: [Matplotlib Axes](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html).
계층·실루엣: [SciPy dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html),
[sklearn silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html).
분류: [ROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html),
[PR](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html),
[calibration](https://scikit-learn.org/stable/modules/calibration.html).
이항 구간: [statsmodels proportion_confint](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html).
