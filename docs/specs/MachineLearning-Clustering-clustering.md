# `src/MachineLearning/Clustering/clustering.py`

[소스 코드](../../src/MachineLearning/Clustering/clustering.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
clustering.py - KMeans, 계층적 군집화, DBSCAN, GaussianMixture
================================================================================
설명:
    - 12/13장 노트북 및 군집화 PDF의 모델과 내부/외부 평가 지표를 모듈화합니다.
    - 군집화는 탐색적 전체 데이터 적합입니다. 테스트 성능으로 해석하지 않습니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 군집화 실행기와 거리·덴드로그램 명세 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `ClusteringProfile`

군집화 조건.

Attributes:
    method: kmeans, hierarchical, dbscan, gaussian_mixture.
    n_clusters: 군집 수. 계층적 거리 임계값을 사용할 때만 None.
    scaling: none, standard, minmax. 입력 전체에서 적합합니다.
    random_state: 재현용 시드.
    n_init: KMeans/GaussianMixture 초기화 반복 수.
    max_iter: KMeans/GaussianMixture 최대 반복 수.
    linkage: 계층적 군집 연결 방법 (ward, complete, average, single).
    metric: 계층적/DBSCAN 거리. precomputed는 지원하지 않습니다.
    distance_threshold: 계층적 거리 임계값. 지정하면 n_clusters=None이어야 합니다.
    eps: DBSCAN 이웃 반경 (양수).
    min_samples: DBSCAN 밀집 영역 최소 표본 수.
    covariance_type: GaussianMixture 공분산 형태 (full, tied, diag, spherical).

## `ClusteringOutputDTO`

군집화 결과.

Attributes:
    model: 적합 Pipeline. KMeans/혼합모형만 새 데이터 predict를 지원합니다.
    labels: 입력 행 순서의 군집 라벨. DBSCAN 잡음은 -1.
    transformed: 스케일링된 입력 특성.
    metrics: silhouette 및 선택적 adjusted_rand, KMeans inertia 또는 혼합모형 AIC/BIC.
        silhouette은 유효한 군집 수가 2 이상, 표본 수 미만일 때만 반환합니다.
        DBSCAN은 잡음을 제외한 데이터에서 silhouette을 계산합니다.
    n_clusters: 잡음을 제외한 실제 군집 수.
    n_noise: 잡음 표본 수.

## `run_clustering`

군집 모델을 적합하고 탐색적 지표를 계산합니다.

Args:
    X: 결측값 없는 수치형 2차원 특성.
    profile: 군집화 설정. None이면 KMeans 기본 설정.
    labels_true: 선택적 외부 평가용 1차원 정답. 학습에는 사용하지 않습니다.
Returns:
    ClusteringOutputDTO: 모델, 라벨, 변환 데이터 및 평가 지표.
Raises:
    ValueError: 잘못된 모델/거리/군집 수, NaN 또는 라벨 길이 불일치.

## `DistanceMatrixOutputDTO`

거리 행렬 출력 DTO.

Attributes:
    matrix: (n_rows,n_columns) 수치 거리 행렬.
    metric: 거리 이름.
Notes:
    np.asarray(result), result.shape를 지원합니다. 원 배열은 result.matrix입니다.

### `shape`



## `LinkageMatrixOutputDTO`

계층 군집 연결 행렬 출력 DTO.

Attributes:
    matrix: (n_samples-1,4) scipy 연결 행렬.
    method: 연결 방식.
    metric: 거리 이름.
Notes:
    scipy 덴드로그램에는 result.matrix를 권장합니다. np.asarray 호환 접근도 제공합니다.

### `shape`



## `distance_matrix`

표본 간 거리 행렬을 반환합니다.

Args:
    X: 기준 수치 특성 행렬.
    Y: 비교 행렬. None이면 X 자체와 비교.
    metric: scipy cdist 거리 이름.
Returns:
    DistanceMatrixOutputDTO: matrix/metric 필드의 거리 분석 결과.
Raises:
    ValueError: 잘못된 특성·거리 또는 서로 다른 열 수.

## `make_linkage_matrix`

scipy dendrogram에 전달할 연결 행렬을 생성합니다.

Args:
    X: 수치 특성. 스케일링이 필요하면 ClusteringResult.transformed를 전달합니다.
    method: ward, single, complete, average 등의 scipy 연결 방법.
    metric: 거리 함수. ward는 euclidean만 허용합니다.
Returns:
    LinkageMatrixOutputDTO: matrix/method/metric 필드. matrix는 (n_samples-1,4).
Raises:
    ValueError: 잘못된 입력 또는 ward와 비유클리드 거리의 조합.
