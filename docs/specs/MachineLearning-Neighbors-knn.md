# `src/MachineLearning/Neighbors/knn.py`

[소스 코드](../../src/MachineLearning/Neighbors/knn.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
knn.py - 최근접 이웃 회귀·분류
================================================================================
설명:
    - 09장 K-최근접 이웃 모델 및 지도학습 PDF를 기존 Profile → 실행 함수 → Result 구조로 재구성합니다.
    - 공통 분할·스케일링·교차검증·평가는 common.py에 위임합니다.
작성 정보:
    - 작성일: 2026-10-10
    - 버전: 1.0.0
변경 이력:
    2026-10-10  v1.0.0  모델 실행기, 입력 조건 및 결과 명세 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 모듈 전용 결과 DTO 및 저장용 변환 추가

## `KNNProfile`

최근접 이웃 회귀·분류 조건. 공통 필드는 SupervisedProfile 명세를 따릅니다.

Attributes:
    task: regression, classification만 지원합니다.
    scaling: 기본 standard.
    n_neighbors: 참조할 최근접 이웃 수. 모든 CV 학습 폴드의 표본 수 이하여야 함 (기본 5).
    weights: uniform 또는 distance (기본 "uniform").
    metric: 거리 함수. 예: euclidean, manhattan, minkowski (기본 "minkowski").
    p: Minkowski 거리 차수 (기본 2).

## `KNNOutputDTO`

KNN 모듈 전용 반환 DTO.

Attributes:
    공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
    to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.

## `run_knn`

최근접 이웃 회귀·분류의 학습·테스트 평가 및 선택적 튜닝을 수행합니다.

Args:
    X: 결측치 없는 수치형 (n_samples, n_features) 행렬 또는 DataFrame.
    y: 길이 n_samples의 1차원 타깃.
    profile: KNNProfile. None이면 기본 설정을 사용합니다.
Returns:
    KNNOutputDTO: 적합 Pipeline, 학습/테스트 지표, 예측, 선택적 CV 결과.
Raises:
    ValueError: 입력·task·모델 인자가 유효하지 않거나 표본 수가 부족한 경우.
