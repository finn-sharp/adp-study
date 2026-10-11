# `src/MachineLearning/Decomposition/manifold.py`

[소스 코드](../../src/MachineLearning/Decomposition/manifold.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
manifold.py - MDS, Isomap, LLE, t-SNE 탐색적 임베딩
================================================================================
설명: 14장 차원 축소 비교 예제를 탐색적 분석으로 분리합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 임베딩 실행기 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `EmbeddingProfile`

탐색적 임베딩 조건.

Attributes:
    method: mds, isomap, lle, tsne.
    n_components: 축 수.
    scaling: none, standard, minmax.
    random_state: 시드.
    n_neighbors: Isomap/LLE 이웃 수 (표본 수 미만).
    perplexity: t-SNE perplexity (양수, 표본 수 미만).

## `EmbeddingOutputDTO`

탐색적 임베딩 결과.

Attributes:
    model: 적합 Pipeline. MDS/t-SNE는 새 데이터 transform을 지원하지 않습니다.
    coordinates: 입력 행 순서의 축소 좌표. 예측 성능 지표가 아닙니다.

## `run_embedding`

수치 특성을 탐색적 저차원 좌표로 변환합니다.

Args:
    X: 유한한 수치형 2차원 특성.
    profile: 임베딩 설정.
Returns:
    EmbeddingOutputDTO: 적합 모델과 좌표.
Raises:
    ValueError: 잘못된 방법, perplexity/이웃 수 또는 입력.

Notes:
    같은 데이터로 좌표 학습·분류·평가한 정확도를 일반화 성능으로 반환하지 않습니다.
