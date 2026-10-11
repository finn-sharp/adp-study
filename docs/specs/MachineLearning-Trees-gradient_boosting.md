# `src/MachineLearning/Trees/gradient_boosting.py`

[소스 코드](../../src/MachineLearning/Trees/gradient_boosting.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
gradient_boosting.py - 그레이디언트 부스팅 회귀·분류
================================================================================
설명:
    - 08장 그레이디언트 부스팅 트리 모델를 기존 Profile → 실행 함수 → Result 구조로 재구성합니다.
    - 공통 분할·스케일링·교차검증·평가는 common.py에 위임합니다.
작성 정보:
    - 작성일: 2026-10-10
    - 버전: 1.0.0
변경 이력:
    2026-10-10  v1.0.0  모델 실행기, 입력 조건 및 결과 명세 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 모듈 전용 결과 DTO 및 저장용 변환 추가

## `GradientBoostingProfile`

그레이디언트 부스팅 회귀·분류 조건. 공통 필드는 SupervisedProfile 명세를 따릅니다.

Attributes:
    task: regression, classification만 지원합니다.
    scaling: 기본 none.
    n_estimators: 부스팅 단계 수 (기본 100).
    learning_rate: 각 단계의 학습률 (기본 0.1).
    max_depth: 기본 트리의 최대 깊이 (기본 3).
    subsample: 각 단계 학습 표본 비율 (기본 1.0).

## `GradientBoostingOutputDTO`

GradientBoosting 모듈 전용 반환 DTO.

Attributes:
    공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
    to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.

## `run_gradient_boosting`

그레이디언트 부스팅 회귀·분류의 학습·테스트 평가 및 선택적 튜닝을 수행합니다.

Args:
    X: 결측치 없는 수치형 (n_samples, n_features) 행렬 또는 DataFrame.
    y: 길이 n_samples의 1차원 타깃.
    profile: GradientBoostingProfile. None이면 기본 설정을 사용합니다.
Returns:
    GradientBoostingOutputDTO: 적합 Pipeline, 학습/테스트 지표, 예측, 선택적 CV 결과.
Raises:
    ValueError: 입력·task·모델 인자가 유효하지 않거나 표본 수가 부족한 경우.
