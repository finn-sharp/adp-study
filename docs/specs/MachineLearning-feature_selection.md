# `src/MachineLearning/feature_selection.py`

[소스 코드](../../src/MachineLearning/feature_selection.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
feature_selection.py - 필터/RFE와 퍼뮤테이션 특성 중요도
================================================================================
설명: 04장 chi2 필터, 07장 퍼뮤테이션 중요도, 08장 RFE를 모듈화합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 학습 데이터 전용 선택기 및 중요도 분석 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `FeatureSelectionProfile`

특성 선택 조건.

Attributes:
    method: chi2 (분류, 비음수), f_regression (회귀) 또는 rfe.
    n_features: 선택할 특성 수 (1 이상 입력 열 수 이하).
    step: RFE에서 각 단계에 제거할 특성 수 또는 비율.

## `FeatureSelectionOutputDTO`

특성 선택 결과.

Attributes:
    selector: 적합 선택기. 테스트 입력에는 selector.transform만 호출합니다.
    transformed: 선택한 학습 특성.
    support: 입력 열 순서의 선택 여부.
    scores: 필터 점수. RFE에서는 None.

## `PermutationImportanceOutputDTO`

퍼뮤테이션 특성 중요도 DTO.

Attributes:
    importances_mean: 입력 열 순서의 평균 점수 감소.
    importances_std: 입력 열 순서의 점수 감소 표준편차.
    importances: (n_features,n_repeats) 반복별 점수 감소; 음수도 가능.

## `run_feature_selection`

학습 데이터에서 특성을 선택합니다.

Args:
    X_train: 분할 후 학습 특성. 테스트 입력을 포함하면 데이터 누수가 발생합니다.
    y_train: 학습 타깃.
    profile: 선택 방법과 특성 수.
    estimator: rfe에 필요한 coef_ 또는 feature_importances_ 지원 미적합 추정기.
Returns:
    FeatureSelectionOutputDTO: 적합 선택기와 선택 결과.
Raises:
    ValueError: 잘못된 방법·선택 수, chi2의 음수 입력 또는 RFE 추정기 누락.

Notes:
    CV 튜닝 시에는 반환된 selector를 재사용하지 말고 미적합 선택기를 Pipeline에 넣습니다.

## `run_permutation_importance`

검증 데이터에서 적합된 모델의 특성 중요도를 계산합니다.

Args:
    model: 적합 추정기 또는 Pipeline.
    X_valid: 모델 입력과 열 순서가 같은 검증 특성.
    y_valid: 검증 정답.
    scoring: sklearn 평가 이름. None이면 추정기 기본 score.
    n_repeats: 열별 순열 반복 수.
    random_state: 순열 시드.
Returns:
    PermutationImportanceOutputDTO: importances_mean, importances_std, importances 필드를 가진 sklearn Bunch.
Raises:
    ValueError: 입력·반복 수가 잘못된 경우. 미적합 모델은 sklearn 오류를 반환합니다.
