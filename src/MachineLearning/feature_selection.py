"""
================================================================================
feature_selection.py - 필터/RFE와 퍼뮤테이션 특성 중요도
================================================================================
설명: 04장 chi2 필터, 07장 퍼뮤테이션 중요도, 08장 RFE를 모듈화합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 학습 데이터 전용 선택기 및 중요도 분석 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_regression
from sklearn.inspection import permutation_importance
from src.MachineLearning.common import validate_features


@dataclass
class FeatureSelectionProfile:
    """특성 선택 조건.

    Attributes:
        method: chi2 (분류, 비음수), f_regression (회귀) 또는 rfe.
        n_features: 선택할 특성 수 (1 이상 입력 열 수 이하).
        step: RFE에서 각 단계에 제거할 특성 수 또는 비율.
    """
    method: str = "chi2"
    n_features: int = 2
    step: Any = 1


@dataclass
class FeatureSelectionOutputDTO(OutputDTO):
    """특성 선택 결과.

    Attributes:
        selector: 적합 선택기. 테스트 입력에는 selector.transform만 호출합니다.
        transformed: 선택한 학습 특성.
        support: 입력 열 순서의 선택 여부.
        scores: 필터 점수. RFE에서는 None.
    """
    selector: Any
    transformed: np.ndarray
    support: np.ndarray
    scores: Any


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
FeatureSelectionResult = FeatureSelectionOutputDTO


@dataclass
class PermutationImportanceOutputDTO(OutputDTO):
    """퍼뮤테이션 특성 중요도 DTO.

    Attributes:
        importances_mean: 입력 열 순서의 평균 점수 감소.
        importances_std: 입력 열 순서의 점수 감소 표준편차.
        importances: (n_features,n_repeats) 반복별 점수 감소; 음수도 가능.
    """
    importances_mean: np.ndarray
    importances_std: np.ndarray
    importances: np.ndarray

def run_feature_selection(X_train: Any, y_train: Any, profile: FeatureSelectionProfile,
                          estimator: Any = None) -> FeatureSelectionOutputDTO:
    """학습 데이터에서 특성을 선택합니다.

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
    """
    arr = validate_features(X_train)
    if not isinstance(profile.n_features, int) or not 1 <= profile.n_features <= arr.shape[1]:
        raise ValueError("n_features는 1 이상 입력 열 수 이하의 정수여야 합니다.")
    if profile.method == "chi2":
        if (arr < 0).any():
            raise ValueError("chi2 필터는 비음수 특성만 지원합니다.")
        selector = SelectKBest(chi2, k=profile.n_features)
    elif profile.method == "f_regression":
        selector = SelectKBest(f_regression, k=profile.n_features)
    elif profile.method == "rfe" and estimator is not None:
        selector = RFE(clone(estimator), n_features_to_select=profile.n_features, step=profile.step)
    else:
        raise ValueError("method를 확인하고 rfe에는 estimator를 지정하세요.")
    transformed = selector.fit_transform(arr, y_train)
    return FeatureSelectionOutputDTO(selector, transformed, selector.get_support(), getattr(selector, "scores_", None))


def run_permutation_importance(model: Any, X_valid: Any, y_valid: Any,
                               scoring: str = None, n_repeats: int = 10,
                               random_state: int = 1234) -> PermutationImportanceOutputDTO:
    """검증 데이터에서 적합된 모델의 특성 중요도를 계산합니다.

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
    """
    return PermutationImportanceOutputDTO.from_result(permutation_importance(model, X_valid, y_valid, scoring=scoring,
                                  n_repeats=n_repeats, random_state=random_state, n_jobs=1))
