"""
================================================================================
bernoulli.py - bernoulli 분포 전용 Naive Bayes
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 학습 적합과 선택은 학습/CV 안에서 수행하고 추론 가정을 확인합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from sklearn.base import BaseEstimator, ClassifierMixin
from src.MachineLearning.NaiveBayes._distribution_common import (
    validate_nb_profile,
    run_distribution,
    numeric_counts,
    labels_for_fit,
    log_priors,
    DistributionClassifierMixin,
)
from src.Preprocessing._common import feature_array


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class BernoulliProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        test_size (float): 최종 보류 비율; fixed test_indices가 우선. 기본 0.3.
        random_state (int): 분할·CV seed. 기본 1234.
        stratify (bool): random 분할의 클래스 계층화. 기본 True.
        split_strategy (str): random/chronological; 후자는 입력 행 순서. 기본 'random'.
        scaling (str): Gaussian만 none/standard/minmax; 나머지는 none. 기본 'none'.
        cv (int): 선택용 학습 CV 폴드, 2 이상. 기본 3.
        scoring (Optional[str]): GridSearchCV 분류 지표; None은 accuracy. 기본 None.
        n_jobs (int): CV 병렬 작업 수; 0 불가. 기본 1.
        param_grid (Optional[Dict[str,Any]]): 분포별 estimator 인자명 또는 model__인자명 탐색. 기본 None.
        alpha (float): 양수 Bernoulli 가산 평활화. 기본 1.0.
        binarize (Optional[float]): None이면 엄격 0/1; 유한 수면 고정 임계값으로 이진화. 기본 None.
    """

    test_size: float = 0.3
    random_state: int = 1234
    stratify: bool = True
    split_strategy: str = "random"
    scaling: str = "none"
    cv: int = 3
    scoring: Optional[str] = None
    n_jobs: int = 1
    param_grid: Optional[Dict[str, Any]] = None
    alpha: float = 1.0
    binarize: Optional[float] = None


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class BernoulliOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        model_name (str): 적합 분포별 클래스 이름.
        model (Any): 전처리 포함 Pipeline; JSON 제외.
        train_metrics (Dict[str,float]): 학습 accuracy·weighted precision/recall/F1.
        test_metrics (Dict[str,float]): 보류 같은 지표 및 이진 ROC AUC.
        y_test (np.ndarray): 보류 원 라벨.
        predictions (np.ndarray): 보류 예측 원 라벨.
        probabilities (np.ndarray): classes 열 순서 posterior.
        classes (np.ndarray): 학습 클래스 순서.
        class_counts (np.ndarray): 학습 클래스 빈도; 전체 자료 빈도 아님.
        confusion (np.ndarray): classes 순서의 참값×예측 혼동행렬.
        train_indices (np.ndarray): 원 입력 학습 행 위치.
        test_indices (np.ndarray): 원 입력 보류 행 위치.
        feature_names (List[str]): 원 Feature 순서.
        best_params (Dict[str,Any]): 학습 CV 선택 파라미터.
        best_cv_score (Optional[float]): 최고 CV 점수 또는 None.
        cv_results (Optional[Dict[str,Any]]): 전체 CV 결과 또는 None.
        warnings (List[str]): 분포·독립·확률 해석 가정.
    """

    model_name: str
    model: Any
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    y_test: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray
    class_counts: np.ndarray
    confusion: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: List[str]
    best_params: Dict[str, Any]
    best_cv_score: Optional[float]
    cv_results: Optional[Dict[str, Any]]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate(profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
    Returns:
        object: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    validate_nb_profile(profile, BernoulliProfile)
    if profile.alpha <= 0:
        raise ValueError("alpha는 양수입니다.")
    if profile.binarize is not None and (
        isinstance(profile.binarize, bool)
        or not isinstance(profile.binarize, (int, float))
        or not np.isfinite(profile.binarize)
    ):
        raise ValueError("binarize 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def build_bernoulli_classifier(profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
        recipe_path: 전용 명세의 recipe_path 입력.
        overrides: 전용 명세의 overrides 입력.
    Returns:
        BernoulliNB: 미적합 분포별 분류기.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "naive-bayes-bernoulli", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    from sklearn.naive_bayes import BernoulliNB

    return BernoulliNB(alpha=profile.alpha, binarize=profile.binarize)


def _run(X, y, test_indices, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        test_indices: 전용 명세의 test_indices 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        BernoulliOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    a, _ = feature_array(X, True, False)
    if profile.binarize is None and not np.isin(a, [0, 1]).all():
        raise ValueError("Bernoulli는 0/1 입력 또는 명시 binarize가 필요합니다.")
    return run_distribution(
        X,
        y,
        build_bernoulli_classifier(profile),
        profile,
        BernoulliOutputDTO,
        test_indices=test_indices,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_bernoulli(
    X, y, test_indices=None, profile=None, *, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        test_indices: 전용 명세의 test_indices 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        BernoulliOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "naive-bayes-bernoulli", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(X, y, test_indices, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
