"""
================================================================================
regression_workflow.py - 회귀 전처리 후보의 CV 선택·원 단위 보류 비교·조건부 추론
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
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
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
from src.stats.regression._common import validate_profile
from ._regression_estimator import RegressionEstimator
from ._workflow import (
    prepare_data,
    split_data,
    candidate_settings,
    fit_candidates,
    regression_metrics,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class RegressionWorkflowProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        test_size (float): 최종 보류 비율 (0,1); 선택에 사용하지 않음. 기본 0.3.
        random_state (int): IID 분할·CV seed. 기본 1234.
        cv (int): 학습 구간 선택 CV 폴드 수, 2 이상. 기본 3.
        split_strategy (str): random/chronological; 후자는 입력 순서 보존. 기본 'random'.
        scoring (str): 원 타깃 단위 CV 지표; neg_root_mean_squared_error/r2/neg_mean_absolute_error. 기본 'neg_root_mean_squared_error'.
        target_missing (str): raise/drop; drop 시 미관측 타깃 행 위치 기록. 기본 'raise'.
        alpha (float): 선택 모형의 조건부 추론 alpha. 기본 0.05.
        cov_type (str): 선택 모형의 nonrobust/HC0/HC1/HC2/HC3/HAC. 기본 'nonrobust'.
        hac_maxlags (int): HAC 최대 지연; chronological에서만 허용. 기본 1.
        candidates (Optional[List[Dict[str,Any]]]): name과 RegressionEstimator 옵션의 명시 후보 목록; None은 기본 5개. 기본 None.
    """

    test_size: float = 0.3
    random_state: int = 1234
    cv: int = 3
    split_strategy: str = "random"
    scoring: str = "neg_root_mean_squared_error"
    target_missing: str = "raise"
    alpha: float = 0.05
    cov_type: str = "nonrobust"
    hac_maxlags: int = 1
    candidates: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class RegressionWorkflowOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        model (Any): CV로 선택해 전체 train에서 재적합한 RegressionEstimator; JSON 제외.
        baseline_model (Any): 처리 전 기준 모형; runtime JSON 제외.
        comparison (pd.DataFrame): 후보별 CV 평균/표준편차·학습 지표·fit 실패 오류; test 지표는 선택 후 두 모형만 평가.
        selected_name (str): CV 점수가 가장 큰 후보 이름.
        train_indices (np.ndarray): 원 입력의 train 위치.
        test_indices (np.ndarray): 원 입력의 최종 test 위치.
        excluded_target_positions (np.ndarray): 명시 target_missing=drop으로 제외한 원 위치.
        baseline_metrics (Dict[str,Any]): 기준 모형 원 y 단위 train/test RMSE·MAE·R² 및 fit R²/수정 R².
        selected_metrics (Dict[str,Any]): 선택 모형의 같은 정의 지표.
        predictions (np.ndarray): 원 y 단위 최종 test 예측.
        y_test (np.ndarray): 원 단위 test 참값.
        inference (Any): 최종 선택 설계의 OLS DTO; 그 모형 좌표계 추론.
        parameters (Dict[str,Any]): 적용 Profile.
        warnings (List[str]): 선택 후 추론·R² 비교·타깃 변환·삭제 해석.
    """

    model: Any
    baseline_model: Any
    comparison: pd.DataFrame
    selected_name: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    excluded_target_positions: np.ndarray
    baseline_metrics: Dict[str, Any]
    selected_metrics: Dict[str, Any]
    predictions: np.ndarray
    y_test: np.ndarray
    inference: Any
    parameters: Dict[str, Any]
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
    validate_profile(profile, RegressionWorkflowProfile)
    if (
        not 0 < profile.test_size < 1
        or profile.cv < 2
        or profile.split_strategy not in ("random", "chronological")
        or profile.target_missing not in ("raise", "drop")
        or profile.scoring
        not in ("neg_root_mean_squared_error", "r2", "neg_mean_absolute_error")
    ):
        raise ValueError("분할·선택·타깃 정책 오류")
    if profile.cov_type == "HAC" and profile.split_strategy != "chronological":
        raise ValueError("HAC는 chronological 자료에서만 사용하세요.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
RegressionWorkflowOutputDTO.runtime_fields = OutputDTO.runtime_fields | frozenset(
    ("baseline_model",)
)


_metrics = regression_metrics


def build_regression_estimator(preprocessor=None, **options):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        preprocessor: 미적합 sklearn 변환기 또는 None. 각 학습/CV 안에서 clone·fit합니다.
        **options: RegressionEstimator의 전처리·목표 변환·VIF·영향도·추론 설정.
    Returns:
        RegressionEstimator: 미적합 sklearn/CV 호환 회귀기.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    return RegressionEstimator(preprocessor=preprocessor, **options)


def _run(X, y, preprocessor, profile):
    """검증·학습 선택·봉인된 test 평가를 연결하고 전용 DTO를 구성합니다.

    Args:
        X: 원 Feature DataFrame 또는 2차원 배열.
        y: X의 행과 대응하는 원 단위 수치 반응.
        preprocessor: 미적합 변환기 또는 None.
        profile: 검증된 RegressionWorkflowProfile.
    Returns:
        RegressionWorkflowOutputDTO: 선택·기준 모형과 같은 보류 행의 평가 결과.
    Raises:
        ValueError/TypeError: 스키마·결측 정책·분할·후보 오류 또는 모든 후보 실패.
    """
    raw, target, missing, valid = prepare_data(X, y, profile)
    train, test, cv = split_data(valid, profile)
    candidates = candidate_settings(preprocessor, profile)
    comparison, selected_name, selected, models = fit_candidates(
        raw.iloc[train],
        target[train],
        cv,
        candidates,
        preprocessor,
        profile,
        build_regression_estimator,
    )
    baseline = models.get("baseline")
    if baseline is None:
        try:
            baseline = build_regression_estimator(
                preprocessor=preprocessor,
                alpha=profile.alpha,
                cov_type=profile.cov_type,
                hac_maxlags=profile.hac_maxlags,
            ).fit(raw.iloc[train], target[train])
        except (ValueError, TypeError, np.linalg.LinAlgError):
            baseline = None
    baseline_metrics = (
        {}
        if baseline is None
        else dict(
            train=_metrics(target[train], baseline.predict(raw.iloc[train])),
            test=_metrics(target[test], baseline.predict(raw.iloc[test])),
            fit=baseline.inference_.fit_metrics,
        )
    )
    predictions = selected.predict(raw.iloc[test])
    selected_metrics = dict(
        train=_metrics(target[train], selected.predict(raw.iloc[train])),
        test=_metrics(target[test], predictions),
        fit=selected.inference_.fit_metrics,
    )
    notes = [
        "모형은 train 내부 CV의 원 y 단위 점수로 선택했습니다. 학습 R² 상승은 보류 설명력 상승을 보장하지 않습니다.",
        "변환 y의 fit R²/수정 R²와 원 y의 평가 R²는 정의가 다릅니다. 선택 후 계수 p/구간은 선택 편향을 보정하지 않은 조건부 결과입니다.",
        "상수는 전처리에서 제거하고 VIF·명시 Cook drop은 학습/CV 내부에서만 수행합니다. test 행은 삭제하지 않습니다.",
    ]
    if baseline is None:
        notes.append(
            "기준 설계가 퇴화하여 baseline 지표를 제공하지 못했습니다. 후보 오류와 VIF를 확인하세요."
        )
    if selected.target_method != "none":
        notes.append(
            "계수·잔차 진단은 변환 y 좌표계입니다. 원 단위 보류 지표와 구별하세요."
        )
    return RegressionWorkflowOutputDTO(
        selected,
        baseline,
        comparison,
        selected_name,
        train,
        test,
        np.flatnonzero(missing),
        baseline_metrics,
        selected_metrics,
        predictions,
        target[test],
        selected.inference_,
        asdict(profile),
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_regression_workflow(
    X, y, preprocessor=None, profile=None, *, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 원 Feature DataFrame 또는 2차원 배열. y와 행 순서가 같아야 합니다.
        y: 원 단위 수치 반응. Series이면 X와 인덱스도 같아야 합니다.
        preprocessor: 미적합 sklearn 변환기 또는 None. 각 학습/CV 안에서 clone·fit합니다.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        RegressionWorkflowOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-workflow", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(X, y, preprocessor, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
