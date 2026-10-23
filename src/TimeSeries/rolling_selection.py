"""
================================================================================
rolling_selection.py - 시간순 rolling 검증 차수 선택과 미래 보류 평가
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 학습 적합과 선택은 학습/CV 안에서 수행하고 추론 가정을 확인합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-23
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-23  김재현(Finn)     v1.0.0    응용 분석 모듈·모의10회 실행 연결
================================================================================
"""
__author__='김재현(Finn)'
__email__='penguin.klg@gmail.com'
__version__='1.0.0'
__status__='Development'

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass,asdict
from typing import Any,Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.stats.regression._common import validate_profile
from ._rolling import rolling_fit

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class RollingSelectionProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        horizon (int): 마지막 최종평가 구간 길이, 2 이상 기본 7.
        validation_size (int): 각 rolling 검증 horizon, 2 이상 기본 7.
        folds (int): 과거 확장 학습 폴드 수, 2 이상 기본 3.
        period (int): 1은 ARIMA, 2 이상은 SARIMA (1,0,0,s)와 계절 naive 기본 1.
        candidate_orders (Optional[List[List[int]]]): 학습 rolling 비교 차수 목록; None이면 [0,0,0],[1,0,0],[2,0,0] 기본 None.
        maxiter (int): 후보 최대 적합 반복, 양수 기본 100.
        alpha (float): 선택 예측구간 유의수준 (0,1) 기본 0.05.
    """
    horizon: int = 7
    validation_size: int = 7
    folds: int = 3
    period: int = 1
    candidate_orders: Optional[List[List[int]]] = None
    maxiter: int = 100
    alpha: float = 0.05

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class RollingSelectionOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        model (Any): 선택 ARIMA/SARIMA DTO; runtime JSON 제외.
        comparison (pd.DataFrame): 후보별 rolling RMSE·MAE·수렴 실패; 최종평가는 선택에 사용 안 함.
        folds (List[Dict[str,Any]]): 원행 위치의 학습/검증 연속 구간.
        selected_order (List[int]): 학습 rolling RMSE로 선택한 차수.
        train_indices (np.ndarray): 최종 적합 과거 원행 위치.
        test_indices (np.ndarray): 최종 미사용 미래 원행 위치.
        predictions (np.ndarray): 선택 모형 고정 horizon 평균 예측.
        baseline_predictions (np.ndarray): 마지막 값 또는 마지막 주기 반복 naive.
        test_metrics (Dict[str,Any]): 선택/naive의 원단위 RMSE·MAE·R² 및 MASE 학습분모.
        residual_diagnostics (Dict[str,Any]): 초기화 후 학습 잔차 Ljung-Box, p/q/P/Q 자유도 보정·AIC/BIC.
        parameters (Dict[str,Any]): Profile 전체.
        warnings (List[str]): 시간순·비교 척도·선택 후 구간 한계.
    """
    model: Any
    comparison: pd.DataFrame
    folds: List[Dict[str,Any]]
    selected_order: List[int]
    train_indices: np.ndarray
    test_indices: np.ndarray
    predictions: np.ndarray
    baseline_predictions: np.ndarray
    test_metrics: Dict[str,Any]
    residual_diagnostics: Dict[str,Any]
    parameters: Dict[str,Any]
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
    validate_profile(profile,RollingSelectionProfile)
    if profile.horizon<2 or profile.validation_size<2 or profile.folds<2 or profile.period<1 or profile.maxiter<1:raise ValueError("시계열 분할/기간 설정 오류")

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run(y, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        y: 전용 명세의 y 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        RollingSelectionOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    return rolling_fit(y,profile,RollingSelectionOutputDTO)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_rolling_selection(y, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        y: 전용 명세의 y 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        RollingSelectionOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('timeseries-rolling-selection',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(y,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
