"""
================================================================================
poisson.py - poisson 전용 GLM 계수·우도·조건별 잔차·영향도
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 반응 분포와 평균 링크를 구별하고 OLS 잔차 정규성·보통 R²를 강요하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-23
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-23  김재현(Finn)     v1.0.0    응용 분석 모듈·모의10회 실행 연결
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
from ._common import fit_family


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class PoissonGLMProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 계수·구간 유의수준 (0,1) 기본 0.05.
        maxiter (int): IRLS 최대 적합 반복, 양수 기본 100.
        cov_type (str): nonrobust/HC0 robust 공분산 기본 'nonrobust'.
    """

    alpha: float = 0.05
    maxiter: int = 100
    cov_type: str = "nonrobust"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class PoissonGLMOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        model (Any): 적합 statsmodels GLM; JSON 제외.
        coefficients (pd.DataFrame): term·coefficient·se·z·양측p·CI·exp(coef).
        fit_metrics (Dict[str,Any]): family·link·llf·AIC·BIC(평균계수 수 k)·deviance·Pearson dispersion·McFadden(null 비교) 또는 None.
        diagnostics (pd.DataFrame): row_position·fitted_mean·Pearson/deviance잔차·leverage·GLM Cook 근사.
        predictions (Optional[np.ndarray]): 선택 X_new의 조건부 평균; logistic은 성공확률.
        parameters (Dict[str,Any]): 적용 Profile.
        warnings (List[str]): 분포/링크/노출/추론·선택 한계.
    """

    model: Any
    coefficients: pd.DataFrame
    fit_metrics: Dict[str, Any]
    diagnostics: pd.DataFrame
    predictions: Optional[np.ndarray]
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
    validate_profile(profile, PoissonGLMProfile)
    if profile.maxiter < 1 or profile.cov_type not in ("nonrobust", "HC0"):
        raise ValueError("반복/공분산 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(X, y, X_new, offset, new_offset, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        X_new: 전용 명세의 X_new 입력.
        offset: 전용 명세의 offset 입력.
        new_offset: 전용 명세의 new_offset 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        PoissonGLMOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    return fit_family(
        X, y, X_new, offset, new_offset, profile, "poisson", PoissonGLMOutputDTO
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_poisson(
    X,
    y,
    X_new=None,
    offset=None,
    new_offset=None,
    profile=None,
    *,
    recipe_path=None,
    overrides=None,
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        X_new: 전용 명세의 X_new 입력.
        offset: 전용 명세의 offset 입력.
        new_offset: 전용 명세의 new_offset 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        PoissonGLMOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "glm-poisson", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(X, y, X_new, offset, new_offset, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
