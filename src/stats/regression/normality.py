"""
================================================================================
normality.py - 회귀 잔차의 정규성 진단
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
from scipy import stats
from src.stats.regression._common import (
    validate_profile,
    numeric_data,
    fit_ols,
    ols_result,
    test_row,
    vif_values,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class RegressionNormalityProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
        method (str): jarque_bera/shapiro/dagostino/anderson 잔차 정규성. 기본 'jarque_bera'.
    """

    alpha: float = 0.05
    method: str = "jarque_bera"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class RegressionNormalityOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
        details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
        parameters (Dict[str,Any]): 적용 Profile.
        warnings (List[str]): 검정 가정·선택 후 추론 정보.
    """

    tests: pd.DataFrame
    details: Dict[str, Any]
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
    validate_profile(profile, RegressionNormalityProfile)
    if profile.method not in ("jarque_bera", "shapiro", "dagostino", "anderson"):
        raise ValueError("정규성 방법 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(model, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        RegressionNormalityOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    from statsmodels.stats.diagnostic import normal_ad

    result = ols_result(model)
    residual = np.asarray(result.resid)
    notes = [
        "반응변수 y가 아닌 잔차를 검사합니다. 적합 잔차의 의존성 때문에 iid 표본 검정의 p를 단독 확정 기준으로 쓰지 않습니다."
    ]
    if np.ptp(residual) == 0:
        raise ValueError("상수 잔차 정규성은 정의되지 않습니다.")
    if profile.method == "jarque_bera":
        from statsmodels.stats.stattools import jarque_bera

        statistic, p, skew, kurtosis = jarque_bera(residual)
    elif profile.method == "shapiro":
        if len(residual) < 3:
            raise ValueError("Shapiro 표본 부족")
        statistic, p = stats.shapiro(residual)
        if len(residual) > 5000:
            notes.append("Shapiro n>5000의 p 근사 정확도를 확인하세요.")
    elif profile.method == "dagostino":
        if len(residual) < 8:
            raise ValueError("D’Agostino는 8개 이상 잔차가 필요합니다.")
        statistic, p = stats.normaltest(residual)
    else:
        statistic, p = normal_ad(residual)
    return RegressionNormalityOutputDTO(
        pd.DataFrame([test_row(profile.method, statistic, p, profile.alpha)]),
        {"n_samples": len(residual), "hypothesis": "정규 오차"},
        asdict(profile),
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_normality(model, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        RegressionNormalityOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-normality", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(model, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
