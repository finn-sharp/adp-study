"""
================================================================================
linearity.py - RESET·Rainbow 선형 사양 진단
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
class LinearityProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
        method (str): reset/rainbow 선형 사양 진단. 기본 'reset'.
        power (int): RESET 추가 fitted 거듭제곱 최대 차수, 2 이상. 기본 3.
        fraction (float): Rainbow 중심 부분 표본 비율 (0,1). 기본 0.5.
    """

    alpha: float = 0.05
    method: str = "reset"
    power: int = 3
    fraction: float = 0.5


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class LinearityOutputDTO(OutputDTO):
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
    validate_profile(profile, LinearityProfile)
    if (
        profile.method not in ("reset", "rainbow")
        or profile.power < 2
        or not 0 < profile.fraction < 1
    ):
        raise ValueError("선형성 설정 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(model, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        LinearityOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    from statsmodels.stats.diagnostic import linear_reset, linear_rainbow

    result = ols_result(model)
    if profile.method == "reset":
        test = linear_reset(result, power=profile.power, test_type="fitted", use_f=True)
        row = test_row(
            "RESET F",
            test.fvalue,
            test.pvalue,
            profile.alpha,
            float(test.df_num),
            float(test.df_denom),
        )
    else:
        statistic, p = linear_rainbow(result, frac=profile.fraction)
        row = test_row("Rainbow F", statistic, p, profile.alpha)
    return LinearityOutputDTO(
        pd.DataFrame([row]),
        {
            "n_samples": int(result.nobs),
            "hypothesis": "해당 대안 대비 선형 사양이 적절함",
        },
        asdict(profile),
        [
            "비기각은 선형성의 증명이 아닙니다. RESET는 누락변수/함수형 등 다른 사양 오류에도 반응하고 Rainbow는 행 순서·등분산 가정에 민감합니다."
        ],
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_linearity(model, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        LinearityOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-linearity", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(model, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
