"""
================================================================================
nested_f.py - 중첩 회귀모형의 추가 설명력 F 검정
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
class NestedFProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
    """

    alpha: float = 0.05


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class NestedFOutputDTO(OutputDTO):
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
    validate_profile(profile, NestedFProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(restricted, full, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        restricted: 전용 명세의 restricted 입력.
        full: 전용 명세의 full 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        NestedFOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    small = ols_result(restricted)
    large = ols_result(full)
    if small.cov_type != "nonrobust" or large.cov_type != "nonrobust":
        raise ValueError("고전적 중첩 F는 nonrobust OLS 결과를 사용하세요.")
    if small.model.endog.shape != large.model.endog.shape or not np.array_equal(
        small.model.endog, large.model.endog
    ):
        raise ValueError("같은 행·반응변수 좌표계가 필요합니다.")
    if (
        small.model.data.row_labels is not None
        and large.model.data.row_labels is not None
        and not small.model.data.row_labels.equals(large.model.data.row_labels)
    ):
        raise ValueError("적합 행 순서가 다릅니다.")
    A = large.model.exog
    B = small.model.exog
    projection = A @ np.linalg.lstsq(A, B, rcond=None)[0]
    if large.df_resid >= small.df_resid or not np.allclose(
        B, projection, rtol=1e-7, atol=1e-9
    ):
        raise ValueError("restricted 설계가 full 설계에 중첩되어야 합니다.")
    statistic, p, df = large.compare_f_test(small)
    return NestedFOutputDTO(
        pd.DataFrame(
            [
                test_row(
                    "nested-model F",
                    statistic,
                    p,
                    profile.alpha,
                    float(df),
                    float(large.df_resid),
                )
            ]
        ),
        {
            "restricted_sse": float(small.ssr),
            "full_sse": float(large.ssr),
            "hypothesis": "추가 계수 모두 0",
        },
        asdict(profile),
        [
            "같은 반응변수·관측·등분산 오차의 중첩 모형만 비교합니다. 다른 변환 y와 AIC/R²를 직접 비교하지 않습니다."
        ],
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_nested_f(restricted, full, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        restricted: 전용 명세의 restricted 입력.
        full: 전용 명세의 full 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        NestedFOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-nested-f", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(restricted, full, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
