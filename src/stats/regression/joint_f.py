"""
================================================================================
joint_f.py - 계수 묶음과 사용자 선형 가설의 F 검정
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
class JointFProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
    """

    alpha: float = 0.05


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class JointFOutputDTO(OutputDTO):
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
    validate_profile(profile, JointFProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(model, contrasts, rhs, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        contrasts: 전용 명세의 contrasts 입력.
        rhs: 전용 명세의 rhs 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        JointFOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    result = ols_result(model)
    p = result.model.exog.shape[1]
    if contrasts is None:
        indices = [
            i
            for i, name in enumerate(result.model.exog_names)
            if name != "Intercept"
            and not (result.model.k_constant and np.ptp(result.model.exog[:, i]) == 0)
        ]
        if not indices:
            raise ValueError("검정할 기울기가 없습니다.")
        R = np.eye(p)[indices]
    else:
        R = np.asarray(contrasts, dtype=float)
    if (
        R.ndim != 2
        or R.shape[1] != p
        or not np.isfinite(R).all()
        or np.linalg.matrix_rank(R) != len(R)
    ):
        raise ValueError("full-row-rank contrast 행렬이 필요합니다.")
    q = np.zeros(len(R)) if rhs is None else np.asarray(rhs, dtype=float)
    if q.shape != (len(R),) or not np.isfinite(q).all():
        raise ValueError("rhs 길이/유한성 오류")
    test = result.f_test((R, q))
    table = pd.DataFrame(
        [
            test_row(
                "joint F",
                test.fvalue,
                test.pvalue,
                profile.alpha,
                float(test.df_num),
                float(test.df_denom),
            )
        ]
    )
    return JointFOutputDTO(
        table,
        {
            "contrasts": R,
            "rhs": q,
            "cov_type": result.cov_type,
            "hypothesis": "R beta = rhs",
        },
        asdict(profile),
        [
            "단일 계수의 같은 공분산 양측 검정에서는 F=t²입니다. 전체 F는 각 계수의 개별 유효성을 보장하지 않습니다."
        ],
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_joint_f(
    model, contrasts=None, rhs=None, profile=None, *, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        contrasts: 전용 명세의 contrasts 입력.
        rhs: 전용 명세의 rhs 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        JointFOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-joint-f", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(model, contrasts, rhs, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
