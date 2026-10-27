"""
================================================================================
independence.py - 카이제곱 독립성 검정
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 독립 표본의 교차 빈도표; 짝지어진 관측에는 McNemar 사용.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    입력 검증·전용 DTO·레시피·예제 추가
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.stats.tests._validation import alpha, vector, counts
from src.MachineLearning.analysis_common import positive_int, finite_number


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class IndependenceProfile:
    """IndependenceProfile 필드 계약.

    Attributes:
        alpha (float): 유의수준, 기본 0.05.
        correction (bool): 자유도 1일 때 Yates 보정, 기본 True.
    """

    alpha: float = 0.05
    correction: bool = True


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class IndependenceOutputDTO(OutputDTO):
    """IndependenceOutputDTO 필드 계약.

    Attributes:
        statistic (float): 검정 통계량.
        p_value (float): 귀무가설 하의 p 값.
        reject_h0 (bool): p_value < alpha.
        alpha (float): 사용한 유의수준.
        null_hypothesis (str): 검정한 귀무가설.
        details (Dict[str, Any]): 자유도·기대도수·표본수 등 검정별 진단.
        warnings (List[str]): 근사 조건과 해석 제약.
    """

    statistic: float
    p_value: float
    reject_h0: bool
    alpha: float
    null_hypothesis: str
    details: Dict[str, Any]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(table: Any, profile: Optional[IndependenceProfile] = None) -> Any:
    """입력 검증.

    Args:
        table: 입력 자료.
        profile: IndependenceProfile.
    Returns:
        Any: 검증된 자료.
    Raises:
        ValueError/TypeError: 입력 조건 오류.
    """
    if not isinstance(profile, IndependenceProfile):
        raise TypeError("IndependenceProfile이 필요합니다")
    alpha(profile.alpha)
    a = counts(table, table=True)
    if min(a.shape) < 2 or (a.sum(axis=0) == 0).any() or (a.sum(axis=1) == 0).any():
        raise ValueError("각 주변합이 양수인 2×2 이상 표가 필요합니다.")
    if not isinstance(profile.correction, bool):
        raise ValueError("correction은 bool입니다.")
    return a


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_test(data: Any, profile: Optional[IndependenceProfile]) -> tuple:
    """검정 계산.

    Args:
        data: 검증 자료.
        profile: 검정 조건.
    Returns:
        tuple: 통계량, p 값, H0, 진단, 경고.
    Raises:
        ValueError: 미정의 통계량.
    """
    s, p, df, exp = stats.chi2_contingency(data, correction=profile.correction)
    return (
        float(s),
        float(p),
        "행 변수와 열 변수가 독립이다",
        {
            "df": df,
            "expected": exp,
            "observed": data,
            "cramers_v": float(
                np.sqrt(
                    stats.chi2_contingency(data, correction=False)[0]
                    / (data.sum() * min(data.shape[0] - 1, data.shape[1] - 1))
                )
            ),
        },
        (
            ["기대도수 5 미만: 근사 검정 대신 정확 검정을 고려하세요."]
            if (exp < 5).any()
            else []
        ),
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_independence(
    table: Any,
    profile: Optional[IndependenceProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> IndependenceOutputDTO:
    """카이제곱 독립성 검정.

    Args:
        table: 독립 표본의 교차 빈도표; 짝지어진 관측에는 McNemar 사용.
        profile: IndependenceProfile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피에 적용할 설정.
    Returns:
        IndependenceOutputDTO: 검정 결과, 진단과 경고.
    Raises:
        ValueError/TypeError: 입력·설정 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "independence", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs(table, profile=profile)
    statistic, p, h0, details, notes = _run_test(data, profile)
    return IndependenceOutputDTO(
        statistic, p, bool(p < profile.alpha), profile.alpha, h0, details, notes
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_independence([[30, 20], [10, 40]]).to_json(indent=2))
