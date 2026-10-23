"""
================================================================================
f_test.py - 독립 두 표본 분산비 F 검정
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 독립 정규 모집단; 이상치와 비정규성에 민감. ANOVA F와 구분.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    입력 검증·전용 DTO·레시피·예제 추가
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
class FTestProfile:
    """FTestProfile 필드 계약.

    Attributes:
        alpha (float): 유의수준, 기본 0.05.
        alternative (str): two-sided/less/greater: var(x)/var(y) 방향. 기본 two-sided.
        ratio_0 (float): 귀무 분산비, 양수. 기본 1.
    """

    alpha: float = 0.05
    alternative: str = "two-sided"
    ratio_0: float = 1.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class FTestOutputDTO(OutputDTO):
    """FTestOutputDTO 필드 계약.

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


def _validate_inputs(x: Any, y: Any, profile: Optional[FTestProfile] = None) -> Any:
    """입력 검증.

    Args:
        x: 첫 독립 수치 표본.
        y: 두 번째 독립 수치 표본.
        profile: FTestProfile.
    Returns:
        Any: 검증된 자료.
    Raises:
        ValueError/TypeError: 입력 조건 오류.
    """
    if not isinstance(profile, FTestProfile):
        raise TypeError("FTestProfile이 필요합니다")
    alpha(profile.alpha)
    a, b = vector(x), vector(y)
    finite_number(profile.ratio_0, "ratio_0", True)
    if profile.alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative 오류")
    if np.var(a, ddof=1) <= 0 or np.var(b, ddof=1) <= 0:
        raise ValueError("양의 표본 분산이 필요합니다.")
    return a, b


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_test(data: Any, profile: Optional[FTestProfile]) -> tuple:
    """검정 계산.

    Args:
        data: 검증 자료.
        profile: 검정 조건.
    Returns:
        tuple: 통계량, p 값, H0, 진단, 경고.
    Raises:
        ValueError: 미정의 통계량.
    """
    a, b = data
    ratio = np.var(a, ddof=1) / np.var(b, ddof=1)
    s = ratio / profile.ratio_0
    d1, d2 = len(a) - 1, len(b) - 1
    p = (
        stats.f.cdf(s, d1, d2)
        if profile.alternative == "less"
        else (
            stats.f.sf(s, d1, d2)
            if profile.alternative == "greater"
            else min(1.0, 2 * min(stats.f.cdf(s, d1, d2), stats.f.sf(s, d1, d2)))
        )
    )
    return (
        float(s),
        float(p),
        "모집단 분산비가 ratio_0와 같다",
        {
            "df_num": d1,
            "df_den": d2,
            "variance_ratio": float(ratio),
            "ratio_0": profile.ratio_0,
            "alternative": profile.alternative,
        },
        ["정규성·독립성을 가정하는 분산비 검정입니다."],
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_f_test(
    x: Any,
    y: Any,
    profile: Optional[FTestProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> FTestOutputDTO:
    """독립 두 표본 분산비 F 검정.

    Args:
        x: 첫 독립 정규 모집단의 표본.
        y: 두 번째 독립 정규 모집단의 표본. 분산비는 var(x)/var(y).
        profile: FTestProfile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피에 적용할 설정.
    Returns:
        FTestOutputDTO: 검정 결과, 진단과 경고.
    Raises:
        ValueError/TypeError: 입력·설정 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "f-test", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs(x, y, profile=profile)
    statistic, p, h0, details, notes = _run_test(data, profile)
    return FTestOutputDTO(
        statistic, p, bool(p < profile.alpha), profile.alpha, h0, details, notes
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_f_test([1, 2, 4, 7, 9], [2, 3, 4, 5, 8]).to_json(indent=2))
