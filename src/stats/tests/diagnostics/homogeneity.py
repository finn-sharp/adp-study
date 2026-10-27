"""
================================================================================
homogeneity.py - 등분산 검정
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 독립 집단; Bartlett는 정규성에 민감, median Levene는 Brown–Forsythe.
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
class HomogeneityProfile:
    """HomogeneityProfile 필드 계약.

    Attributes:
        alpha (float): 유의수준, 기본 0.05.
        method (str): levene/bartlett/fligner. 기본 levene.
        center (str): Levene/Fligner 중심 mean/median/trimmed. 기본 median.
    """

    alpha: float = 0.05
    method: str = "levene"
    center: str = "median"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class HomogeneityOutputDTO(OutputDTO):
    """HomogeneityOutputDTO 필드 계약.

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


def _validate_inputs(*groups: Any, profile: Optional[HomogeneityProfile] = None) -> Any:
    """입력 검증.

    Args:
        groups: 입력 자료.
        profile: HomogeneityProfile.
    Returns:
        Any: 검증된 자료.
    Raises:
        ValueError/TypeError: 입력 조건 오류.
    """
    if not isinstance(profile, HomogeneityProfile):
        raise TypeError("HomogeneityProfile이 필요합니다")
    alpha(profile.alpha)
    if profile.method not in (
        "levene",
        "bartlett",
        "fligner",
    ) or profile.center not in ("mean", "median", "trimmed"):
        raise ValueError("method/center 오류")
    if len(groups) < 2:
        raise ValueError("2개 이상 독립 집단이 필요합니다.")
    arrays = [vector(g) for g in groups]
    if any(np.var(g, ddof=1) <= 0 for g in arrays):
        raise ValueError("각 집단의 양의 분산이 필요합니다.")
    if profile.method == "bartlett" and profile.center != "median":
        raise ValueError("Bartlett는 center를 사용하지 않습니다. 기본값을 유지하세요.")
    return arrays


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_test(data: Any, profile: Optional[HomogeneityProfile]) -> tuple:
    """검정 계산.

    Args:
        data: 검증 자료.
        profile: 검정 조건.
    Returns:
        tuple: 통계량, p 값, H0, 진단, 경고.
    Raises:
        ValueError: 미정의 통계량.
    """
    r = (
        stats.bartlett(*data)
        if profile.method == "bartlett"
        else getattr(stats, profile.method)(*data, center=profile.center)
    )
    if not np.isfinite(r.statistic) or not np.isfinite(r.pvalue):
        raise ValueError("미정의 검정: 집단의 중심 편차 분산을 확인하세요.")
    return (
        float(r.statistic),
        float(r.pvalue),
        "모든 집단의 모집단 분산이 같다",
        {
            "group_sizes": [len(g) for g in data],
            "method": profile.method,
            "center": None if profile.method == "bartlett" else profile.center,
        },
        (
            ["Bartlett는 각 집단의 정규성을 가정합니다."]
            if profile.method == "bartlett"
            else []
        ),
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_homogeneity(
    *groups: Any,
    profile: Optional[HomogeneityProfile] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> HomogeneityOutputDTO:
    """등분산 검정.

    Args:
        groups: 독립 집단; Bartlett는 정규성에 민감, median Levene는 Brown–Forsythe.
        profile: HomogeneityProfile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피에 적용할 설정.
    Returns:
        HomogeneityOutputDTO: 검정 결과, 진단과 경고.
    Raises:
        ValueError/TypeError: 입력·설정 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "homogeneity", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs(*groups, profile=profile)
    statistic, p, h0, details, notes = _run_test(data, profile)
    return HomogeneityOutputDTO(
        statistic, p, bool(p < profile.alpha), profile.alpha, h0, details, notes
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_homogeneity([1, 2, 5, 8, 9], [2, 3, 4, 6, 7]).to_json(indent=2))
