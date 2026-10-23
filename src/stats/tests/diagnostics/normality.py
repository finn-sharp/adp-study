"""
================================================================================
normality.py - 정규성 검정
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 독립 동일 분포 수치 표본; 기각하지 않음은 정규성 증명이 아님.
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
class NormalityProfile:
    """NormalityProfile 필드 계약.

    Attributes:
        alpha (float): 유의수준, 기본 0.05.
        method (str): shapiro/dagostino/jarque_bera. 기본 shapiro.
    """

    alpha: float = 0.05
    method: str = "shapiro"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class NormalityOutputDTO(OutputDTO):
    """NormalityOutputDTO 필드 계약.

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


def _validate_inputs(x: Any, profile: Optional[NormalityProfile] = None) -> Any:
    """입력 검증.

    Args:
        x: 입력 자료.
        profile: NormalityProfile.
    Returns:
        Any: 검증된 자료.
    Raises:
        ValueError/TypeError: 입력 조건 오류.
    """
    if not isinstance(profile, NormalityProfile):
        raise TypeError("NormalityProfile이 필요합니다")
    alpha(profile.alpha)
    if profile.method not in ("shapiro", "dagostino", "jarque_bera"):
        raise ValueError("method 오류")
    a = vector(x, 8 if profile.method == "dagostino" else 3)
    if np.ptp(a) == 0:
        raise ValueError("상수 표본은 정규성 검정 불가")
    return a


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_test(data: Any, profile: Optional[NormalityProfile]) -> tuple:
    """검정 계산.

    Args:
        data: 검증 자료.
        profile: 검정 조건.
    Returns:
        tuple: 통계량, p 값, H0, 진단, 경고.
    Raises:
        ValueError: 미정의 통계량.
    """
    func = {
        "shapiro": stats.shapiro,
        "dagostino": stats.normaltest,
        "jarque_bera": stats.jarque_bera,
    }[profile.method]
    r = func(data)
    notes = []
    if profile.method == "shapiro" and len(data) > 5000:
        notes.append("Shapiro n>5000에서 p 값 정확성이 제한됩니다.")
    if profile.method == "jarque_bera" and len(data) < 2000:
        notes.append("Jarque–Bera는 큰 표본의 점근 근사입니다.")
    return (
        float(r.statistic),
        float(r.pvalue),
        "표본은 정규 모집단에서 추출되었다",
        {"n": len(data), "method": profile.method},
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_normality(
    x: Any,
    profile: Optional[NormalityProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> NormalityOutputDTO:
    """정규성 검정.

    Args:
        x: 독립 동일 분포 수치 표본; 기각하지 않음은 정규성 증명이 아님.
        profile: NormalityProfile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피에 적용할 설정.
    Returns:
        NormalityOutputDTO: 검정 결과, 진단과 경고.
    Raises:
        ValueError/TypeError: 입력·설정 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "normality", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs(x, profile=profile)
    statistic, p, h0, details, notes = _run_test(data, profile)
    return NormalityOutputDTO(
        statistic, p, bool(p < profile.alpha), profile.alpha, h0, details, notes
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_normality([1, 2, 3, 4, 5, 6, 8, 10]).to_json(indent=2))
