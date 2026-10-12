"""
================================================================================
mcnemar.py - McNemar 대응 이진 검정
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 동일 개체의 두 이진 관측 2×2 빈도표; 비대각 셀이 불일치쌍.
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
class McNemarProfile:
    """McNemarProfile 필드 계약.

    Attributes:
        alpha (float): 유의수준, 기본 0.05.
        exact (bool): 이항 정확 검정, 기본 True.
        correction (bool): 근사 검정의 연속성 보정, exact에서는 무시. 기본 True.
    """
    alpha: float = 0.05
    exact: bool = True
    correction: bool = True


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class McNemarOutputDTO(OutputDTO):
    """McNemarOutputDTO 필드 계약.

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

def _validate_inputs(table: Any, profile: Optional[McNemarProfile] = None) -> Any:
    """입력 검증.

    Args:
        table: 입력 자료.
        profile: McNemarProfile.
    Returns:
        Any: 검증된 자료.
    Raises:
        ValueError/TypeError: 입력 조건 오류.
    """
    if not isinstance(profile, McNemarProfile): raise TypeError("McNemarProfile이 필요합니다")
    alpha(profile.alpha)
    a = counts(table, table=True)
    if a.shape != (2, 2) or a.sum() <= 0: raise ValueError("2×2 양의 총빈도 대응 표가 필요합니다.")
    if not isinstance(profile.exact, bool) or not isinstance(profile.correction, bool): raise ValueError("exact/correction은 bool입니다.")
    if not profile.exact and a[0,1]+a[1,0] == 0: raise ValueError("근사 검정은 불일치쌍이 필요합니다.")
    return a


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_test(data: Any, profile: Optional[McNemarProfile]) -> tuple:
    """검정 계산.

    Args:
        data: 검증 자료.
        profile: 검정 조건.
    Returns:
        tuple: 통계량, p 값, H0, 진단, 경고.
    Raises:
        ValueError: 미정의 통계량.
    """
    from statsmodels.stats.contingency_tables import mcnemar
    r = mcnemar(data, exact=profile.exact, correction=profile.correction)
    discordant = int(data[0,1]+data[1,0])
    return float(r.statistic), float(r.pvalue), "두 대응 이진 관측의 주변 비율이 같다", {"discordant_pairs": discordant, "exact": profile.exact, "correction_applied": not profile.exact and profile.correction, "df": None if profile.exact else 1}, (["불일치쌍이 적습니다. exact=True를 권장합니다."] if not profile.exact and discordant < 25 else [])


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_mcnemar(table: Any, profile: Optional[McNemarProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> McNemarOutputDTO:
    """McNemar 대응 이진 검정.

    Args:
        table: 동일 개체의 두 이진 관측 2×2 빈도표; 비대각 셀이 불일치쌍.
        profile: McNemarProfile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피에 적용할 설정.
    Returns:
        McNemarOutputDTO: 검정 결과, 진단과 경고.
    Raises:
        ValueError/TypeError: 입력·설정 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile("mcnemar", profile, recipe_path=recipe_path, overrides=overrides)
    data = _validate_inputs(table, profile=profile)
    statistic, p, h0, details, notes = _run_test(data, profile)
    return McNemarOutputDTO(statistic, p, bool(p < profile.alpha), profile.alpha, h0, details, notes)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_mcnemar([[20, 8], [2, 30]]).to_json(indent=2))
