"""
================================================================================
chi_square.py - 카이제곱 적합도 검정
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 독립 관측의 범주별 빈도; 기대도수 합은 관측 합과 동일.
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
class ChiSquareProfile:
    """ChiSquareProfile 필드 계약.

    Attributes:
        alpha (float): 유의수준, 기본 0.05.
        ddof (int): 추정 모수 수에 따른 자유도 감소, 기본 0.
    """
    alpha: float = 0.05
    ddof: int = 0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class ChiSquareOutputDTO(OutputDTO):
    """ChiSquareOutputDTO 필드 계약.

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

def _validate_inputs(observed: Any, expected: Any = None, profile: Optional[ChiSquareProfile] = None) -> Any:
    """입력 검증.

    Args:
        observed: 독립 관측의 범주별 비음수 정수 빈도.
        expected: 같은 총빈도의 양의 기대도수 또는 None(균등 기대도수).
        profile: ChiSquareProfile.
    Returns:
        Any: 검증된 자료.
    Raises:
        ValueError/TypeError: 입력 조건 오류.
    """
    if not isinstance(profile, ChiSquareProfile): raise TypeError("ChiSquareProfile이 필요합니다")
    alpha(profile.alpha)
    obs = counts(observed)
    if len(obs) < 2 or obs.sum() <= 0: raise ValueError("2개 이상 범주와 양의 총빈도가 필요합니다.")
    exp = np.full(len(obs), obs.mean()) if expected is None else np.asarray(expected, dtype=float)
    if exp.shape != obs.shape or not np.isfinite(exp).all() or (exp <= 0).any() or not np.isclose(exp.sum(), obs.sum(), rtol=1e-8, atol=1e-8): raise ValueError("양의 기대도수와 같은 총빈도가 필요합니다.")
    positive_int(profile.ddof, "ddof", 0)
    if len(obs)-1-profile.ddof <= 0: raise ValueError("자유도는 양수여야 합니다.")
    return obs, exp


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_test(data: Any, profile: Optional[ChiSquareProfile]) -> tuple:
    """검정 계산.

    Args:
        data: 검증 자료.
        profile: 검정 조건.
    Returns:
        tuple: 통계량, p 값, H0, 진단, 경고.
    Raises:
        ValueError: 미정의 통계량.
    """
    obs, exp = data
    result = stats.chisquare(obs, exp, ddof=profile.ddof)
    return float(result.statistic), float(result.pvalue), "범주 분포가 지정 기대 분포와 같다", {"df": len(obs)-1-profile.ddof, "observed": obs, "expected": exp}, (["기대도수 5 미만: 카이제곱 근사의 적절성을 확인하세요."] if (exp < 5).any() else [])


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_chi_square(observed: Any, expected: Any = None, profile: Optional[ChiSquareProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ChiSquareOutputDTO:
    """카이제곱 적합도 검정.

    Args:
        observed: 독립 관측의 범주별 비음수 정수 빈도.
        expected: 같은 총빈도의 양의 기대도수 또는 None(균등 기대도수).
        profile: ChiSquareProfile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피에 적용할 설정.
    Returns:
        ChiSquareOutputDTO: 검정 결과, 진단과 경고.
    Raises:
        ValueError/TypeError: 입력·설정 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile("chi-square", profile, recipe_path=recipe_path, overrides=overrides)
    data = _validate_inputs(observed, expected, profile=profile)
    statistic, p, h0, details, notes = _run_test(data, profile)
    return ChiSquareOutputDTO(statistic, p, bool(p < profile.alpha), profile.alpha, h0, details, notes)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_chi_square([20, 30, 25, 25]).to_json(indent=2))
