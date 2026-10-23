"""
================================================================================
runs - Wald–Wolfowitz runs 검정
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 원 순서가 있는 이진 자료 또는 기준으로 이진화한 수치; 순서 무작위성.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    통계 PDF 전체 범위 모듈화·입력 검증·명세
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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.output import OutputDTO
from src.Recipes import resolve_profile
from scipy import stats
from src.stats.tests._validation import alpha
from src.stats.tests._common import number, numeric, tail, result


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class RunsProfile:
    """RunsProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        cutoff (Optional[float]): None은 입력 0/1만 허용. 수치 기준일 때 동일 값은 제외 기본 None.
        correction (bool): 양측 근사 z 연속성 보정 기본 True.
        nan_policy (str): raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise".
    """

    alpha: float = 0.05
    cutoff: Optional[float] = None
    correction: bool = True
    nan_policy: str = "raise"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class RunsOutputDTO(OutputDTO):
    """RunsOutputDTO 전체 필드 계약.

    Attributes:
        test_type (str): 검정 이름.
        statistic (float): 기법별 통계량; OR/RR은 log scale z.
        p_value (Optional[float]): p 값; 임계값 기반 AD에는 None.
        reject_h0 (bool): p<alpha 또는 명시한 임계값 비교.
        alpha (float): 적용 유의수준.
        null_hypothesis (str): 검정한 귀무가설.
        alternative (str): 대립가설 방향/omnibus.
        method_used (str): 실제 사용한 정확/근사/보정 방법.
        details (Dict[str, Any]): 문서화된 기법별 진단 필드.
        warnings (List[str]): 가정·작은 표본·근사 제한.
        runs (int): 연속 동일 값 묶음 수.
        expected_runs (float): 귀무 기대 run 수.
        variance_runs (float): 귀무 run 분산.
        group_counts (List[int]): 0/1 관측 수.
    """

    test_type: str
    statistic: float
    p_value: Optional[float]
    reject_h0: bool
    alpha: float
    null_hypothesis: str
    alternative: str
    method_used: str
    details: Dict[str, Any]
    warnings: List[str]
    runs: int
    expected_runs: float
    variance_runs: float
    group_counts: List[int]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_data(args, p):
    """공개 입력을 설계별로 정규화합니다.

    Args:
        args: 공개 실행기에 전달한 원본 자료의 사전.
        p: 해당 기법의 입력 Profile.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    alpha(p.alpha)
    alt = getattr(p, "alternative", "two-sided")
    if alt not in ("two-sided", "less", "greater"):
        raise ValueError("alternative 오류")
    for name in ("known_var", "equal_var", "correction", "estimated_parameters"):
        if hasattr(p, name) and (not isinstance(getattr(p, name), bool)):
            raise ValueError(name + "는 bool입니다.")
    policy = getattr(p, "nan_policy", "raise")
    if policy not in ("raise", "omit"):
        raise ValueError("nan_policy 오류")
    for name in ("mu_0", "diff_0", "location_0"):
        if hasattr(p, name):
            number(getattr(p, name), name)
    a = numeric(args["x"], 1, 3, policy)
    if p.cutoff is None:
        if not np.isin(a, [0, 1]).all():
            raise ValueError("cutoff 생략 시 0/1 자료 필요")
    else:
        number(p.cutoff, "cutoff")
        a = a[a != p.cutoff]
        a = (a > p.cutoff).astype(int)
    if len(a) < 3 or len(np.unique(a)) != 2:
        raise ValueError("두 상태와 3개 이상 유효 관측 필요")
    return {"x": a}


def _validate_inputs(arguments: Dict[str, Any], profile: RunsProfile) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: RunsProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, RunsProfile):
        raise TypeError("RunsProfile이 필요합니다.")
    return _validate_data(arguments, profile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _calculate(d, p):
    """기법별 통계 계산과 전용 결과 필드를 제공합니다.

    Args:
        d: 입력 검증을 통과한 자료의 사전.
        p: 해당 기법의 입력 Profile.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    alt = getattr(p, "alternative", "two-sided")
    a = d["x"]
    n1 = int((a == 1).sum())
    n0 = len(a) - n1
    n = n0 + n1
    runs = 1 + int((a[1:] != a[:-1]).sum())
    expected = 1 + 2 * n0 * n1 / n
    var = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n * n * (n - 1))
    if var <= 0:
        raise ValueError("양의 run 분산 필요")
    delta = runs - expected
    if p.correction:
        delta = np.sign(delta) * max(0, abs(delta) - 0.5)
    z = delta / np.sqrt(var)
    return result(
        "runs",
        p,
        z,
        tail(z, "two-sided", stats.norm),
        "이진 관측 순서가 무작위이다",
        "normal-approximation",
        dict(
            runs=runs, expected_runs=expected, variance_runs=var, group_counts=[n0, n1]
        ),
        notes=["순서의 무작위성 검정; 관측을 정렬하면 안 됩니다."],
    )


def _run_test(data: Dict[str, Any], profile: RunsProfile) -> RunsOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        RunsOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return RunsOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_runs(
    x: Any,
    profile: Optional[RunsProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RunsOutputDTO:
    """Wald–Wolfowitz runs 검정.

    Args:
        x: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 RunsProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        RunsOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "runs", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs({"x": x}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_runs([0, 0, 1, 0, 1, 1, 0, 1])
    print(result.to_json(indent=2))
