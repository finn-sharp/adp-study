"""
================================================================================
two_sample - 독립 두 표본 평균 Z/Student/Welch t
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 독립 집단; Student는 등분산, Welch는 이분산 허용.
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
from src.stats.tests._common import number, numeric, tail, mean_interval, result


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class TwoSampleMeanProfile:
    """TwoSampleMeanProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        diff_0 (float): 귀무 평균차 기본 0.0.
        known_var (bool): 알려진 두 모분산이면 Z 기본 False.
        sigma1 (Optional[float]): 집단1 모표준편차 기본 None.
        sigma2 (Optional[float]): 집단2 모표준편차 기본 None.
        equal_var (bool): Student/Welch 선택 기본 True.
        alternative (str): two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided".
        nan_policy (str): raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise".
    """

    alpha: float = 0.05
    diff_0: float = 0.0
    known_var: bool = False
    sigma1: Optional[float] = None
    sigma2: Optional[float] = None
    equal_var: bool = True
    alternative: str = "two-sided"
    nan_policy: str = "raise"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class TwoSampleMeanOutputDTO(OutputDTO):
    """TwoSampleMeanOutputDTO 전체 필드 계약.

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
        mean_difference (float): 원 입력 x−y 또는 표본 평균−기준; 방향 명시.
        standard_error (float): 평균차 표준오차.
        df (float): t 자유도; Z는 infinity.
        confidence_interval (Tuple[float, float]): 평균차/평균의 방향 일치 신뢰구간; 비유계 끝점 JSON null.
        n_effective (int): 유효 표본/대응 쌍 수.
        group_sizes (List[int]): 첫/둘째 표본 수.
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
    mean_difference: float
    standard_error: float
    df: float
    confidence_interval: Tuple[float, float]
    n_effective: int
    group_sizes: List[int]


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
    a = numeric(args["x"], 1, 2, policy)
    b = numeric(args["y"], 1, 2, policy) if "y" in args else None
    if p.known_var:
        number(p.sigma1, "sigma1", True)
        number(p.sigma2, "sigma2", True)
    return {"x": a, "y": b}


def _validate_inputs(
    arguments: Dict[str, Any], profile: TwoSampleMeanProfile
) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: TwoSampleMeanProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, TwoSampleMeanProfile):
        raise TypeError("TwoSampleMeanProfile이 필요합니다.")
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
    b = d["y"]
    n = len(a)
    m = len(b)
    v1 = a.var(ddof=1)
    v2 = b.var(ddof=1)
    value = float(a.mean() - b.mean())
    null = p.diff_0
    if p.known_var:
        se = np.sqrt(p.sigma1**2 / n + p.sigma2**2 / m)
        df = float("inf")
        method = "z"
    elif p.equal_var:
        df = n + m - 2
        se = np.sqrt(((n - 1) * v1 + (m - 1) * v2) / df * (1 / n + 1 / m))
        method = "student-t"
    else:
        vv1 = v1 / n
        vv2 = v2 / m
        se = np.sqrt(vv1 + vv2)
        df = (vv1 + vv2) ** 2 / (vv1**2 / (n - 1) + vv2**2 / (m - 1))
        method = "welch-t"
    extras = {"group_sizes": [n, m]}
    if not np.isfinite(se) or se <= 0:
        raise ValueError("양의 유한 표준오차가 필요합니다.")
    if method == "z":
        df = float("inf")
        dist = stats.norm
    else:
        dist = stats.t(df)
    s = (value - null) / se
    extras.update(
        mean_difference=value,
        standard_error=float(se),
        df=float(df),
        confidence_interval=mean_interval(value, se, p.alpha, alt, dist),
        n_effective=n + len(b),
    )
    return result(
        (
            "Two-sample Z-test"
            if method == "z"
            else (
                "Two-sample T-test (Equal Variance)"
                if method == "student-t"
                else "Welch's T-test (Unequal Variance)"
            )
        ),
        p,
        s,
        tail(s, alt, dist),
        "평균 또는 평균차가 지정 귀무값과 같다",
        method,
        extras,
        dict(
            null_value=null,
            estimate=value,
            **{
                "sample_mean1": float(a.mean()),
                "sample_mean2": float(b.mean()),
                "sample_std1": float(a.std(ddof=1)),
                "sample_std2": float(b.std(ddof=1)),
            },
        ),
    )


def _run_test(
    data: Dict[str, Any], profile: TwoSampleMeanProfile
) -> TwoSampleMeanOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        TwoSampleMeanOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return TwoSampleMeanOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_two_sample_mean(
    x: Any,
    y: Any,
    profile: Optional[TwoSampleMeanProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TwoSampleMeanOutputDTO:
    """독립 두 표본 평균 Z/Student/Welch t.

    Args:
        x: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        y: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 TwoSampleMeanProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        TwoSampleMeanOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "two-sample-mean", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs({"x": x, "y": y}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_two_sample_mean([1, 2, 4, 5, 8], [2, 3, 4, 6, 9])
    print(result.to_json(indent=2))
