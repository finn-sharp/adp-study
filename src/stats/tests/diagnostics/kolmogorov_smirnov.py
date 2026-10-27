"""
================================================================================
kolmogorov_smirnov.py - kolmogorov_smirnov - KS 또는 Lilliefors 적합도
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 완전히 지정한 연속 CDF의 KS. 표본 추정 정규/지수는 Lilliefors.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    통계 PDF 전체 범위 모듈화·입력 검증·명세
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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.output import OutputDTO
from src.Recipes import resolve_profile
from scipy import stats
from src.stats.tests._validation import alpha
from src.stats.tests._common import number, numeric, result


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class KolmogorovSmirnovProfile:
    """KolmogorovSmirnovProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        distribution (str): norm/expon/uniform/logistic 기본 "norm".
        parameters (Optional[Dict[str, float]]): 고정 분포 모수 loc/scale; None은 기본 loc0/scale1 기본 None.
        estimated_parameters (bool): True는 norm/expon Lilliefors 보정; 고정 parameters와 동시 금지 기본 False.
        alternative (str): two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided".
        nan_policy (str): raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise".
    """

    alpha: float = 0.05
    distribution: str = "norm"
    parameters: Optional[Dict[str, float]] = None
    estimated_parameters: bool = False
    alternative: str = "two-sided"
    nan_policy: str = "raise"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class KolmogorovSmirnovOutputDTO(OutputDTO):
    """KolmogorovSmirnovOutputDTO 전체 필드 계약.

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
        distribution_parameters (Optional[Dict[str, float]]): 고정 또는 추정한 loc/scale.
        n_effective (int): 유효 관측 수.
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
    distribution_parameters: Dict[str, float]
    n_effective: int


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
    a = numeric(args["x"], 1, 4, policy)
    if np.ptp(a) == 0:
        raise ValueError("상수 자료의 검정은 정의되지 않습니다.")
    if p.distribution not in ("norm", "expon", "uniform", "logistic"):
        raise ValueError("KS 지원 분포 오류")
    if p.parameters is not None:
        if not isinstance(p.parameters, dict) or set(p.parameters) - {"loc", "scale"}:
            raise ValueError("parameters는 loc/scale 사전입니다.")
        for name, value in p.parameters.items():
            number(value, name, name == "scale")
    if p.estimated_parameters and (
        p.parameters is not None
        or p.distribution not in ("norm", "expon")
        or alt != "two-sided"
    ):
        raise ValueError("Lilliefors는 norm/expon 양측 및 고정 모수 없는 경우입니다.")
    if p.estimated_parameters and p.distribution == "expon" and (a < 0).any():
        raise ValueError("지수 Lilliefors는 loc=0 비음수 자료입니다.")
    return {"x": a}


def _validate_inputs(
    arguments: Dict[str, Any], profile: KolmogorovSmirnovProfile
) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: KolmogorovSmirnovProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, KolmogorovSmirnovProfile):
        raise TypeError("KolmogorovSmirnovProfile이 필요합니다.")
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
    params = {"loc": 0.0, "scale": 1.0}
    params.update(p.parameters or {})
    if p.estimated_parameters:
        from statsmodels.stats.diagnostic import lilliefors

        s, pv = lilliefors(
            a, dist="norm" if p.distribution == "norm" else "exp", pvalmethod="table"
        )
        params = (
            {"loc": float(a.mean()), "scale": float(a.std(ddof=1))}
            if p.distribution == "norm"
            else {"loc": 0.0, "scale": float(a.mean())}
        )
        method = "lilliefors-table"
    else:
        frozen = getattr(stats, p.distribution)(**params)
        r = stats.kstest(a, frozen.cdf, alternative=alt)
        s, pv = (r.statistic, r.pvalue)
        method = "ks-fixed-parameters"
    return result(
        "kolmogorov-smirnov",
        p,
        s,
        pv,
        "표본 CDF가 지정 연속분포 CDF와 같다",
        method,
        dict(distribution_parameters=params, n_effective=len(a)),
    )


def _run_test(
    data: Dict[str, Any], profile: KolmogorovSmirnovProfile
) -> KolmogorovSmirnovOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        KolmogorovSmirnovOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return KolmogorovSmirnovOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_kolmogorov_smirnov(
    x: Any,
    profile: Optional[KolmogorovSmirnovProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> KolmogorovSmirnovOutputDTO:
    """KS 또는 Lilliefors 적합도.

    Args:
        x: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 KolmogorovSmirnovProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        KolmogorovSmirnovOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "kolmogorov-smirnov", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs({"x": x}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_kolmogorov_smirnov([0.1, 0.3, 0.5, 0.7, 1.1, 1.3, 1.7, 2.1])
    print(result.to_json(indent=2))
