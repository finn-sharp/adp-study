"""
================================================================================
fisher_exact.py - fisher_exact - Fisher 정확·RxC 조건부 검정
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 고정 주변합 도수표의 조건부 독립성. 2×2 exact와 RxC 열거/Monte Carlo 구분.
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
import math
from scipy import stats
from src.stats.tests._validation import alpha
from src.stats.tests._common import (
    number,
    integer,
    table_data,
    result,
    log_table_probability,
    enumerate_tables,
    sample_table,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class FisherExactProfile:
    """FisherExactProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        method (str): auto:2×2 exact, RxC enumeration; exact/enumeration/monte_carlo 기본 "auto".
        max_tables (int): RxC 정확 열거 최대 표 수. 초과 오류 기본 100000.
        n_resamples (int): Monte Carlo 표 수, 양수 기본 10000.
        random_state (int): Monte Carlo seed, 0 이상 기본 1234.
        alternative (str): two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided".
    """

    alpha: float = 0.05
    method: str = "auto"
    max_tables: int = 100000
    n_resamples: int = 10000
    random_state: int = 1234
    alternative: str = "two-sided"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class FisherExactOutputDTO(OutputDTO):
    """FisherExactOutputDTO 전체 필드 계약.

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
        odds_ratio (Optional[float]): 2×2 cross-product OR; RxC None.
        n_tables (int): 열거/재표본 표 수. 2×2 기본0.
        mc_standard_error (Optional[float]): Monte Carlo p 추정 SE, 정확 모드 None.
        table_probability (float): 관측표의 고정 주변합 조건부 확률.
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
    odds_ratio: Optional[float]
    n_tables: int
    mc_standard_error: Optional[float]
    table_probability: float


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
    a = table_data(args["table"], False)
    if p.method not in ("auto", "exact", "enumeration", "monte_carlo"):
        raise ValueError("method 오류")
    integer(p.max_tables, "max_tables", 1)
    integer(p.n_resamples, "n_resamples", 1)
    integer(p.random_state, "random_state")
    if a.shape != (2, 2) and alt != "two-sided":
        raise ValueError("RxC는 확률 순서의 양측 검정만 지원합니다.")
    if a.shape != (2, 2) and p.method == "exact":
        raise ValueError("RxC는 enumeration/monte_carlo를 지정하세요.")
    return {"table": a}


def _validate_inputs(
    arguments: Dict[str, Any], profile: FisherExactProfile
) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: FisherExactProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, FisherExactProfile):
        raise TypeError("FisherExactProfile이 필요합니다.")
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
    t = d["table"]
    logobs = log_table_probability(t)
    shape = t.shape
    method = p.method
    if method == "auto":
        method = "exact" if shape == (2, 2) else "enumeration"
    odds = (
        float(t[0, 0] * float(t[1, 1]) / (float(t[0, 1]) * t[1, 0]))
        if shape == (2, 2) and float(t[0, 1]) * float(t[1, 0]) > 0
        else float("inf") if shape == (2, 2) else None
    )
    se = None
    n = 0
    if method == "exact":
        s, pv = stats.fisher_exact(t, alternative=alt)
    else:
        if alt != "two-sided":
            raise ValueError("확률 순서 검정은 양측만 지원합니다.")
        rows = t.sum(axis=1)
        cols = t.sum(axis=0)
        if method == "enumeration":
            mass = []
            total = []
            for candidate in enumerate_tables(rows, cols, p.max_tables):
                lp = log_table_probability(candidate)
                v = math.exp(lp)
                total.append(v)
                n += 1
                if lp <= logobs + 1e-12:
                    mass.append(v)
            if not np.isclose(math.fsum(total), 1.0, atol=1e-07, rtol=1e-07):
                raise ValueError("정확 열거의 확률 질량 검증 실패")
            pv = min(1.0, math.fsum(mass))
        else:
            rng = np.random.RandomState(p.random_state)
            hits = 0
            n = p.n_resamples
            for _ in range(n):
                hits += (
                    log_table_probability(sample_table(rows, cols, rng))
                    <= logobs + 1e-12
                )
            pv = (hits + 1) / (n + 1)
            se = float(np.sqrt(pv * (1 - pv) / (n + 1)))
        s = math.exp(logobs)
    return result(
        "fisher-exact",
        p,
        s,
        pv,
        "고정 주변합 하에서 행과 열 변수가 독립이다",
        method,
        dict(
            odds_ratio=odds,
            n_tables=n,
            mc_standard_error=se,
            table_probability=math.exp(logobs),
        ),
        {
            "statistic_definition": (
                "cross-product OR"
                if method == "exact"
                else "conditional table probability"
            ),
            "random_state": p.random_state if method == "monte_carlo" else None,
        },
        ["Monte Carlo p는 +1 보정한 추정값입니다."] if se is not None else [],
    )


def _run_test(
    data: Dict[str, Any], profile: FisherExactProfile
) -> FisherExactOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        FisherExactOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return FisherExactOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_fisher_exact(
    table: Any,
    profile: Optional[FisherExactProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> FisherExactOutputDTO:
    """Fisher 정확·RxC 조건부 검정.

    Args:
        table: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 FisherExactProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        FisherExactOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "fisher-exact", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs({"table": table}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_fisher_exact([[2, 8], [7, 3]])
    print(result.to_json(indent=2))
