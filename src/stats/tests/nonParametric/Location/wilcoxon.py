"""
================================================================================
wilcoxon.py - wilcoxon - Wilcoxon 부호순위
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 차이분포의 0 중심 대칭성; 0차이·동률은 근사로 명시 처리.
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
import inspect
import warnings as pywarnings
from scipy import stats
from src.stats.tests._validation import alpha
from src.stats.tests._common import number, numeric, paired, result


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class WilcoxonProfile:
    """WilcoxonProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        sample_design (str): one_sample/paired 기본 "one_sample".
        location_0 (float): 귀무 위치 차이 기본 0.0.
        method (str): auto/exact/asymptotic 기본 "auto".
        zero_method (str): wilcox/pratt/zsplit 기본 "wilcox".
        correction (bool): 근사 연속성 보정 기본 False.
        alternative (str): two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided".
        nan_policy (str): raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise".
    """

    alpha: float = 0.05
    sample_design: str = "one_sample"
    location_0: float = 0.0
    method: str = "auto"
    zero_method: str = "wilcox"
    correction: bool = False
    alternative: str = "two-sided"
    nan_policy: str = "raise"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class WilcoxonOutputDTO(OutputDTO):
    """WilcoxonOutputDTO 전체 필드 계약.

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
        n_effective (int): 0이 아닌 차이 수.
        zero_count (int): 0 차이 수.
        positive_rank_sum (float): 양의 차이 순위 합.
        negative_rank_sum (float): 음의 차이 순위 합.
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
    n_effective: int
    zero_count: int
    positive_rank_sum: float
    negative_rank_sum: float


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
    if p.sample_design not in ("one_sample", "paired"):
        raise ValueError("sample_design 오류")
    if p.sample_design == "paired":
        if args["y"] is None:
            raise ValueError("paired는 y가 필요합니다.")
        a, b = paired(args["x"], args["y"], policy)
        d = a - b - p.location_0
    else:
        if args["y"] is not None:
            raise ValueError("one_sample에서는 y를 지정하지 않습니다.")
        d = numeric(args["x"], 1, 1, policy) - p.location_0
    if not np.isfinite(d).all():
        raise ValueError("차이 계산 overflow입니다.")
    if p.method not in ("auto", "exact", "asymptotic") or p.zero_method not in (
        "wilcox",
        "pratt",
        "zsplit",
    ):
        raise ValueError("method/zero_method 오류")
    return {"d": d}


def _validate_inputs(
    arguments: Dict[str, Any], profile: WilcoxonProfile
) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: WilcoxonProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, WilcoxonProfile):
        raise TypeError("WilcoxonProfile이 필요합니다.")
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
    diff = d["d"]
    zeros = int((diff == 0).sum())
    nonzero = diff[diff != 0]
    n = len(nonzero)
    ranks = stats.rankdata(np.abs(nonzero if p.zero_method == "wilcox" else diff))
    used = nonzero if p.zero_method == "wilcox" else diff
    wp = float(ranks[used > 0].sum())
    wm = float(ranks[used < 0].sum())
    if p.zero_method == "zsplit":
        wp += float(ranks[used == 0].sum() / 2)
        wm += float(ranks[used == 0].sum() / 2)
    extras = dict(
        n_effective=n, zero_count=zeros, positive_rank_sum=wp, negative_rank_sum=wm
    )
    if n == 0:
        return result(
            "wilcoxon",
            p,
            min(wp, wm) if alt == "two-sided" else wp,
            1.0,
            "차이분포가 지정 위치를 중심으로 대칭이다",
            "degenerate-all-zero",
            extras,
            notes=["모든 차이가 0입니다. 순위 추론 정보가 없습니다."],
        )
    ties = len(np.unique(np.abs(nonzero))) < n
    method = (
        ("exact" if n <= 50 and (not ties) and (not zeros) else "asymptotic")
        if p.method == "auto"
        else p.method
    )
    if method == "exact" and (ties or zeros):
        raise ValueError("exact는 동률/0차이 없는 자료만 지원합니다.")
    kwargs = {
        "alternative": alt,
        "zero_method": p.zero_method,
        "correction": p.correction,
    }
    name = (
        "method" if "method" in inspect.signature(stats.wilcoxon).parameters else "mode"
    )
    kwargs[name] = (
        method if name == "method" else "approx" if method == "asymptotic" else method
    )
    with pywarnings.catch_warnings(record=True) as caught:
        r = stats.wilcoxon(diff, **kwargs)
    return result(
        "wilcoxon",
        p,
        r.statistic,
        r.pvalue,
        "차이분포가 지정 위치를 중심으로 대칭이다",
        method,
        extras,
        {"ties": ties},
        [str(w.message) for w in caught],
    )


def _run_test(data: Dict[str, Any], profile: WilcoxonProfile) -> WilcoxonOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        WilcoxonOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return WilcoxonOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_wilcoxon(
    x: Any,
    y: Any = None,
    profile: Optional[WilcoxonProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> WilcoxonOutputDTO:
    """Wilcoxon 부호순위.

    Args:
        x: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        y: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 WilcoxonProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        WilcoxonOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "wilcoxon", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs({"x": x, "y": y}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_wilcoxon([1, 2, -3, 4, 5])
    print(result.to_json(indent=2))
