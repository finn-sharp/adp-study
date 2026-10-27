"""
================================================================================
sign.py - sign - 부호 검정
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 단일 또는 대응 차이의 부호; 0차이는 제외, 연속 중앙값 가정.
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
from src.stats.tests._common import number, numeric, paired, result


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class SignProfile:
    """SignProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        sample_design (str): one_sample/paired 기본 "one_sample".
        location_0 (float): 단일 기준 중앙값 또는 귀무 차이 기본 0.0.
        alternative (str): two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided".
        nan_policy (str): raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise".
    """

    alpha: float = 0.05
    sample_design: str = "one_sample"
    location_0: float = 0.0
    alternative: str = "two-sided"
    nan_policy: str = "raise"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class SignOutputDTO(OutputDTO):
    """SignOutputDTO 전체 필드 계약.

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
        positive_count (int): 양의 차이 수.
        negative_count (int): 음의 차이 수.
        zero_count (int): 제외한 0 차이 수.
        n_effective (int): 양+음 부호 수.
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
    positive_count: int
    negative_count: int
    zero_count: int
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
    return {"d": d}


def _validate_inputs(arguments: Dict[str, Any], profile: SignProfile) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: SignProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, SignProfile):
        raise TypeError("SignProfile이 필요합니다.")
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
    if n == 0:
        raise ValueError("0이 아닌 차이가 필요합니다.")
    pos = int((nonzero > 0).sum())
    neg = n - pos
    pv = stats.binomtest(pos, n, 0.5, alternative=alt).pvalue
    return result(
        "sign",
        p,
        pos,
        pv,
        "중앙값/대응 차이 중앙값이 location_0와 같다",
        "exact-binomial",
        dict(positive_count=pos, negative_count=neg, zero_count=zeros, n_effective=n),
    )


def _run_test(data: Dict[str, Any], profile: SignProfile) -> SignOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        SignOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return SignOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_sign(
    x: Any,
    y: Any = None,
    profile: Optional[SignProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SignOutputDTO:
    """부호 검정.

    Args:
        x: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        y: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 SignProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        SignOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "sign", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs({"x": x, "y": y}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_sign([1, 2, -3, 4, 5])
    print(result.to_json(indent=2))
