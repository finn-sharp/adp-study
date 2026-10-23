"""
================================================================================
box_m - Box 공분산 동질성 M
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 독립 다변량 정규 그룹; 각 그룹 양의 정부호 공분산과 충분한 n.
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
from src.stats.tests._common import number, numeric, result, covariance


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class BoxMProfile:
    """BoxMProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        nan_policy (str): raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise".
    """

    alpha: float = 0.05
    nan_policy: str = "raise"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class BoxMOutputDTO(OutputDTO):
    """BoxMOutputDTO 전체 필드 계약.

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
        m_value (float): 보정 전 M.
        correction_factor (float): χ² 근사에 곱하는 1−c.
        df (int): (k−1)p(p+1)/2.
        covariances (List[np.ndarray]): 그룹 공분산.
        pooled_covariance (np.ndarray): pooled 공분산.
        group_sizes (List[int]): 그룹 n.
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
    m_value: float
    correction_factor: float
    df: int
    covariances: List[np.ndarray]
    pooled_covariance: np.ndarray
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
    if len(args["groups"]) < 2:
        raise ValueError("2개 이상 독립 그룹 필요")
    gs = [numeric(x, 2, 2, policy) for x in args["groups"]]
    if len({a.shape[1] for a in gs}) != 1:
        raise ValueError("특성 수 불일치")
    return {"groups": gs}


def _validate_inputs(arguments: Dict[str, Any], profile: BoxMProfile) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: BoxMProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, BoxMProfile):
        raise TypeError("BoxMProfile이 필요합니다.")
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
    gs = d["groups"]
    ns = [len(a) for a in gs]
    covs = [covariance(a) for a in gs]
    q = gs[0].shape[1]
    N = sum(ns)
    k = len(gs)
    pooled = sum(((n - 1) * c for n, c in zip(ns, covs))) / (N - k)
    M = float(
        (N - k) * np.linalg.slogdet(pooled)[1]
        - sum(((n - 1) * np.linalg.slogdet(c)[1] for n, c in zip(ns, covs)))
    )
    c = (
        (2 * q * q + 3 * q - 1)
        / (6 * (q + 1) * (k - 1))
        * (sum((1 / (n - 1) for n in ns)) - 1 / (N - k))
    )
    factor = 1 - c
    if factor <= 0:
        raise ValueError("Box M 보정 근사를 위한 표본이 부족합니다.")
    df = (k - 1) * q * (q + 1) // 2
    s = max(0.0, M) * factor
    return result(
        "box-m",
        p,
        s,
        stats.chi2.sf(s, df),
        "그룹의 공분산 행렬이 같다",
        "corrected-chi-square",
        dict(
            m_value=M,
            correction_factor=factor,
            df=df,
            covariances=covs,
            pooled_covariance=pooled,
            group_sizes=ns,
        ),
        notes=["다변량 정규성에 민감한 점근 검정입니다."],
    )


def _run_test(data: Dict[str, Any], profile: BoxMProfile) -> BoxMOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        BoxMOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return BoxMOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_box_m(
    *groups: Any,
    profile: Optional[BoxMProfile] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> BoxMOutputDTO:
    """Box 공분산 동질성 M.

    Args:
        groups: 독립 표본 목록.
        profile: 명시적 BoxMProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        BoxMOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "box-m", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs({"groups": groups}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    result = run_box_m(rng.normal(size=(15, 2)), rng.normal(size=(18, 2)) * 1.5)
    print(result.to_json(indent=2))
