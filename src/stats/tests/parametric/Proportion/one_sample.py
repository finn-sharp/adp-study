"""
================================================================================
one_sample - one 모비율 검정
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 이진 성공 수/시행 수; 독립 시행. 두 표본은 pooled 귀무 분산의 Z.
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
from src.stats.tests._common import (number, integer, tail, result)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class OneSampleProportionProfile:
    """OneSampleProportionProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        p_0 (float): 귀무 성공 확률 0<p0<1 기본 0.5.
        method (str): exact/normal_score; exact는 이항 검정 기본 "exact".
        alternative (str): two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided".
    """
    alpha: float = 0.05
    p_0: float = 0.5
    method: str = "exact"
    alternative: str = "two-sided"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class OneSampleProportionOutputDTO(OutputDTO):
    """OneSampleProportionOutputDTO 전체 필드 계약.

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
        proportions (List[float]): 표본 성공률.
        difference (float): 단일 p_hat−p0 또는 p1−p2.
        confidence_intervals (List[Tuple[float, float]]): 단일 exact Clopper–Pearson/score Wilson, 두 표본은 각 Wilson 구간.
        difference_interval (Optional[Tuple[float, float]]): 두 표본 독립 Newcombe 점수 차이 구간; 단일 None.
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
    proportions: List[float]
    difference: float
    confidence_intervals: List[Tuple[float, float]]
    difference_interval: Optional[Tuple[float, float]]


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
    alt = getattr(p, 'alternative', 'two-sided')
    if alt not in ('two-sided', 'less', 'greater'):
        raise ValueError('alternative 오류')
    for name in ('known_var', 'equal_var', 'correction', 'estimated_parameters'):
        if hasattr(p, name) and (not isinstance(getattr(p, name), bool)):
            raise ValueError(name + '는 bool입니다.')
    policy = getattr(p, 'nan_policy', 'raise')
    if policy not in ('raise', 'omit'):
        raise ValueError('nan_policy 오류')
    for name in ('mu_0', 'diff_0', 'location_0'):
        if hasattr(p, name):
            number(getattr(p, name), name)
    pairs = [('successes', 'n')]
    d = {}
    for s, n in pairs:
        d[s] = integer(args[s], s)
        d[n] = integer(args[n], n, 1)
        if d[s] > d[n]:
            raise ValueError('성공 수는 시행 수 이하여야 합니다.')
    if not 0 < number(p.p_0, 'p_0') < 1 or p.method not in ('exact', 'normal_score'):
        raise ValueError('p_0/method 오류')
    return d

def _validate_inputs(arguments: Dict[str, Any], profile: OneSampleProportionProfile) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: OneSampleProportionProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, OneSampleProportionProfile): raise TypeError("OneSampleProportionProfile이 필요합니다.")
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
    alt = getattr(p, 'alternative', 'two-sided')
    from statsmodels.stats.proportion import proportion_confint
    ia = p.alpha if alt == 'two-sided' else 2 * p.alpha
    if ia >= 1:
        raise ValueError('단측 비율 구간은 alpha<0.5가 필요합니다.')
    notes = []

    def ci(s, n, method):
        """ci의 공유 계산을 수행합니다.

            Args:
                s: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
                n: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
                method: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
            Returns:
                Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
            Raises:
                ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
            """
        low, high = proportion_confint(s, n, alpha=ia, method=method)
        return (float(low), 1.0) if alt == 'greater' else (0.0, float(high)) if alt == 'less' else (float(low), float(high))
    s, n = (d['successes'], d['n'])
    estimate = s / n
    if p.method == 'exact':
        stat = estimate
        pv = stats.binomtest(s, n, p.p_0, alternative=alt).pvalue
        interval = ci(s, n, 'beta')
    else:
        stat = (estimate - p.p_0) / np.sqrt(p.p_0 * (1 - p.p_0) / n)
        pv = tail(stat, alt, stats.norm)
        interval = ci(s, n, 'wilson')
        if min(n * p.p_0, n * (1 - p.p_0)) < 5:
            notes.append('작은 기대 성공/실패 수: 정확 이항 검정을 고려하세요.')
    return result('one-sample-proportion', p, stat, pv, '모비율이 p_0와 같다', p.method, dict(proportions=[estimate], difference=estimate - p.p_0, confidence_intervals=[interval], difference_interval=None), {'successes': s, 'n': n, 'p_0': p.p_0}, notes)

def _run_test(data: Dict[str, Any], profile: OneSampleProportionProfile) -> OneSampleProportionOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        OneSampleProportionOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return OneSampleProportionOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_one_sample_proportion(successes: Any, n: Any, profile: Optional[OneSampleProportionProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> OneSampleProportionOutputDTO:
    """one 모비율 검정.

    Args:
        successes: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        n: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 OneSampleProportionProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        OneSampleProportionOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile("one-sample-proportion", profile, recipe_path=recipe_path, overrides=overrides)
    data = _validate_inputs({'successes': successes, 'n': n}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_one_sample_proportion(7, 20)
    print(result.to_json(indent=2))
