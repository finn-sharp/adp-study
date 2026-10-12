"""
================================================================================
two_sample - two 모비율 검정
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
class TwoSampleProportionProfile:
    """TwoSampleProportionProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        alternative (str): two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided".
    """
    alpha: float = 0.05
    alternative: str = "two-sided"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class TwoSampleProportionOutputDTO(OutputDTO):
    """TwoSampleProportionOutputDTO 전체 필드 계약.

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
    pairs = [('successes1', 'n1'), ('successes2', 'n2')]
    d = {}
    for s, n in pairs:
        d[s] = integer(args[s], s)
        d[n] = integer(args[n], n, 1)
        if d[s] > d[n]:
            raise ValueError('성공 수는 시행 수 이하여야 합니다.')
    return d

def _validate_inputs(arguments: Dict[str, Any], profile: TwoSampleProportionProfile) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: TwoSampleProportionProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, TwoSampleProportionProfile): raise TypeError("TwoSampleProportionProfile이 필요합니다.")
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
    s1, n1, s2, n2 = (d['successes1'], d['n1'], d['successes2'], d['n2'])
    a = s1 / n1
    b = s2 / n2
    pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se <= 0:
        raise ValueError('pooled 성공확률 0/1은 Z 검정 미정의입니다.')
    stat = (a - b) / se
    c1 = ci(s1, n1, 'wilson')
    c2 = ci(s2, n2, 'wilson')
    raw1 = proportion_confint(s1, n1, alpha=ia, method='wilson')
    raw2 = proportion_confint(s2, n2, alpha=ia, method='wilson')
    low = a - b - np.sqrt((a - raw1[0]) ** 2 + (raw2[1] - b) ** 2)
    high = a - b + np.sqrt((raw1[1] - a) ** 2 + (b - raw2[0]) ** 2)
    diffci = (float(low), 1.0) if alt == 'greater' else (-1.0, float(high)) if alt == 'less' else (float(low), float(high))
    if min(n1 * pool, n1 * (1 - pool), n2 * pool, n2 * (1 - pool)) < 5:
        notes.append('정규 근사 조건의 기대도수가 작습니다.')
    return result('two-sample-proportion', p, stat, tail(stat, alt, stats.norm), '두 독립 모비율이 같다', 'pooled-z', dict(proportions=[a, b], difference=a - b, confidence_intervals=[c1, c2], difference_interval=diffci), {'successes': [s1, s2], 'n': [n1, n2], 'pooled_proportion': pool}, notes)

def _run_test(data: Dict[str, Any], profile: TwoSampleProportionProfile) -> TwoSampleProportionOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        TwoSampleProportionOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return TwoSampleProportionOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_two_sample_proportion(successes1: Any, n1: Any, successes2: Any, n2: Any, profile: Optional[TwoSampleProportionProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> TwoSampleProportionOutputDTO:
    """two 모비율 검정.

    Args:
        successes1: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        n1: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        successes2: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        n2: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 TwoSampleProportionProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        TwoSampleProportionOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile("two-sample-proportion", profile, recipe_path=recipe_path, overrides=overrides)
    data = _validate_inputs({'successes1': successes1, 'n1': n1, 'successes2': successes2, 'n2': n2}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_two_sample_proportion(7, 20, 14, 25)
    print(result.to_json(indent=2))
