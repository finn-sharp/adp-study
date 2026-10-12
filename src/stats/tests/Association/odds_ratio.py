"""
================================================================================
odds_ratio - 독립 2×2 odds ratio
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 행=노출/비노출, 열=사건/비사건. RR은 cohort 위험 설계; 대응표본과 구분.
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
from src.stats.tests._common import (number, table_data, tail, mean_interval, result)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class OddsRatioProfile:
    """OddsRatioProfile 전체 필드 계약.

    Attributes:
        alpha (float): 유한한 0<alpha<1 유의수준 기본 0.05.
        zero_correction (str): none/haldane; 0셀 존재 시만 전체 셀에 0.5 더함 기본 "none".
        alternative (str): two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided".
    """
    alpha: float = 0.05
    zero_correction: str = "none"
    alternative: str = "two-sided"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class OddsRatioOutputDTO(OutputDTO):
    """OddsRatioOutputDTO 전체 필드 계약.

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
        estimate (float): OR 또는 RR.
        log_estimate (float): log 효과크기.
        standard_error (float): log 추정량 SE.
        confidence_interval (Tuple[float, float]): 방향 일치 log-Wald 구간을 원 scale로 변환.
        table_used (np.ndarray): 실제 계산 도수; 보정 여부 확인.
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
    estimate: float
    log_estimate: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    table_used: np.ndarray


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
    a = table_data(args['table'], False)
    if a.shape != (2, 2) or p.zero_correction not in ('none', 'haldane'):
        raise ValueError('2×2와 zero_correction 필요')
    if (a == 0).any() and p.zero_correction == 'none':
        raise ValueError('0셀의 log-Wald 추론은 정의되지 않습니다. haldane을 명시하세요.')
    return {'table': a}

def _validate_inputs(arguments: Dict[str, Any], profile: OddsRatioProfile) -> Dict[str, Any]:
    """자료와 설정을 검증합니다.

    Args:
        arguments: 공개 입력 인자의 사전.
        profile: OddsRatioProfile.
    Returns:
        Dict[str, Any]: 검증된 수치·설계 자료.
    Raises:
        TypeError/ValueError: 잘못된 Profile·표본·설정·가정 조건.
    """
    if not isinstance(profile, OddsRatioProfile): raise TypeError("OddsRatioProfile이 필요합니다.")
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
    t = d['table'].astype(float)
    corrected = (t == 0).any()
    if corrected:
        t = t + 0.5
    a, b, c, e = t.ravel()
    estimate = a * e / (b * c)
    se = np.sqrt((1 / t).sum())
    if se <= 0:
        raise ValueError('양의 효과크기 표준오차 필요')
    log = float(np.log(estimate))
    ci = mean_interval(log, se, p.alpha, alt, stats.norm)
    bounds = tuple((float(np.exp(v)) for v in ci))
    return result('odds-ratio', p, log / se, tail(log / se, alt, stats.norm), 'OR/RR=1 (독립 집단의 연관/위험 동일)', 'log-wald', dict(estimate=float(estimate), log_estimate=log, standard_error=float(se), confidence_interval=bounds, table_used=t), {'zero_correction_applied': bool(corrected), 'row_order': ['exposed', 'unexposed'], 'column_order': ['event', 'non_event']}, ['0셀 존재로 전체 표에 0.5 Haldane 보정을 적용했습니다.'] if corrected else [])

def _run_test(data: Dict[str, Any], profile: OddsRatioProfile) -> OddsRatioOutputDTO:
    """검증된 자료를 계산합니다.

    Args:
        data: 검증 자료.
        profile: 기법 설정.
    Returns:
        OddsRatioOutputDTO: 통계량·판단·기법별 결과.
    Raises:
        ValueError: 퇴화 자료·미정의 통계량·정확 계산 한도.
    """
    return OddsRatioOutputDTO(**_calculate(data, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_odds_ratio(table: Any, profile: Optional[OddsRatioProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> OddsRatioOutputDTO:
    """독립 2×2 odds ratio.

    Args:
        table: 기법별 입력 자료. 자세한 설계·shape·H0는 docs/STATISTICS_COMPLETE.md 참고.
        profile: 명시적 OddsRatioProfile 또는 None(기법별 JSON 사용).
        recipe_path: 사용자 JSON 경로.
        overrides: JSON 위에 적용할 일회성 조건.
    Returns:
        OddsRatioOutputDTO: 정의된 모든 결과 필드; 기본 JSON에서 runtime 제외.
    Raises:
        TypeError/ValueError: 입력/설정/설계/계산 조건 위반.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile("odds-ratio", profile, recipe_path=recipe_path, overrides=overrides)
    data = _validate_inputs({'table': table}, profile)
    return _run_test(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_odds_ratio([[12,28],[8,42]])
    print(result.to_json(indent=2))
