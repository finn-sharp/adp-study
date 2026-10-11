"""
================================================================================
OneWay.py - 일원배치 ANOVA·Welch ANOVA
================================================================================
설명:
    - 일원배치 ANOVA·Welch ANOVA을 수행하고 Profile·Output DTO로 입력과 결과를 정의합니다.

분석 가정:
    - 관측은 독립입니다. 고전 ANOVA는 집단 내 정규성·등분산, Welch는 정규성 및 양의 집단 분산을 가정합니다.
    - 결측값·무한대는 사전에 처리하며 가정 검정은 자동 수행하지 않습니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)     v1.0.0    최초 작성
    2026-10-11  김재현(Finn)     v1.1.0    통계 명세·검증·Output DTO 통일 및 검정 구현
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.1.0"
__status__ = "Development"




# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.output import OutputDTO
from ._common import validate_alpha, validate_groups, validate_labels
from statsmodels.stats.oneway import anova_oneway


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class OneWayANOVAProfile:
    """일원배치 ANOVA·Welch ANOVA 입력 조건.

    Attributes:
        equal_var (bool): 등분산 가정 여부. 기본 True: 고전 ANOVA, False: Welch ANOVA.
        alpha (float): 유의수준. 기본 0.05, 유한한 0<값<1. 기각 기준은 p_value<alpha.
    """

    equal_var: bool = True
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class OneWayANOVAOutputDTO(OutputDTO):
    """일원배치 ANOVA·Welch ANOVA 검정 결과.

    Attributes:
        test_type (str): 실행한 고전 또는 Welch 검정 이름.
        statistic (float): F 검정 통계량.
        p_value (float): 전체 집단 평균 동일성의 p-value.
        reject_h0 (bool): p_value<alpha 여부. 어느 집단 쌍인지는 사후검정으로 확인.
        num_groups (int): 집단 수. 최소 2.
        group_means (List[float]): 입력 집단 순서의 평균.
        group_stds (List[float]): 입력 집단 순서의 표본 표준편차, ddof=1.
        group_ns (List[int]): 입력 집단 순서의 표본 수.
        df_between (float): 분자 자유도. k-1.
        df_within (float): 분모 자유도. Welch에서는 Satterthwaite 근사.
        anova_table (pd.DataFrame): 고전은 SS·df·MS·F·p 표, Welch는 근사 F·df·p만 제공. Welch의 SS·MS는 NaN.
    """

    test_type: str
    statistic: float
    p_value: float
    reject_h0: bool
    num_groups: int
    group_means: List[float]
    group_stds: List[float]
    group_ns: List[int]
    df_between: float
    df_within: float
    anova_table: pd.DataFrame

# 기존 공개 결과명 호환 별칭. 전체 필드는 위 DTO에 명시합니다.
OneWayANOVAResult = OneWayANOVAOutputDTO


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate_inputs(groups: Any, profile: OneWayANOVAProfile) -> List[np.ndarray]:
    """일원배치 검정 조건과 집단을 검증합니다.

    Args:
        groups: 최소 2개 집단의 1차원 수치 배열 목록. 각 집단 최소 2행.
        profile: OneWayANOVAProfile.
    Returns:
        List[np.ndarray]: 입력 순서의 유한한 float 집단 배열.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: alpha·등분산 설정·입력·분산 오류.
    """
    if not isinstance(profile, OneWayANOVAProfile):
        raise TypeError("OneWayANOVAProfile이 필요합니다.")
    validate_alpha(profile.alpha)
    if not isinstance(profile.equal_var, bool):
        raise ValueError("equal_var는 bool이어야 합니다.")
    arrays = validate_groups(groups)
    variances = np.array([g.var(ddof=1) for g in arrays])
    if (not profile.equal_var and np.any(variances <= 0)) or variances.sum() <= 0:
        raise ValueError("고전 ANOVA는 양의 잔차 분산, Welch는 모든 집단의 양의 분산이 필요합니다.")
    return arrays


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_anova(groups: List[np.ndarray], profile: OneWayANOVAProfile) -> OneWayANOVAOutputDTO:
    """고전 또는 Welch ANOVA를 계산합니다.

    Args:
        groups: 검증된 집단 배열 목록.
        profile: 검증된 검정 조건.
    Returns:
        OneWayANOVAOutputDTO: F·p·자유도·집단 요약·검정표.
    Raises:
        ValueError: 수치 계산이 비유한 결과를 반환한 경우.
    """
    means = np.array([g.mean() for g in groups])
    ns = np.array([len(g) for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    if profile.equal_var:
        df1, df2 = float(len(groups)-1), float(ns.sum()-len(groups))
        grand = np.average(means, weights=ns)
        ssb = float(np.sum(ns*(means-grand)**2))
        ssw = float(np.sum((ns-1)*variances))
        statistic = (ssb/df1)/(ssw/df2)
        p = float(stats.f.sf(statistic, df1, df2))
        table = pd.DataFrame({'sum_sq':[ssb,ssw,ssb+ssw], 'df':[df1,df2,float(ns.sum()-1)],
                              'mean_sq':[ssb/df1,ssw/df2,np.nan], 'F':[statistic,np.nan,np.nan],
                              'PR(>F)':[p,np.nan,np.nan]}, index=['Between Groups','Within Groups','Total'])
        name = "One-way ANOVA (Equal Variance)"
    else:
        result = anova_oneway(groups, use_var="unequal", welch_correction=True)
        statistic, p = float(result.statistic), float(result.pvalue)
        df1, df2 = map(float, result.df)
        table = pd.DataFrame({'sum_sq':[np.nan,np.nan], 'df':[df1,df2],
                              'mean_sq':[np.nan,np.nan], 'F':[statistic,np.nan],
                              'PR(>F)':[p,np.nan]}, index=['Welch Between Groups','Approximate Residual'])
        name = "Welch's One-way ANOVA (Unequal Variance)"
    if not np.isfinite([statistic,p,df1,df2]).all():
        raise ValueError("ANOVA 수치 계산 결과가 유한하지 않습니다.")
    return OneWayANOVAOutputDTO(name,float(statistic),p,bool(p<profile.alpha),len(groups),
                                means.tolist(),np.sqrt(variances).tolist(),ns.tolist(),df1,df2,table)


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_oneway_anova(groups: Any, profile: Optional[OneWayANOVAProfile] = None) -> OneWayANOVAOutputDTO:
    """집단 평균의 전체 동일성을 검정합니다.

    Args:
        groups: 입력 집단 순서의 수치형 1차원 배열 목록.
        profile: 조건. None이면 고전 ANOVA, alpha=0.05.
    Returns:
        OneWayANOVAOutputDTO: 검정·요약·검정표 및 to_dict()/to_json() 지원.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·분산 또는 수치 계산 오류.
    Notes:
        기존 OneWayANOVAResult 이름은 같은 DTO의 호환 별칭입니다.
        사후검정은 자동 실행하지 않으며 별도 run_posthoc을 호출합니다.
    """
    profile = profile if profile is not None else OneWayANOVAProfile()
    return _run_anova(_validate_inputs(groups, profile), profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    groups = [[1,2,3,4], [4,5,6,8], [8,9,10,12]]
    for equal_var in (True, False):
        result = run_oneway_anova(groups, OneWayANOVAProfile(equal_var=equal_var))
        print(result.test_type, result.p_value)
        print(result.anova_table)
        print(result.to_json()[:200])
