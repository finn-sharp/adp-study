"""
================================================================================
OneWay.py - 세 집단 이상의 평균 검정(One-way ANOVA) 기능 구현
================================================================================
설명:
    - 세 집단 이상의 평균 검정을 수행하는 기능을 제공합니다.
    - 고전적 ANOVA 및 Welch ANOVA 수행 시 ANOVA 분산분석표(pandas DataFrame)를 생성합니다.

작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력:
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)       v1.0.0    최초 작성
    2026-09-10  김재현(Finn)       v1.1.0    ANOVA 분산분석표(anova_table) 출력 기능 추가
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.1.0"
__status__ = "Development"

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from scipy import stats
from typing_extensions import Literal  # pip install typing_extensions


# 1. Data Models (Profile & Result)
@dataclass
class OneWayANOVAProfile:
    """세 집단 이상의 평균 검정에 필요한 모집단 사전 정보 및 조건을 정의하는 데이터 클래스.
    
    Attributes:
        equal_var: 등분산 가정 여부 (True: Classic ANOVA, False: Welch's ANOVA)
        alpha: 유의수준 (기본값 0.05)
    """
    equal_var: bool = True  # 등분산 가정 여부 (True: Classic ANOVA, False: Welch's ANOVA)
    alpha: float = 0.05     # 유의수준


@dataclass
class OneWayANOVAResult:
    """세 집단 이상의 평균 검정 결과를 통일된 규격으로 반환하는 데이터 클래스.
    
    Attributes:
        test_type: 수행된 검정 유형 (예: "One-way ANOVA (Equal Variance)", "Welch's One-way ANOVA")
        statistic: 검정 통계량 값 (F 값)
        p_value: p-value 값
        reject_h0: 귀무가설 기각 여부 (True/False)
        num_groups: 집단 수
        group_means: 각 집단의 평균 리스트
        group_stds: 각 집단의 표준편차 리스트
        group_ns: 각 집단의 표본 크기 리스트
        df_between: 집단 간 자유도 (df1)
        df_within: 집단 내 자유도 (df2)
        anova_table: 전체 ANOVA 분산분석표 (pandas DataFrame)
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


# 2. Individual Test Runners (개별 검정 수행 전담 함수)
def _run_classic_anova(
        groups: List[np.ndarray],
        profile: OneWayANOVAProfile,
        means: List[float],
        stds: List[float],
        ns: List[int]) -> OneWayANOVAResult:
    """ 고전적 일원배치 분산분석 및 요약 테이블을 생성합니다 (등분산 가정: Classic Fisher ANOVA).
    
    Args:
        groups: 각 집단의 데이터 리스트/배열 모음
        profile: 검정 조건을 정의한 OneWayANOVAProfile 객체
        means: 각 집단의 평균 리스트
        stds: 각 집단의 표준편차 리스트
        ns: 각 집단의 표본 크기 리스트
    
    Returns:
        OneWayANOVAResult: 검정 결과 및 ANOVA 테이블을 통일된 규격으로 담은 객체
    
    Raises:
        ValueError: 집단 수가 2개 미만이거나, 특정 집단의 표본 크기가 2 미만인 경우. 
    """
    
    if len(groups) < 2:
        raise ValueError("ANOVA를 수행하려면 최소 2개 이상의 집단이 필요합니다.")
    
    k = len(groups)
    ns_arr = np.array(ns)
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    
    # 전체 데이터 및 전체 평균
    all_data = np.concatenate(groups)
    grand_mean = float(np.mean(all_data))
    total_n = len(all_data)

    # 제곱합(SS) 산출
    ss_between = float(np.sum(ns_arr * ((means_arr - grand_mean) ** 2)))
    ss_within = float(np.sum((ns_arr - 1) * (stds_arr ** 2)))
    ss_total = ss_between + ss_within

    # 자유도(df) 산출
    df1 = float(k - 1)
    df2 = float(total_n - k)
    df_total = float(total_n - 1)

    # 평균제곱(MS) 및 F 통계량 산출
    ms_between = ss_between / df1
    ms_within = ss_within / df2
    stat = ms_between / ms_within
    p_val = float(stats.f.sf(stat, df1, df2))

    # ANOVA 분산분석표(DataFrame) 구성
    anova_table = pd.DataFrame({
        'sum_sq': [ss_between, ss_within, ss_total],
        'df': [df1, df2, df_total],
        'mean_sq': [ms_between, ms_within, np.nan],
        'F': [stat, np.nan, np.nan],
        'PR(>F)': [p_val, np.nan, np.nan]
    }, index=['Between Groups', 'Within Groups', 'Total'])

    return OneWayANOVAResult(
        test_type="One-way ANOVA (Equal Variance)",
        statistic=float(stat),
        p_value=p_val,
        reject_h0=bool(p_val < profile.alpha),
        num_groups=k,
        group_means=means,
        group_stds=stds,
        group_ns=ns,
        df_between=df1,
        df_within=df2,
        anova_table=anova_table
    )


def _run_welch_anova(
        groups: List[np.ndarray],
        profile: OneWayANOVAProfile,
        means: List[float],
        stds: List[float],
        ns: List[int]) -> OneWayANOVAResult:
    """Welch의 일원배치 분산분석 및 요약 테이블을 생성합니다 (이분산 허용: Welch's ANOVA).
    
    Args:
        groups: 각 집단의 데이터 리스트/배열 모음
        profile: 검정 조건을 정의한 OneWayANOVAProfile 객체
        means: 각 집단의 평균 리스트
        stds: 각 집단의 표준편차 리스트
        ns: 각 집단의 표본 크기 리스트
    
    Returns:
        OneWayANOVAResult: 검정 결과 및 ANOVA 테이블을 통일된 규격으로 담은 객체
    
    Raises:
        ValueError: 집단 수가 2개 미만이거나, 특정 집단의 표본 크기가 2 미만인 경우.
        ValueError: 특정 집단의 표본 표준편차가 0이거나, 다른 유효성 검증 조건을 만족하지 않는 경우.
    """

    if len(groups) < 2:
        raise ValueError("Welch ANOVA를 수행하려면 최소 2개 이상의 집단이 필요합니다.")

    
    k = len(groups)
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ns_arr = np.array(ns)

    if np.any(stds_arr == 0):
        raise ValueError("Welch ANOVA를 수행하려면 모든 집단의 표본 표준편차가 0보다 커야 합니다.")

    # Welch ANOVA 가중치 산출: w_i = n_i / s_i^2
    weights = ns_arr / (stds_arr ** 2)
    total_weight = np.sum(weights)

    # 가중평균 산출
    weighted_mean = np.sum(weights * means_arr) / total_weight

    df1 = float(k - 1)
    f_num = np.sum(weights * ((means_arr - weighted_mean) ** 2)) / df1

    # 보정 인자 (Lambda) 계산
    lambda_term = (3 / (k ** 2 - 1)) * np.sum((1 / (ns_arr - 1)) * ((1 - (weights / total_weight)) ** 2))

    stat = float(f_num / (1 + (2 * lambda_term * (k - 2) / 3)))
    df2 = float(1 / lambda_term)

    p_val = float(stats.f.sf(stat, df1, df2))

    # Welch ANOVA 요약표(DataFrame) 구성
    anova_table = pd.DataFrame({
        'F': [stat],
        'df1 (Between)': [df1],
        'df2 (Within)': [df2],
        'PR(>F)': [p_val]
    }, index=['Between Groups (Welch)'])

    return OneWayANOVAResult(
        test_type="Welch's One-way ANOVA (Unequal Variance)",
        statistic=stat,
        p_value=p_val,
        reject_h0=bool(p_val < profile.alpha),
        num_groups=k,
        group_means=means,
        group_stds=stds,
        group_ns=ns,
        df_between=df1,
        df_within=df2,
        anova_table=anova_table
    )


# 3. Main Dispatcher (유효성 검증 및 분기 담당 메인 함수)
def run_oneway_anova(
        groups: Sequence[Union[List[float], np.ndarray]], 
        profile: OneWayANOVAProfile) -> OneWayANOVAResult:
    """세 집단 이상의 평균 차이 검정(One-way ANOVA)을 유효성 검증 후 적절한 알고리즘으로 수행합니다.
    
    Args:
        groups: 각 집단의 데이터 리스트/배열 모음 (최소 2개 이상의 집단 시퀀스).
        profile: 검정 조건을 정의한 OneWayANOVAProfile 객체.
    
    Returns:
        OneWayANOVAResult: 검정 결과 및 ANOVA 테이블을 통일된 규격으로 담은 객체.
    
    Raises:
        ValueError: 집단 수가 2개 미만이거나, 특정 집단의 표본 크기가 2 미만인 경우.
        ValueError: profile이 None이거나, profile의 alpha가 0보다 작거나 1보다 큰 경우.
    """
    if len(groups) < 2:
        raise ValueError("ANOVA를 수행하려면 최소 2개 이상의 집단이 필요합니다.")

    formatted_groups: List[np.ndarray] = []
    means: List[float] = []
    stds: List[float] = []
    ns: List[int] = []

    for idx, g in enumerate(groups):
        arr = np.asarray(g, dtype=float)
        n = len(arr)
        if n < 2:
            raise ValueError(f"{idx + 1}번째 집단의 표본 크기가 최소 2 이상이어야 합니다. (현재 크기: {n})")

        formatted_groups.append(arr)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr, ddof=1)))
        ns.append(n)

    # 책임 분기: 등분산성 가정 조건에 따라 알맞은 ANOVA 실행기로 위임
    if profile.equal_var:
        return _run_classic_anova(formatted_groups, profile, means, stds, ns)
    else:
        return _run_welch_anova(formatted_groups, profile, means, stds, ns)


# ---------------------------------------------------------------------------
# 4. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 세 집단 예시 데이터 생성
    group_a = [12.1, 13.4, 11.8, 12.9, 13.0]
    group_b = [14.2, 15.1, 14.8, 13.9, 15.0]
    group_c = [10.5, 11.2, 10.9, 11.5, 10.8]

    groups_data = [group_a, group_b, group_c]

    # 1. 등분산성을 가정한 고전적 ANOVA (Classic Fisher ANOVA)
    profile_classic = OneWayANOVAProfile(equal_var=True, alpha=0.05)
    res_classic = run_oneway_anova(groups_data, profile_classic)

    print("=== 1. Classic One-way ANOVA ===")
    print(f"Test Type  : {res_classic.test_type}")
    print(f"F Statistic: {res_classic.statistic:.4f}")
    print(f"p-value    : {res_classic.p_value:.4e}")
    print(f"Reject H0  : {res_classic.reject_h0}\n")
    print("[ANOVA Table]")
    print(res_classic.anova_table)
    print("\n" + "=" * 80 + "\n")

    # 2. 이분산성을 허용한 Welch's ANOVA
    profile_welch = OneWayANOVAProfile(equal_var=False, alpha=0.05)
    res_welch = run_oneway_anova(groups_data, profile_welch)

    print("=== 2. Welch's One-way ANOVA ===")
    print(f"Test Type  : {res_welch.test_type}")
    print(f"F Statistic: {res_welch.statistic:.4f}")
    print(f"p-value    : {res_welch.p_value:.4e}")
    print(f"Reject H0  : {res_welch.reject_h0}\n")
    print("[ANOVA Table]")
    print(res_welch.anova_table)