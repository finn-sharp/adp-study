"""
================================================================================
Bivariate.py - 두 변수 간의 평균 검정(Bivariate Average Test) 기능 구현
================================================================================
설명:
    - 모집단의 정보와 두 표본의 정보를 기반으로 두 표본 평균 검정을 수행하는 기능을 제공합니다.

작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)       v1.0.0    두 표본 평균 검정(Two-sample Average Test) 기능 구현
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from dataclasses import dataclass
from typing import Optional
from typing_extensions import Literal # 3.8 미만 지원 (pip install typing_extensions)
import numpy as np
from scipy import stats


# 1. Data Models (Profile & Result)
@dataclass
class TwoSampleTestProfile:
    """두 집단 검정에 필요한 모집단 사전 정보 및 조건을 정의하는 데이터 클래스.
    
    Attributes:
        diff_0: 귀무가설 모평균 차이 (H0: μ1 - μ2 = diff_0, 기본값 0.0)
        known_var: 알려진 모분산(모표준편차) 유무
        sigma1: 집단 1의 알려진 모표준편차 (known_var=True 일 때 필수)
        sigma2: 집단 2의 알려진 모표준편차 (known_var=True 일 때 필수)
        equal_var: 모분산을 모를 때 등분산 가정 여부 (True: Student T-test, False: Welch T-test)
        alpha: 유의수준 (기본값 0.05)
        alternative: 대립가설 방향 ('two-sided', 'less', 'greater')
    """
    
    diff_0: float = 0.0                             # 귀무가설 모평균 차이 (H0: μ1 - μ2 = diff_0)
    known_var: bool = False                          # 알려진 모분산 유무
    sigma1: Optional[float] = None                  # 집단 1 알려진 모표준편차
    sigma2: Optional[float] = None                  # 집단 2 알려진 모표준편차
    equal_var: bool = True                          # 등분산 가정 여부 (Student vs Welch)
    alpha: float = 0.05                             # 유의수준
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'  # 대립가설 방향


@dataclass
class TwoSampleTestResult:
    """두 집단 검정 결과를 통일된 규격으로 반환하는 데이터 클래스.
    
    Attributes:
        test_type: 수행된 검정 유형 (예: "Two-sample Z-test", "Welch's T-test")
        statistic: 검정 통계량 값
        p_value: p-value 값
        reject_h0: 귀무가설 기각 여부 (True/False)
        diff_mean: 두 표본의 평균 차이 (mean1 - mean2)
        sample_mean1: 집단 1 표본 평균
        sample_mean2: 집단 2 표본 평균
        sample_std1: 집단 1 표본 표준편차
        sample_std2: 집단 2 표본 표준편차
        n1: 집단 1 표본 크기
        n2: 집단 2 표본 크기
    """
    
    test_type: str
    statistic: float
    p_value: float
    reject_h0: bool
    diff_mean: float
    sample_mean1: float
    sample_mean2: float
    sample_std1: float
    sample_std2: float
    n1: int
    n2: int


# 2. Individual Test Runners (개별 검정 수행 전담 함수)
def _run_two_sample_z_test(
        data1: np.ndarray,
        data2: np.ndarray,
        profile: TwoSampleTestProfile,
        mean1: float,
        mean2: float,
        std1: float,
        std2: float) -> TwoSampleTestResult:
    
    """ 두 집단 Z-검정을 수행합니다 (모분산을 아는 경우).
    
    Args:
        data1: 집단 1 표본 데이터 (numpy 배열)
        data2: 집단 2 표본 데이터 (numpy 배열)
        profile: 검정 조건 및 모집단 사전 정보를 담은 TwoSampleTestProfile 객체
        mean1: 집단 1 표본 평균
        mean2: 집단 2 표본 평균
        std1: 집단 1 표본 표준편차
        std2: 집단 2 표본 표준편차
    
    Returns:
        TwoSampleTestResult: 검정 결과를 담은 TwoSampleTestResult 객체
    
    Raises:
        ValueError: sigma1 또는 sigma2가 지정되지 않았거나 0 이하인 경우 Z-검정을 수행할 수 없습니다.
    """
    
    if profile.sigma1 is None or profile.sigma1 <= 0 or profile.sigma2 is None or profile.sigma2 <= 0:
        raise ValueError("Z-검정을 수행하려면 양수의 sigma1과 sigma2가 모두 지정되어야 합니다.")

    n1, n2 = len(data1), len(data2)
    diff_mean = mean1 - mean2
    
    # 두 집단 Z-검정 표준오차: sqrt(σ1²/n1 + σ2²/n2)
    se = np.sqrt((profile.sigma1 ** 2 / n1) + (profile.sigma2 ** 2 / n2))
    stat = (diff_mean - profile.diff_0) / se

    # p-value 계산
    if profile.alternative == 'two-sided':
        p_val = stats.norm.sf(np.abs(stat)) * 2
    elif profile.alternative == 'greater':
        p_val = stats.norm.sf(stat)
    else:  # 'less'
        p_val = stats.norm.cdf(stat)

    return TwoSampleTestResult(
        test_type="Two-sample Z-test",
        statistic=float(stat),
        p_value=float(p_val),
        reject_h0=bool(p_val < profile.alpha),
        diff_mean=float(diff_mean),
        sample_mean1=mean1,
        sample_mean2=mean2,
        sample_std1=std1,
        sample_std2=std2,
        n1=n1,
        n2=n2
    )


def _run_two_sample_t_test(
        data1: np.ndarray,
        data2: np.ndarray,
        profile: TwoSampleTestProfile,
        mean1: float,
        mean2: float,
        std1: float,
        std2: float) -> TwoSampleTestResult:
    
    """ 두 집단 T-검정을 수행합니다 (모분산을 모르는 경우).
    
    Args:
        data1: 집단 1 표본 데이터 (numpy 배열)
        data2: 집단 2 표본 데이터 (numpy 배열)
        profile: 검정 조건 및 모집단 사전 정보를 담은 TwoSampleTestProfile 객체
        mean1: 집단 1 표본 평균
        mean2: 집단 2 표본 평균
        std1: 집단 1 표본 표준편차
        std2: 집단 2 표본 표준편차
    
    Returns:
        TwoSampleTestResult: 검정 결과를 담은 TwoSampleTestResult 객체
    
    Raises:
        ValueError: 각 집단의 표본 크기가 2 이하인 경우 T-검정을 수행할 수 없습니다.
    """
    
    n1, n2 = len(data1), len(data2)
    if n1 < 2 or n2 < 2:
        raise ValueError("T-검정을 수행하려면 각 집단의 표본 크기가 최소 2 이상이어야 합니다.")

    # diff_0 != 0 인 경우를 대비해 data1에서 차이값을 빼고 scipy ttest_ind 호출
    res = stats.ttest_ind(
        data1 - profile.diff_0,
        data2,
        equal_var=profile.equal_var,
        alternative=profile.alternative
    )

    test_type = "Two-sample T-test (Equal Variance)" if profile.equal_var else "Welch's T-test (Unequal Variance)"

    return TwoSampleTestResult(
        test_type=test_type,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        reject_h0=bool(res.pvalue < profile.alpha),
        diff_mean=float(mean1 - mean2),
        sample_mean1=mean1,
        sample_mean2=mean2,
        sample_std1=std1,
        sample_std2=std2,
        n1=n1,
        n2=n2
    )


if __name__ == "__main__":
    sample1 = np.array([5.1, 5.3, 5.2, 5.4, 5.0])
    sample2 = np.array([5.5, 5.6, 5.7, 5.8, 5.9])
    profile = TwoSampleTestProfile(
        diff_0=0.0,
        known_var=False,
        equal_var=True,
        alpha=0.05,
        alternative='two-sided'
    )

    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    std1 = np.std(sample1, ddof=1)
    std2 = np.std(sample2, ddof=1) 

    if profile.known_var:
        result = _run_two_sample_z_test(sample1, sample2, profile, mean1, mean2, std1, std2)
    else:
        result = _run_two_sample_t_test(sample1, sample2, profile, mean1, mean2, std1, std2)    

    print(f"Test Type: {result.test_type}")
    print(f"Statistic: {result.statistic:.4f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Reject H0: {result.reject_h0}")
    print(f"Mean Difference: {result.diff_mean:.4f}")
    print(f"Sample Mean 1: {result.sample_mean1:.4f}, Sample Std 1: {result.sample_std1:.4f}, n1: {result.n1}")
    print(f"Sample Mean 2: {result.sample_mean2:.4f}, Sample Std 2: {result.sample_std2:.4f}, n2: {result.n2}")
    