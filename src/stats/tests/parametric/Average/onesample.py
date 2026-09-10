"""
================================================================================
onesample.py - 파일에 대한 간략한 설명
================================================================================
설명:
    - 이 파일의 주요 기능 및 목적을 작성합니다.

작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)       v1.0.0    단일 표본 평균 검정(One-sample Average Test) 기능 구현
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from dataclasses import dataclass
from typing import Optional
from typing_extensions import Literal # 3.8 미만에서 사용 (pip install typing_extensions)
import numpy as np
from scipy import stats

# 1. 검정 조건 및 모집단 사전 정보를 정의하는 Profile
@dataclass
class TestProfile:
    """검정에 필요한 모집단 사전 정보 및 조건을 정의하는 데이터 클래스.
    
    Attributes:
        mu_0: 귀무가설 모평균 (H0: μ = mu_0)
        known_var: 알려진 모분산(모표준편차) 유무
        sigma: 알려진 모표준편차 (known_var=True 일 때 지정)
    """
    
    mu_0: float                                      # 귀무가설 모평균 (H0: μ = mu_0)
    known_var: bool = False                          # 알려진 모분산(모표준편차) 유무
    sigma: Optional[float] = None                    # 알려진 모표준편차 (known_var=True 일 때 지정)
    alpha: float = 0.05                              # 유의수준 (기본값 0.05)
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'  # 대립가설 방향

# 2. 검정 결과를 통일된 규격으로 반환하는 Result 객체
@dataclass
class TestResult:
    """검정 결과를 통일된 규격으로 반환하는 데이터 클래스.
    
    Attributes:
        test_type: 수행된 검정 유형 (예: "One-sample Z-test", "One-sample T-test")
        statistic: 검정 통계량 값
        p_value: p-value 값
        reject_h0: 귀무가설 기각 여부 (True/False)
        sample_mean: 표본 평균
        sample_std: 표본 표준편차
        n: 표본 크기
    """
    
    test_type: str
    statistic: float
    p_value: float
    reject_h0: bool
    sample_mean: float
    sample_std: float
    n: int

# ---------------------------------------------------------------------------
# 2. Individual Test Runners (개별 검정 수행 전담 함수)
# ---------------------------------------------------------------------------

def _run_z_test(
        data: np.ndarray, 
        profile: TestProfile, 
        sample_mean: float, 
        sample_std: float) -> TestResult:
    """Z-검정을 수행합니다. (모분산을 아는 경우)
    
    Args:
        data: 표본 데이터 (numpy 배열)
        profile: 검정 조건 및 모집단 사전 정보를 담은 TestProfile 객체
        sample_mean: 표본 평균
        sample_std: 표본 표준편차
    
    Returns:
        TestResult: 검정 결과를 담은 TestResult 객체
    
    Raises:
        ValueError: 모분산을 아는 Z-검정을 수행하려면 양수의 sigma(모표준편차)가 지정되어야 합니다.
    """
    
    if profile.sigma is None or profile.sigma <= 0:
        raise ValueError("모분산을 아는 Z-검정을 수행하려면 양수의 sigma(모표준편차)가 지정되어야 합니다.")

    n = len(data)
    se = profile.sigma / np.sqrt(n)
    stat = (sample_mean - profile.mu_0) / se

    # p-value 계산
    if profile.alternative == 'two-sided':
        p_val = stats.norm.sf(np.abs(stat)) * 2
    elif profile.alternative == 'greater':
        p_val = stats.norm.sf(stat)
    else:  # 'less'
        p_val = stats.norm.cdf(stat)

    return TestResult(
        test_type="One-sample Z-test",
        statistic=float(stat),
        p_value=float(p_val),
        reject_h0=bool(p_val < profile.alpha),
        sample_mean=sample_mean,
        sample_std=sample_std,
        n=n
    )


def _run_t_test(
        data: np.ndarray, 
        profile: TestProfile, 
        sample_mean: float, 
        sample_std: float) -> TestResult:
    """ T-검정을 수행합니다. (모분산을 모르는 경우)
    
    Args:
        data: 표본 데이터 (numpy 배열)
        profile: 검정 조건 및 모집단 사전 정보를 담은 TestProfile 객체
        sample_mean: 표본 평균
        sample_std: 표본 표준편차
    
    Returns:
        TestResult: 검정 결과를 담은 TestResult 객체
    
    Raises:
        ValueError: 표본 크기가 1 이하인 경우 T-검정을 수행할 수 없습니다.
    """
    if len(data) <= 1:
        raise ValueError("표본 크기가 1 이하인 경우 T-검정을 수행할 수 없습니다.")

    res = stats.ttest_1samp(
        data, 
        popmean=profile.mu_0, 
        alternative=profile.alternative
    )
    p_val = float(res.pvalue)

    return TestResult(
        test_type="One-sample T-test",
        statistic=float(res.statistic),
        p_value=p_val,
        reject_h0=bool(p_val < profile.alpha),
        sample_mean=sample_mean,
        sample_std=sample_std,
        n=len(data)
    )

if __name__ == "__main__":
    sample_data = np.array([5.1, 5.3, 5.2, 5.4, 5.0])
    profile = TestProfile(mu_0=5.0, known_var=False, alpha=0.05, alternative='two-sided')

    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)

    if profile.known_var:
        result = _run_z_test(sample_data, profile, sample_mean, sample_std)
    else:
        result = _run_t_test(sample_data, profile, sample_mean, sample_std)

    print(f"Test Type: {result.test_type}")
    print(f"Statistic: {result.statistic}")
    print(f"P-value: {result.p_value}")
    print(f"Reject H0: {result.reject_h0}") 
    