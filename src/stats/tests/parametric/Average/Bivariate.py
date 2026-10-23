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

from dataclasses import dataclass, fields
from src.output import OutputDTO
from typing import Optional
from typing_extensions import Literal  # 3.8 미만 지원 (pip install typing_extensions)
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

    diff_0: float = 0.0  # 귀무가설 모평균 차이 (H0: μ1 - μ2 = diff_0)
    known_var: bool = False  # 알려진 모분산 유무
    sigma1: Optional[float] = None  # 집단 1 알려진 모표준편차
    sigma2: Optional[float] = None  # 집단 2 알려진 모표준편차
    equal_var: bool = True  # 등분산 가정 여부 (Student vs Welch)
    alpha: float = 0.05  # 유의수준
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"  # 대립가설 방향


@dataclass
class TwoSampleTestResult(OutputDTO):
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

from .two_sample import (
    TwoSampleMeanProfile,
    TwoSampleMeanOutputDTO,
    run_two_sample_mean,
)


def _convert_result(value):
    """새 DTO를 기존 Result 필드와 생성자 순서로 변환합니다.

    Args:
        value: 새 평균 검정 DTO.
    Returns:
        OutputDTO: 기존 이름/필드의 Result.
    Raises:
        AttributeError: 잘못된 새 결과.
    """
    info = value.details
    return TwoSampleTestResult(
        value.test_type,
        value.statistic,
        value.p_value,
        value.reject_h0,
        value.mean_difference,
        info["sample_mean1"],
        info["sample_mean2"],
        info["sample_std1"],
        info["sample_std2"],
        value.group_sizes[0],
        value.group_sizes[1],
    )


def _run_two_sample_z_test(data1, data2, profile, mean1, mean2, std1, std2):
    """이전 내부 함수의 호환 호출입니다. 요약 통계는 원 표본에서 재계산합니다.

    Args:
        data1, data2, profile, mean1, mean2, std1, std2: 기존 위치 인자; Profile 조건과 원 표본 사용.
    Returns:
        TwoSampleTestResult: 기존 Result 필드 유지, JSON 변환 추가.
    Raises:
        ValueError/TypeError: 잘못된 표본/설정.
    """
    settings = {f.name: getattr(profile, f.name) for f in fields(profile)}
    settings["known_var"] = True
    value = run_two_sample_mean(data1, data2, profile=TwoSampleMeanProfile(**settings))
    return _convert_result(value)


def _run_two_sample_t_test(data1, data2, profile, mean1, mean2, std1, std2):
    """이전 내부 함수의 호환 호출입니다. 요약 통계는 원 표본에서 재계산합니다.

    Args:
        data1, data2, profile, mean1, mean2, std1, std2: 기존 위치 인자; Profile 조건과 원 표본 사용.
    Returns:
        TwoSampleTestResult: 기존 Result 필드 유지, JSON 변환 추가.
    Raises:
        ValueError/TypeError: 잘못된 표본/설정.
    """
    settings = {f.name: getattr(profile, f.name) for f in fields(profile)}
    settings["known_var"] = False
    value = run_two_sample_mean(data1, data2, profile=TwoSampleMeanProfile(**settings))
    return _convert_result(value)


if __name__ == "__main__":
    print(run_two_sample_mean([1, 2, 4, 5, 8], [2, 3, 4, 6, 9]).to_json(indent=2))
