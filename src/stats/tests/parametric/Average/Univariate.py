"""
================================================================================
Univariate.py - 단일 표본 평균 검정(One-sample Average Test) 기능 구현
================================================================================
설명:
    - 모집단의 정보가 알려진 경우와 알려지지 않은 경우에 따라 단일 표본 평균 검정을 수행하는 기능을 제공합니다.

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

from dataclasses import dataclass, fields
from src.output import OutputDTO
from typing import Optional
from typing_extensions import (
    Literal,
)  # 3.8 미만에서 사용 (pip install typing_extensions)
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

    mu_0: float  # 귀무가설 모평균 (H0: μ = mu_0)
    known_var: bool = False  # 알려진 모분산(모표준편차) 유무
    sigma: Optional[float] = None  # 알려진 모표준편차 (known_var=True 일 때 지정)
    alpha: float = 0.05  # 유의수준 (기본값 0.05)
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"  # 대립가설 방향


# 2. 검정 결과를 통일된 규격으로 반환하는 Result 객체
@dataclass
class TestResult(OutputDTO):
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


from .one_sample import (
    OneSampleMeanProfile,
    OneSampleMeanOutputDTO,
    run_one_sample_mean,
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
    return TestResult(
        value.test_type,
        value.statistic,
        value.p_value,
        value.reject_h0,
        value.sample_mean,
        value.sample_std,
        value.n_effective,
    )


def _run_z_test(data, profile, sample_mean, sample_std):
    """이전 내부 함수의 호환 호출입니다. 요약 통계는 원 표본에서 재계산합니다.

    Args:
        data, profile, sample_mean, sample_std: 기존 위치 인자; Profile 조건과 원 표본 사용.
    Returns:
        TestResult: 기존 Result 필드 유지, JSON 변환 추가.
    Raises:
        ValueError/TypeError: 잘못된 표본/설정.
    """
    settings = {f.name: getattr(profile, f.name) for f in fields(profile)}
    settings["known_var"] = True
    value = run_one_sample_mean(data, profile=OneSampleMeanProfile(**settings))
    return _convert_result(value)


def _run_t_test(data, profile, sample_mean, sample_std):
    """이전 내부 함수의 호환 호출입니다. 요약 통계는 원 표본에서 재계산합니다.

    Args:
        data, profile, sample_mean, sample_std: 기존 위치 인자; Profile 조건과 원 표본 사용.
    Returns:
        TestResult: 기존 Result 필드 유지, JSON 변환 추가.
    Raises:
        ValueError/TypeError: 잘못된 표본/설정.
    """
    settings = {f.name: getattr(profile, f.name) for f in fields(profile)}
    settings["known_var"] = False
    value = run_one_sample_mean(data, profile=OneSampleMeanProfile(**settings))
    return _convert_result(value)


if __name__ == "__main__":
    print(run_one_sample_mean([1, 2, 4, 5, 8]).to_json(indent=2))
