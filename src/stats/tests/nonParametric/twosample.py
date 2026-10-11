"""
================================================================================
twosample.py - 독립/대응 두 표본 비모수 검정
================================================================================
설명: Mann-Whitney U와 Wilcoxon 부호순위 검정을 공통 검정 실행기에 연결합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 두 표본 비모수 검정 구현
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 모듈 전용 결과 DTO 및 저장용 변환 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from dataclasses import dataclass
from src.stats.analysis import StatisticalProfile, StatisticalResult, run_statistical_test


@dataclass
class TwoSampleRankProfile:
    """두 표본 검정 조건.

    Attributes:
        paired: True이면 Wilcoxon 대응 검정, False이면 독립 Mann-Whitney U.
        alpha: 유의수준 (0, 1).
        alternative: two-sided, less, greater.
    """
    paired: bool = False
    alpha: float = 0.05
    alternative: str = "two-sided"


@dataclass
class TwoSampleRankOutputDTO(StatisticalResult):
    """TwoSampleRank 모듈 전용 반환 DTO.

    Attributes:
        공통 StatisticalOutputDTO의 검정 유형·통계량·p-value·보조 정보 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def run_two_sample_rank_test(data1, data2, profile=None) -> TwoSampleRankOutputDTO:
    """표본 관계에 따라 비모수 두 표본 검정을 실행합니다.

    Args:
        data1: 첫 번째 유한한 1차원 표본.
        data2: 두 번째 표본. paired=True이면 길이 및 대상 순서가 같아야 합니다.
        profile: TwoSampleRankProfile. None이면 독립 표본 검정.
    Returns:
        TwoSampleRankOutputDTO: 검정통계량, p-value 및 기각 여부.
    Raises:
        ValueError: 유효하지 않은 입력·조건 또는 정의 불가능한 검정.

    Notes:
        Mann-Whitney는 분포의 상대적 위치를 검정합니다. 일반적으로 평균 차이 검정은 아닙니다.
    """
    p = profile if profile is not None else TwoSampleRankProfile()
    return TwoSampleRankOutputDTO.from_result(run_statistical_test(data1, StatisticalProfile(test="wilcoxon" if p.paired else "mannwhitney",
                                 alpha=p.alpha, alternative=p.alternative), other=data2))
