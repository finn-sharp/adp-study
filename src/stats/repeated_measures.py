"""
================================================================================
repeated_measures.py - 균형 반복측정 ANOVA
================================================================================
설명: 통계분석 PDF의 AnovaRM을 동일 대상의 반복 관측용 실행기로 분리합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 반복측정 분산분석 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


@dataclass
class RepeatedANOVAProfile:
    """반복측정 조건.

    Attributes:
        response: 수치 반응 열 이름.
        subject: 대상 식별 열 이름.
        within: 대상 내 요인 열 이름 목록.
        alpha: 유의수준 (0, 1).
    """
    response: str
    subject: str
    within: List[str]
    alpha: float = 0.05


@dataclass
class RepeatedANOVAOutputDTO(OutputDTO):
    """반복측정 결과.

    Attributes:
        model: statsmodels AnovaResults 객체.
        anova_table: 요인별 F/자유도/p-value/기각 여부.
    """
    model: object
    anova_table: pd.DataFrame


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
RepeatedANOVAResult = RepeatedANOVAOutputDTO


def run_repeated_anova(data: pd.DataFrame, profile: RepeatedANOVAProfile) -> RepeatedANOVAOutputDTO:
    """균형 반복측정 ANOVA를 실행합니다.

    Args:
        data: 대상/요인/반응을 가진 long-form DataFrame. 대상-요인 조합당 한 행.
        profile: 반응/대상/대상 내 요인 설정.
    Returns:
        RepeatedANOVAOutputDTO: 분산분석표.
    Raises:
        ValueError: 결측치, 중복 관측, 불균형 설계 또는 잘못된 설정.
        ImportError: statsmodels 미설치.

    Notes:
        AnovaRM의 지원 범위에 따릅니다. 집단 간 요인과 구형성 보정은 제공하지 않습니다.
    """
    from statsmodels.stats.anova import AnovaRM
    p = profile
    columns = [p.response, p.subject] + list(p.within)
    if not p.within or len(columns) != len(set(columns)) or not 0 < p.alpha < 1:
        raise ValueError("고유한 열 이름과 within 요인, 유효한 alpha가 필요합니다.")
    if not set(columns).issubset(data.columns) or data[columns].isna().any().any():
        raise ValueError("필수 열을 확인하고 결측값을 먼저 처리하세요.")
    if not np.isfinite(data[p.response].astype(float)).all() or data.duplicated([p.subject] + p.within).any():
        raise ValueError("반응은 유한한 수치이고 대상-요인 조합은 고유해야 합니다.")
    result = AnovaRM(data, depvar=p.response, subject=p.subject, within=p.within).fit()
    table = result.anova_table.copy()
    table["reject_h0"] = table["Pr > F"] < p.alpha
    return RepeatedANOVAOutputDTO(result, table)
