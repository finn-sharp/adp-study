"""
================================================================================
anova.py - 일원·Welch·이원·반복측정 ANOVA 통합 공개 API
================================================================================
설명:
    - 기존 OneWay/TwoWay/RepeatedANOVA 구현을 재사용하며 long-form 입력을 통일합니다.
    - 새로운 계산식을 중복하지 않고 원래 Result 객체와 ANOVA 표를 반환합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-11
    - 버전: 1.0.0
변경 이력: 2026-10-11 v1.0.0 통합 실행기·입력 검증·가정 명세 추가
사용 예제: python -m examples.multivariate_workflow
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 모든 방식의 명시적 공통 반환 DTO 추가
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ANOVAProfile:
    """long-form ANOVA 설정.

    Attributes:
        response: 수치 반응 열 이름.
        factors: one_way는 한 요인, two_way는 두 요인, repeated는 대상 내 요인 목록.
        method: one_way, two_way, repeated 중 하나.
        equal_var: one_way에서 True=Classic, False=Welch. 다른 방법은 True만 허용.
        interaction: two_way 교호작용 포함 여부.
        ss_type: two_way 제곱합 타입 1/2/3. Type III는 Sum 대비 사용.
        subject: repeated 대상 식별 열 이름. 다른 방법에는 지정하지 않음.
        alpha: 유의수준 (0,1).
    """
    response: str
    factors: List[str] = field(default_factory=list)
    method: str = "one_way"
    equal_var: bool = True
    interaction: bool = True
    ss_type: int = 2
    subject: Optional[str] = None
    alpha: float = 0.05


@dataclass
class ANOVAOutputDTO(OutputDTO):
    """모든 ANOVA 방식의 공통 반환 DTO.

    Attributes:
        method: one_way/two_way/repeated.
        anova_table: 방법별 ANOVA DataFrame.
        analysis: 기존 방법별 Output DTO. statistic/p_value/f_factor1 등은 이 DTO에 정의.
    Notes:
        이전의 result.statistic/result.f_factor1 접근은 analysis로 위임해 호환성을 유지합니다.
    """
    method: str
    anova_table: pd.DataFrame
    analysis: OutputDTO

    def __getattr__(self, name):
        analysis = self.__dict__.get('analysis')
        if analysis is not None and name in analysis.keys():
            return getattr(analysis,name)
        raise AttributeError(name)


def run_anova(data: pd.DataFrame, profile: ANOVAProfile) -> ANOVAOutputDTO:
    """long-form 입력으로 일원/Welch/이원/반복측정 분산분석을 실행합니다.

    Args:
        data: 반응·요인 및 선택적 subject 열을 가진 DataFrame. 행은 독립 관측 또는 명시된 반복 관측.
        profile: 방법/반응/요인/대상 및 검정 조건.
    Returns:
        ANOVAOutputDTO: method, anova_table과 방법별 analysis DTO.
            일원 analysis.group_means 순서는 pd.unique 요인 순서.
    Raises:
        ValueError: 방법/열/요인 수/결측·무한대/표본 부족/잘못된 설정.
        ImportError: statsmodels 미설치.
    Notes:
        독립 ANOVA는 잔차 정규성·등분산을, Welch는 독립성과 집단별 분산 추정을 가정합니다.
        반복측정은 균형 설계만 지원하며 구형성 보정과 집단 간 요인을 제공하지 않습니다.
        검정은 가정을 자동으로 확정하지 않으며 사후검정은 posthoc.py에서 별도 호출합니다.
    """
    p = profile
    if p is None or p.method not in ("one_way", "two_way", "repeated") or not p.factors or not 0 < p.alpha < 1:
        raise ValueError("method/factors/alpha 설정을 확인하세요.")
    columns = [p.response] + list(p.factors) + ([p.subject] if p.subject is not None else [])
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique or len(set(columns)) != len(columns) or not set(columns).issubset(data.columns):
        raise ValueError("중복 없는 필수 열 이름이 필요합니다.")
    if data[columns].isna().any().any() or not np.isfinite(data[p.response].astype(float)).all():
        raise ValueError("결측값을 처리하고 반응을 유한한 수치로 지정하세요.")
    if not isinstance(p.equal_var, bool) or not isinstance(p.interaction, bool) or isinstance(p.ss_type, bool) or p.ss_type not in (1,2,3):
        raise ValueError("equal_var/interaction은 bool, ss_type은 1/2/3이어야 합니다.")
    if p.method != "repeated" and p.subject is not None:
        raise ValueError("subject는 repeated에만 사용합니다.")
    if p.method != "one_way" and not p.equal_var:
        raise ValueError("이원/반복측정의 Welch 변형은 지원하지 않습니다.")
    if p.method == "one_way":
        if len(p.factors) != 1:
            raise ValueError("one_way에는 한 요인이 필요합니다.")
        from src.stats.tests.parametric.Average.Multivariate.OneWay import OneWayANOVAProfile, run_oneway_anova
        groups = [data.loc[data[p.factors[0]] == label, p.response].values for label in pd.unique(data[p.factors[0]])]
        analysis = run_oneway_anova(groups, OneWayANOVAProfile(p.equal_var, p.alpha))
        return ANOVAOutputDTO(p.method, analysis.anova_table, analysis)
    if p.method == "two_way":
        if len(p.factors) != 2:
            raise ValueError("two_way에는 두 요인이 필요합니다.")
        from src.stats.tests.parametric.Average.Multivariate.TwoWay import TwoWayANOVAProfile, run_twoway_anova
        # 입력 열 이름의 공백/문장부호를 patsy 수식으로 전달하지 않음.
        analysis = run_twoway_anova(data[p.response].values, data[p.factors[0]].values,
            data[p.factors[1]].values, TwoWayANOVAProfile(p.interaction, p.ss_type, p.alpha))
        return ANOVAOutputDTO(p.method, analysis.anova_table, analysis)
    if p.subject is None:
        raise ValueError("repeated에는 subject가 필요합니다.")
    from src.stats.repeated_measures import RepeatedANOVAProfile, run_repeated_anova
    analysis = run_repeated_anova(data, RepeatedANOVAProfile(p.response, p.subject, p.factors, p.alpha))
    return ANOVAOutputDTO(p.method, analysis.anova_table, analysis)
