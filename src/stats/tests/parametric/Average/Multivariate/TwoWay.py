"""
================================================================================
TwoWay.py - 두 요인에 대한 평균 검정(Two-way ANOVA) 기능 구현
================================================================================
설명:
    - 두 개의 범주형 독립변수(요인)에 따른 수치형 반응변수의 평균 차이를 검정합니다.
    - 교호작용항(Interaction Effect) 포함 여부 및 제곱합 타입(Type I, II, III) 선택 기능을 제공합니다.

작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력:
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)       v1.0.0    최초 작성
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing_extensions import Literal  # pip install typing_extensions



# 1. Data Models (Profile & Result)
@dataclass
class TwoWayANOVAProfile:
    """ 이원배치 분산분석(Two-way ANOVA) 사전 정보 및 검정 조건을 정의하는 데이터 클래스.
    
    Attributes:
        interaction: 교호작용항(Factor1 x Factor2) 포함 여부 (기본값 True)
        ss_type: 제곱합(Sum of Squares) 계산 방식 (Type 1, 2, 3 / 기본값 2)
        alpha: 유의수준 (기본값 0.05)
    """
    
    interaction: bool = True               # 교호작용항 포함 여부 (True: 주효과+교호작용, False: 주효과 전용)
    ss_type: Literal[1, 2, 3] = 2          # 제곱합 계산 방식 (Type II SS가 주효과 검정의 기본 표준)
    alpha: float = 0.05                    # 유의수준


@dataclass
class TwoWayANOVAResult:
    """ Two-way ANOVA 검정 결과를 통일된 규격으로 반환하는 데이터 클래스.
    
    Attributes:
        test_type: 수행된 검정 유형 (예: "Two-way ANOVA (With Interaction)", "Two-way ANOVA (Main Effects Only)")
        factor1_name: 요인 1 변수명
        factor2_name: 요인 2 변수명
        f_factor1: 요인 1 F 통계량
        p_factor1: 요인 1 p-value
        reject_h0_factor1: 요인 1 귀무가설 기각 여부
        f_factor2: 요인 2 F 통계량
        p_factor2: 요인 2 p-value
        reject_h0_factor2: 요인 2 귀무가설 기각 여부
        f_interaction: 교호작용항 F 통계량 (interaction=False 시 None)
        p_interaction: 교호작용항 p-value (interaction=False 시 None)
        reject_h0_interaction: 교호작용항 귀무가설 기각 여부 (interaction=False 시 None)
        anova_table: 전체 ANOVA 분산분석표 (pandas DataFrame)
    """
    
    test_type: str
    factor1_name: str
    factor2_name: str
    f_factor1: float
    p_factor1: float
    reject_h0_factor1: bool
    f_factor2: float
    p_factor2: float
    reject_h0_factor2: bool
    f_interaction: Optional[float]
    p_interaction: Optional[float]
    reject_h0_interaction: Optional[bool]
    anova_table: pd.DataFrame


# 2. Individual Test Runners (개별 검정 수행 전담 함수)
def _run_twoway_with_interaction(
        df: pd.DataFrame,
        profile: TwoWayANOVAProfile,
        f1_name: str,
        f2_name: str,
        response_name: str) -> TwoWayANOVAResult:

    """ 교호작용항을 포함한 이원배치 분산분석(Full Factorial Model)을 수행합니다.
    
    Args:
        df: pandas DataFrame 형태의 데이터 (반응변수와 두 요인 포함)
        profile: 검정 조건을 정의한 TwoWayANOVAProfile 객체
        f1_name: 요인 1 변수명
        f2_name: 요인 2 변수명
        response_name: 반응 변수의 변수명
    
    Returns:
        TwoWayANOVAResult: 검정 결과를 통일된 규격으로 담은 객체.
    
    Raises:
        ValueError: 입력 데이터가 충분하지 않거나, 교호작용항이 포함되지 않은 경우.
    """

    if profile is None:
        raise ValueError("검정 조건을 정의한 TwoWayANOVAProfile 객체가 필요합니다.")

    if not profile.interaction:
        raise ValueError("교호작용항이 포함되지 않은 프로필에서는 이 함수를 사용할 수 없습니다. interaction=True로 설정하세요.")
    
    formula = f'Q("{response_name}") ~ C(Q("{f1_name}")) + C(Q("{f2_name}")) + C(Q("{f1_name}")):C(Q("{f2_name}"))'
    model = ols(formula, data=df).fit()
    anova_df = sm.stats.anova_lm(model, typ=profile.ss_type)

    # ANOVA 표 결과 추출 (Factor1, Factor2, Interaction)
    f1_f = float(anova_df.iloc[0]['F'])
    f1_p = float(anova_df.iloc[0]['PR(>F)'])

    f2_f = float(anova_df.iloc[1]['F'])
    f2_p = float(anova_df.iloc[1]['PR(>F)'])

    inter_f = float(anova_df.iloc[2]['F'])
    inter_p = float(anova_df.iloc[2]['PR(>F)'])

    return TwoWayANOVAResult(
        test_type=f"Two-way ANOVA with Interaction (Type {profile.ss_type} SS)",
        factor1_name=f1_name,
        factor2_name=f2_name,
        f_factor1=f1_f,
        p_factor1=f1_p,
        reject_h0_factor1=bool(f1_p < profile.alpha),
        f_factor2=f2_f,
        p_factor2=f2_p,
        reject_h0_factor2=bool(f2_p < profile.alpha),
        f_interaction=inter_f,
        p_interaction=inter_p,
        reject_h0_interaction=bool(inter_p < profile.alpha),
        anova_table=anova_df
    )


def _run_twoway_without_interaction(
        df: pd.DataFrame,
        profile: TwoWayANOVAProfile,
        f1_name: str,
        f2_name: str,
        response_name: str) -> TwoWayANOVAResult:
    """ 교호작용항을 제외한 이원배치 분산분석(Main Effects Only Model)을 수행합니다.
    
    Args:
        df: pandas DataFrame 형태의 데이터 (반응변수와 두 요인 포함)
        profile: 검정 조건을 정의한 TwoWayANOVAProfile 객체
        f1_name: 요인 1 변수명
        f2_name: 요인 2 변수명
        response_name: 반응 변수의 변수명
    
    Returns:
        TwoWayANOVAResult: 검정 결과를 통일된 규격으로 담은 객체.
    
    Raises:
        ValueError: 입력 데이터가 충분하지 않거나, 교호작용항이 포함된 경우.
    """
    if profile is None:
        raise ValueError("검정 조건을 정의한 TwoWayANOVAProfile 객체가 필요합니다.")
    
    if profile.interaction:
        raise ValueError("교호작용항이 포함된 프로필에서는 이 함수를 사용할 수 없습니다. interaction=False로 설정하세요.")
    
    formula = f'Q("{response_name}") ~ C(Q("{f1_name}")) + C(Q("{f2_name}"))'
    model = ols(formula, data=df).fit()
    anova_df = sm.stats.anova_lm(model, typ=profile.ss_type)

    # ANOVA 표 결과 추출 (Factor1, Factor2)
    f1_f = float(anova_df.iloc[0]['F'])
    f1_p = float(anova_df.iloc[0]['PR(>F)'])

    f2_f = float(anova_df.iloc[1]['F'])
    f2_p = float(anova_df.iloc[1]['PR(>F)'])

    return TwoWayANOVAResult(
        test_type=f"Two-way ANOVA Main Effects Only (Type {profile.ss_type} SS)",
        factor1_name=f1_name,
        factor2_name=f2_name,
        f_factor1=f1_f,
        p_factor1=f1_p,
        reject_h0_factor1=bool(f1_p < profile.alpha),
        f_factor2=f2_f,
        p_factor2=f2_p,
        reject_h0_factor2=bool(f2_p < profile.alpha),
        f_interaction=None,
        p_interaction=None,
        reject_h0_interaction=None,
        anova_table=anova_df
    )


# 3. Main Dispatcher (유효성 검증 및 분기 담당 메인 함수)
def run_twoway_anova(
        y: Sequence[Union[float, int]],
        factor1: Sequence[Any],
        factor2: Sequence[Any],
        profile: TwoWayANOVAProfile,
        factor1_name: str = "Factor1",
        factor2_name: str = "Factor2",
        response_name: str = "Response") -> TwoWayANOVAResult:
    """ 두 요인에 대한 평균 차이 및 교호작용 검정(Two-way ANOVA)을 유효성 검증 후 적절한 알고리즘으로 수행합니다.
    
    Args:
        y: 반응 변수 데이터 (수치형 데이터 시퀀스).
        factor1: 첫 번째 독립변수 요인 데이터 (범주형 데이터 시퀀스).
        factor2: 두 번째 독립변수 요인 데이터 (범주형 데이터 시퀀스).
        profile: 검정 조건을 정의한 TwoWayANOVAProfile 객체.
        factor1_name: 첫 번째 요인의 변수명 (기본값 "Factor1").
        factor2_name: 두 번째 요인의 변수명 (기본값 "Factor2").
        response_name: 반응 변수의 변수명 (기본값 "Response").
    
    Returns:
        TwoWayANOVAResult: 검정 결과를 통일된 규격으로 담은 객체.
    
    Raises:
        ValueError: 입력 데이터 간 길이가 다르거나, 표본 크기가 부족한 경우.
    """
    y_arr = np.asarray(y, dtype=float)
    f1_arr = np.asarray(factor1)
    f2_arr = np.asarray(factor2)

    if not (len(y_arr) == len(f1_arr) == len(f2_arr)):
        raise ValueError("반응변수(y)와 두 요인(factor1, factor2)의 데이터 길이가 일치해야 합니다.")

    if len(y_arr) < 4:
        raise ValueError("Two-way ANOVA를 수행하려면 최소 4개 이상의 표본 데이터가 필요합니다.")

    df = pd.DataFrame({
        response_name: y_arr,
        factor1_name: f1_arr,
        factor2_name: f2_arr
    })

    # 책임 분기: 프로필의 interaction 설정에 따라 개별 실행기로 위임
    if profile.interaction:
        return _run_twoway_with_interaction(df, profile, factor1_name, factor2_name, response_name)
    else:
        return _run_twoway_without_interaction(df, profile, factor1_name, factor2_name, response_name)



if __name__ == "__main__":
    # 예시 데이터: 비료 종류(Fertilizer: A, B)와 급수 방법(Watering: Low, High)에 따른 식물 성장량(Yield)
    yield_data = [
        12.5, 13.1, 11.8, 14.0,  # Fertilizer A, Low Water
        16.2, 15.8, 17.1, 16.5,  # Fertilizer A, High Water
        10.1, 11.0, 10.5, 9.8,   # Fertilizer B, Low Water
        12.0, 12.8, 13.2, 12.1   # Fertilizer B, High Water
    ]

    fertilizer = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
    watering = ['Low', 'Low', 'Low', 'Low', 'High', 'High', 'High', 'High', 'Low', 'Low', 'Low', 'Low', 'High', 'High', 'High', 'High']

    # 1. 교호작용항을 포함하는 이원배치 분산분석 (With Interaction)
    profile_interaction = TwoWayANOVAProfile(interaction=True, ss_type=2, alpha=0.05)
    res_interaction = run_twoway_anova(
        y=yield_data,
        factor1=fertilizer,
        factor2=watering,
        profile=profile_interaction,
        factor1_name="Fertilizer",
        factor2_name="Watering",
        response_name="Yield"
    )

    print("=== 1. Two-way ANOVA (With Interaction) ===")
    print(f"Test Type   : {res_interaction.test_type}")
    print(f"[{res_interaction.factor1_name}] F: {res_interaction.f_factor1:.4f}, p: {res_interaction.p_factor1:.4e}, Reject H0: {res_interaction.reject_h0_factor1}")
    print(f"[{res_interaction.factor2_name}] F: {res_interaction.f_factor2:.4f}, p: {res_interaction.p_factor2:.4e}, Reject H0: {res_interaction.reject_h0_factor2}")
    if res_interaction.f_interaction is not None and res_interaction.p_interaction is not None:
        print(f"[Interaction] F: {res_interaction.f_interaction:.4f}, p: {res_interaction.p_interaction:.4e}, Reject H0: {res_interaction.reject_h0_interaction}")
    print("\n[ANOVA Table]")
    print(res_interaction.anova_table)
    print("\n" + "=" * 80 + "\n")

    # 2. 주효과만 고려하는 이원배치 분산분석 (Main Effects Only)
    profile_additive = TwoWayANOVAProfile(interaction=False, ss_type=2, alpha=0.05)
    res_additive = run_twoway_anova(
        y=yield_data,
        factor1=fertilizer,
        factor2=watering,
        profile=profile_additive,
        factor1_name="Fertilizer",
        factor2_name="Watering",
        response_name="Yield"
    )

    print("=== 2. Two-way ANOVA (Main Effects Only) ===")
    print(f"Test Type   : {res_additive.test_type}")
    print(f"[{res_additive.factor1_name}] F: {res_additive.f_factor1:.4f}, p: {res_additive.p_factor1:.4e}, Reject H0: {res_additive.reject_h0_factor1}")
    print(f"[{res_additive.factor2_name}] F: {res_additive.f_factor2:.4f}, p: {res_additive.p_factor2:.4e}, Reject H0: {res_additive.reject_h0_factor2}")
    print("\n[ANOVA Table]")
    print(res_additive.anova_table)