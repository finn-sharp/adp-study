"""
================================================================================
TwoWay.py - 이원배치 ANOVA
================================================================================
설명:
    - 이원배치 ANOVA을 수행하고 Profile·Output DTO로 입력과 결과를 정의합니다.

분석 가정:
    - 독립 관측·잔차 정규성·등분산을 가정합니다. 교호작용 모델은 모든 요인 조합에 최소 두 관측이 필요합니다.
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
    2026-10-11  김재현(Finn)     v1.2.0    모델별 JSON 레시피 로딩과 공개 실행기 연결
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.2.0"
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
from src.Recipes import load_profile, resolve_profile
from ._common import validate_alpha, validate_groups, validate_labels
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class TwoWayANOVAProfile:
    """이원배치 ANOVA 입력 조건.

    Attributes:
        interaction (bool): 교호작용 포함 여부. 기본 True. False는 가법 주효과 모델.
        ss_type (int): 제곱합 타입 1/2/3. 기본 2. Type I는 요인 1→요인 2의 순차 검정. Sum 코딩 사용.
        alpha (float): 유의수준. 기본 0.05, 유한한 0<값<1. 기각 기준은 p_value<alpha.
    """

    interaction: bool = True
    ss_type: int = 2
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class TwoWayANOVAOutputDTO(OutputDTO):
    """이원배치 ANOVA 검정 결과.

    Attributes:
        test_type (str): 교호작용·SS 타입을 포함한 이원배치 검정 이름.
        factor1_name (str): 첫 번째 요인의 표시 이름.
        factor2_name (str): 두 번째 요인의 표시 이름.
        f_factor1 (float): 요인 1 F 통계량.
        p_factor1 (float): 요인 1 p-value.
        reject_h0_factor1 (bool): 요인 1 p<alpha 여부.
        f_factor2 (float): 요인 2 F 통계량.
        p_factor2 (float): 요인 2 p-value.
        reject_h0_factor2 (bool): 요인 2 p<alpha 여부.
        f_interaction (Optional[float]): 교호작용 F 통계량. interaction=False이면 None.
        p_interaction (Optional[float]): 교호작용 p-value. interaction=False이면 None.
        reject_h0_interaction (Optional[bool]): 교호작용 p<alpha 여부. interaction=False이면 None.
        anova_table (pd.DataFrame): SS·df·F·p 표. 행은 요인·교호작용·잔차. Type III는 절편 검정 포함.
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

TwoWayANOVAResult = TwoWayANOVAOutputDTO


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate_inputs(y: Any, factor1: Any, factor2: Any, profile: TwoWayANOVAProfile,
                     names: List[str]) -> pd.DataFrame:
    """이원배치 입력과 요인 조합을 검증합니다.

    Args:
        y: 유한한 수치형 1차원 반응변수.
        factor1: 첫 번째 요인 라벨, 최소 두 수준.
        factor2: 두 번째 요인 라벨, 최소 두 수준.
        profile: TwoWayANOVAProfile.
        names: 서로 다른 비어 있지 않은 요인1·요인2·반응 표시 이름.
    Returns:
        pd.DataFrame: 안전한 내부 열 이름으로 구성한 분석 데이터.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 조건·입력·요인 조합·이름 오류.
    """
    if not isinstance(profile, TwoWayANOVAProfile):
        raise TypeError("TwoWayANOVAProfile이 필요합니다.")
    validate_alpha(profile.alpha)
    if not isinstance(profile.interaction, bool) or isinstance(profile.ss_type, bool) or not isinstance(profile.ss_type,int) or profile.ss_type not in (1,2,3):
        raise ValueError("interaction은 bool, ss_type은 1/2/3 정수여야 합니다.")
    if any(not isinstance(s,str) or not s.strip() for s in names) or len(set(names)) != 3:
        raise ValueError("표시 이름은 서로 다른 비어 있지 않은 문자열이어야 합니다.")
    response = np.asarray(y,dtype=float)
    if response.ndim != 1 or len(response)<4 or not np.isfinite(response).all():
        raise ValueError("y는 최소 4행의 유한한 1차원 수치 자료여야 합니다.")
    f1, f2 = validate_labels(factor1,len(response)), validate_labels(factor2,len(response))
    data = pd.DataFrame({'Response':response,'Factor1':f1,'Factor2':f2})
    if profile.interaction:
        counts = data.groupby(['Factor1','Factor2']).size()
        if len(counts)!=len(np.unique(f1))*len(np.unique(f2)) or counts.min()<2:
            raise ValueError("교호작용 모델은 모든 요인 조합에 최소 두 관측이 필요합니다.")
    return data


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_twoway(data: pd.DataFrame, profile: TwoWayANOVAProfile,
                factor1_name: str, factor2_name: str) -> TwoWayANOVAOutputDTO:
    """Sum 코딩 OLS를 적합하고 이름으로 효과를 추출합니다.

    Args:
        data: 검증된 내부 열 데이터.
        profile: 검정 조건.
        factor1_name: 요인 1 표시 이름.
        factor2_name: 요인 2 표시 이름.
    Returns:
        TwoWayANOVAOutputDTO: 두 주효과와 선택적 교호작용 검정.
    Raises:
        ValueError: 설계 rank 부족·잔차 자유도/분산 부족·비유한 검정 결과.
    Notes:
        Type III는 Sum 대비를 사용하며 행 위치로 효과를 추출하지 않습니다.
    """
    t1,t2 = 'C(Factor1, Sum)','C(Factor2, Sum)'
    formula = 'Response ~ '+t1+(' * ' if profile.interaction else ' + ')+t2
    model = ols(formula,data=data).fit()
    if np.linalg.matrix_rank(model.model.exog)<model.model.exog.shape[1] or model.df_resid<=0 or model.ssr<=0:
        raise ValueError("추정 가능한 설계와 양의 잔차 자유도·분산이 필요합니다.")
    table = anova_lm(model,typ=profile.ss_type)
    values = {}
    for suffix,term in [('factor1',t1),('factor2',t2),('interaction',t1+':'+t2)]:
        if suffix=='interaction' and not profile.interaction:
            values.update(f_interaction=None,p_interaction=None,reject_h0_interaction=None)
            continue
        f,p = float(table.loc[term,'F']),float(table.loc[term,'PR(>F)'])
        if not np.isfinite([f,p]).all():
            raise ValueError("효과의 F 또는 p-value가 유한하지 않습니다.")
        values['f_'+suffix],values['p_'+suffix],values['reject_h0_'+suffix] = f,p,bool(p<profile.alpha)
    table = table.rename(index={t1:'C('+factor1_name+', Sum)',t2:'C('+factor2_name+', Sum)',
                               t1+':'+t2:'C('+factor1_name+', Sum):C('+factor2_name+', Sum)'})
    return TwoWayANOVAOutputDTO(test_type='Two-way ANOVA (Type {} SS, interaction={})'.format(profile.ss_type,profile.interaction),
                                factor1_name=factor1_name,factor2_name=factor2_name,anova_table=table,**values)


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_twoway_anova(y: Any, factor1: Any, factor2: Any,
                     profile: Optional[TwoWayANOVAProfile] = None,
                     factor1_name: str = "Factor1", factor2_name: str = "Factor2",
                     response_name: str = "Response",
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TwoWayANOVAOutputDTO:
    """두 요인의 평균 차이 및 선택적 교호작용을 검정합니다.

    Args:
        y: 수치형 1차원 반응변수.
        factor1: y와 같은 길이의 첫 번째 요인.
        factor2: y와 같은 길이의 두 번째 요인.
        profile: None이면 twoway-anova.json에서 구성.
        factor1_name: 요인 1 표시 이름. 수식에 삽입하지 않음.
        factor2_name: 요인 2 표시 이름. 수식에 삽입하지 않음.
        response_name: 반응변수 표시 이름. 수식에 삽입하지 않음.
        recipe_path: 사용자 JSON 경로. None이면 src/Recipes의 모델별 기본 파일.
        overrides: Profile이 없을 때 레시피 위에 적용할 명시적 설정 사전.
    Returns:
        TwoWayANOVAOutputDTO: 효과별 검정 및 분산분석표.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·설계·잔차 또는 수치 계산 오류.
        FileNotFoundError/OSError: 레시피 파일 누락 또는 읽기 실패.
    Notes:
        기존 함수 인자와 TwoWayANOVAResult 이름은 호환됩니다.
        유의한 교호작용은 주효과만으로 해석하지 말고 단순효과를 확인해야 합니다.
    """
    profile = resolve_profile("twoway-anova", profile, recipe_path=recipe_path, overrides=overrides)
    data = _validate_inputs(y,factor1,factor2,profile,[factor1_name,factor2_name,response_name])
    return _run_twoway(data,profile,factor1_name,factor2_name)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    y = [1,2,3,4, 3,4,5,7, 5,6,7,9, 10,11,12,14]
    factor1 = ['A']*8+['B']*8
    factor2 = (['Low']*4+['High']*4)*2
    for ss_type in (1,2,3):
        result = run_twoway_anova(y,factor1,factor2,load_profile("twoway-anova", overrides={"ss_type": ss_type}))
        print(result.anova_table)
        print(result.to_json()[:200])
