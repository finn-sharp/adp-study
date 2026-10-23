"""
================================================================================
heteroscedasticity.py - Breusch–Pagan/Koenker·White 등분산성 진단
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 학습 적합과 선택은 학습/CV 안에서 수행하고 추론 가정을 확인합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
================================================================================
"""
__author__='김재현(Finn)'
__email__='penguin.klg@gmail.com'
__version__='1.0.0'
__status__='Development'

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass,asdict
from typing import Any,Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from scipy import stats
from src.stats.regression._common import validate_profile,numeric_data,fit_ols,ols_result,test_row,vif_values


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class HeteroscedasticityProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
        method (str): breusch_pagan/white. 기본 'breusch_pagan'.
        koenker (bool): BP robust=True의 Koenker 버전; 정규성 완화. 기본 True.
    """
    alpha: float = 0.05
    method: str = 'breusch_pagan'
    koenker: bool = True

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class HeteroscedasticityOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
        details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
        parameters (Dict[str,Any]): 적용 Profile.
        warnings (List[str]): 검정 가정·선택 후 추론 정보.
    """
    tests: pd.DataFrame
    details: Dict[str,Any]
    parameters: Dict[str,Any]
    warnings: List[str]

# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate(profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
    Returns:
        object: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    validate_profile(profile,HeteroscedasticityProfile)
    if profile.method not in ('breusch_pagan','white'):raise ValueError('등분산 진단 방법 오류')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run(model, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        HeteroscedasticityOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    from statsmodels.stats.diagnostic import het_breuschpagan,het_white
    result=ols_result(model)
    if not result.model.k_constant:raise ValueError('등분산 보조회귀에는 절편을 포함한 OLS가 필요합니다.')
    if profile.method=='white':
        d=result.model.exog.shape[1]
        if result.nobs<=d*(d+1)//2:raise ValueError('White 보조회귀의 곱/제곱 항 대비 표본 수가 부족합니다.')
        lm,p,f,fp=het_white(result.resid,result.model.exog)
    else:lm,p,f,fp=het_breuschpagan(result.resid,result.model.exog,robust=profile.koenker)
    rows=[test_row(profile.method+' LM',lm,p,profile.alpha),test_row(profile.method+' F',f,fp,profile.alpha)]
    return HeteroscedasticityOutputDTO(pd.DataFrame(rows),{'n_samples':int(result.nobs),'hypothesis':'오차 등분산'},asdict(profile),['이분산이면 변환·사양·HC3/모형화된 가중회귀 등을 검토하세요. robust 표준오차는 이분산 자체를 제거하거나 예측분산을 자동 모형화하지 않습니다.'])

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_heteroscedasticity(model, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        HeteroscedasticityOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('regression-heteroscedasticity',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(model,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
