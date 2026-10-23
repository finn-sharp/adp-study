"""
================================================================================
ols.py - OLS 계수 t·전체 F·R²·평균/관측 예측구간
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
class OLSProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
        fit_intercept (bool): 절편을 추가; 입력 상수 Feature는 거부. 기본 True.
        cov_type (str): nonrobust/HC0/HC1/HC2/HC3/HAC 공분산. 기본 'nonrobust'.
        hac_maxlags (int): HAC의 비음수 최대 지연; 행 순서/간격 책임은 호출자. 기본 1.
    """
    alpha: float = 0.05
    fit_intercept: bool = True
    cov_type: str = 'nonrobust'
    hac_maxlags: int = 1

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class OLSOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        model (Any): 적합 statsmodels OLS; runtime JSON 제외.
        coefficients (pd.DataFrame): term/coefficient/standard_error/t_statistic/p_value/ci_lower/ci_upper/reject_h0.
        fit_metrics (Dict[str,Any]): 학습 R²·수정 R²·F·p·자유도·AIC/BIC; 절편 없는 R²는 uncentered.
        design (pd.DataFrame): 적합 행 순서의 절편 포함 설계 행렬.
        fitted_values (np.ndarray): 학습 적합값; 일반화 평가 아님.
        residuals (np.ndarray): y-fitted; 모델 반응변수 좌표계.
        new_predictions (Optional[pd.DataFrame]): 새 행의 mean·평균 CI·관측 PI; 목표 변환 시 그 좌표계.
        parameters (Dict[str,Any]): 적용 설정.
        warnings (List[str]): 추론 가정·robust/선택 후 해석 정보.
    """
    model: Any
    coefficients: pd.DataFrame
    fit_metrics: Dict[str,Any]
    design: pd.DataFrame
    fitted_values: np.ndarray
    residuals: np.ndarray
    new_predictions: Optional[pd.DataFrame]
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
    validate_profile(profile,OLSProfile)
    if profile.cov_type not in ('nonrobust','HC0','HC1','HC2','HC3','HAC') or profile.hac_maxlags<0:raise ValueError('공분산 설정 오류')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run(X, y, X_new, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        X_new: 전용 명세의 X_new 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        OLSOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    result,raw,target=fit_ols(X,y,profile)
    interval=np.asarray(result.conf_int(alpha=profile.alpha));coef=np.asarray(result.params);p=np.asarray(result.pvalues)
    table=pd.DataFrame({'term':result.model.exog_names,'coefficient':coef,'standard_error':np.asarray(result.bse),'t_statistic':np.asarray(result.tvalues),'p_value':p,'ci_lower':interval[:,0],'ci_upper':interval[:,1],'reject_h0':p<profile.alpha})
    new=None
    if X_new is not None:
        future,_=numeric_data(X_new)
        if future.columns.tolist()!=raw.columns.tolist():raise ValueError('새 Feature 이름/순서가 다릅니다.')
        if profile.fit_intercept:future.insert(0,'Intercept',1.)
        new=result.get_prediction(future).summary_frame(alpha=profile.alpha)
    metrics=dict(r2=float(result.rsquared),adjusted_r2=float(result.rsquared_adj),r2_kind='centered' if profile.fit_intercept else 'uncentered',f_statistic=float(result.fvalue),f_p_value=float(result.f_pvalue),f_kind='classical overall F' if profile.cov_type=='nonrobust' else 'covariance-based Wald F',df_model=float(result.df_model),df_resid=float(result.df_resid),n_samples=len(raw),aic=float(result.aic),bic=float(result.bic))
    notes=['계수 검정은 주어진 설계/모형에 조건부입니다. 변수·변환·관측 선택 후 p 값은 선택 과정을 반영하지 않습니다.']
    if profile.cov_type=='nonrobust':notes.append('고전적 소표본 t/F/구간은 올바른 선형 사양·독립·등분산·정규 오차 가정을 확인해야 합니다. 반응변수 자체의 정규성은 요구하지 않습니다.')
    if result.ssr<=np.finfo(float).eps*np.sum(target**2):notes.append('잔차분산이 거의 0이어서 t/F와 구간 추론이 퇴화할 수 있습니다. 비유한 값은 JSON null입니다.')
    if profile.cov_type!='nonrobust':notes.append('robust/HAC 공분산의 t·Wald F는 고전적 소표본 exact 검정이 아닙니다. robust 관측 PI는 일정 오차분산 가정의 근사입니다.')
    return OLSOutputDTO(result,table,metrics,pd.DataFrame(result.model.exog,index=raw.index,columns=result.model.exog_names),np.asarray(result.fittedvalues),np.asarray(result.resid),new,asdict(profile),notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_ols(X, y, X_new=None, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        X_new: 전용 명세의 X_new 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        OLSOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('regression-ols',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(X, y, X_new,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
