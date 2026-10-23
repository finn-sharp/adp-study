"""
================================================================================
leverage.py - leverage·studentized 잔차·Cook 등고선
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 그리기 함수는 사용자 ax만 사용하며 모형을 다시 적합하거나 관측을 삭제하지 않습니다.
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
from src.Visualization._common import PlotOutputDTO,validate_profile,start,finish,paired,vector
from src.stats.regression._common import ols_result,numeric_data
from statsmodels.stats.outliers_influence import OLSInfluence


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class LeverageProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
        color (str): 선/표식 색. 기본 'C0'.
        alpha (float): 표식 투명도 [0,1]. 기본 0.8.
        cook_levels (Optional[List[float]]): 양수 Cook 등고선; None은 [0.5,1.0]. 기본 None.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8
    cook_levels: Optional[List[float]] = None

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class LeverageOutputDTO(PlotOutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        figure (Any): 원 ax Figure; runtime JSON 제외.
        axes (Any): 전달한 동일 Axes; JSON 제외.
        artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
        data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
        diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
        parameters (Dict[str,Any]): 적용 Profile.
        warnings (List[str]): 진단 가정·선택 후 해석 정보.
    """
    figure: Any
    axes: Any
    artists: List[Any]
    data: pd.DataFrame
    diagnostics: Dict[str,Any]
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
    validate_profile(profile,LeverageProfile)
    if profile.cook_levels is not None and (not isinstance(profile.cook_levels,list) or not profile.cook_levels or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in profile.cook_levels)):raise ValueError('Cook 등고선 오류')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run(model, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        LeverageOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    before=start(ax)
    result=ols_result(model);influence=OLSInfluence(result);hat=np.asarray(influence.hat_matrix_diag);residual=np.asarray(influence.resid_studentized_internal);cook=np.asarray(influence.cooks_distance[0]);p=result.model.exog.shape[1]
    if not np.isfinite(residual).all() or (hat>=1-1e-12).any():raise ValueError('leverage/studentized 잔차 퇴화')
    levels=[.5,1.] if profile.cook_levels is None else profile.cook_levels
    ax.scatter(hat,residual,color=profile.color,alpha=profile.alpha,s=16);upper=min(.999,max(float(hat.max())*1.2,.1));grid=np.linspace(max(1e-4,float(hat.min())*.5),upper,100)
    for level in levels:
        contour=np.sqrt(level*p*(1-grid)/grid);ax.plot(grid,contour,'--',linewidth=.7,label='Cook='+str(level));ax.plot(grid,-contour,'--',linewidth=.7,color='gray')
    limit=max(3.,float(np.max(np.abs(residual)))*1.2);ax.set_ylim(-limit,limit);ax.axhline(0,color='black',linewidth=.5);ax.set_xlabel('Leverage h');ax.set_ylabel('Internally studentized residual');ax.legend(fontsize=8);data=pd.DataFrame({'position':np.arange(len(hat)),'leverage':hat,'studentized_internal':residual,'cooks_distance':cook});diagnostics={'p_including_intercept':p,'cook_levels':levels};notes=['등고선은 D=t_internal² h/[p(1-h)]의 기술 관계입니다. leverage가 높다는 이유만으로 삭제하지 않습니다.']
    return finish(ax,before,data,diagnostics,profile,LeverageOutputDTO,'Regression — leverage',notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def plot_leverage(model, *, ax, profile=None, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        LeverageOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('plot-regression-leverage',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(model, ax,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
