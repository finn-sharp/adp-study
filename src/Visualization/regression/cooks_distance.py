"""
================================================================================
cooks_distance.py - Cook 관측 영향력과 명시 기준
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
class CooksDistanceProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
        color (str): 선/표식 색. 기본 'C0'.
        alpha (float): 표식 투명도 [0,1]. 기본 0.8.
        threshold (Optional[float]): 양수 검토 기준; None=4/n. 기본 None.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8
    threshold: Optional[float] = None

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CooksDistanceOutputDTO(PlotOutputDTO):
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
    validate_profile(profile,CooksDistanceProfile)
    if profile.threshold is not None and (isinstance(profile.threshold,bool) or not isinstance(profile.threshold,(int,float)) or not np.isfinite(profile.threshold) or profile.threshold<=0):raise ValueError('Cook 기준 오류')

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
        CooksDistanceOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    before=start(ax)
    result=ols_result(model);cook=np.asarray(OLSInfluence(result).cooks_distance[0]);n=len(cook)
    if not np.isfinite(cook).all():raise ValueError('Cook 지표가 퇴화했습니다.')
    threshold=4/n if profile.threshold is None else profile.threshold;positions=np.arange(n);flag=cook>threshold
    ax.vlines(positions,0,cook,color=profile.color,alpha=profile.alpha);ax.scatter(positions[flag],cook[flag],color='C3',s=16,label='Review');ax.axhline(threshold,color='black',linestyle='--',label='Threshold='+format(threshold,'.3g'));ax.legend(fontsize=8);ax.set_xlabel('Fitted observation position');ax.set_ylabel("Cook's distance");labels=result.model.data.row_labels
    if labels is None:labels=positions
    data=pd.DataFrame({'position':positions,'row_label':list(labels),'cooks_distance':cook,'flag':flag});diagnostics={'threshold':threshold,'flagged':int(flag.sum()),'n':n};notes=['영향 관측은 오류/결측과 다릅니다. 이 함수는 표시만 하며 자동 삭제하지 않습니다.']
    return finish(ax,before,data,diagnostics,profile,CooksDistanceOutputDTO,'Regression — cooks distance',notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def plot_cooks_distance(model, *, ax, profile=None, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        CooksDistanceOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('plot-regression-cooks-distance',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(model, ax,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
