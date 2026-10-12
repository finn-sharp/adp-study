"""
================================================================================
k_distance.py - 밀도 군집 eps 후보와 거리의 급격한 변화를 검토합니다.
================================================================================
설명:
    - Profile → JSON → 단일 ax 시각화 → 수치 근거·runtime DTO.
분석 가정:
    - 사용자가 전달한 2D Axes에만 그리며 figure 생성·show·save·예측 모델 재학습을 수행하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    Axes 조합 EDA·의사결정 시각화 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.Recipes import resolve_profile
from src.Visualization._common import PlotOutputDTO, validate_profile, start, finish, vector, frame, paired, levels, heatmap, binary_scores

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class ClusteringKDistanceProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        k (int): 자기 자신 포함 k번째 이웃; 2 이상. 기본 5.
        metric (str): NearestNeighbors 거리. 기본 'euclidean'.
        eps (Optional[float]): 사용자가 검토하는 비음수 DBSCAN eps. 기본 None.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8
    k: int = 5
    metric: str = 'euclidean'
    eps: Optional[float] = None

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ClusteringKDistanceOutputDTO(PlotOutputDTO):
    """그린 축과 의사결정 수치.

    Attributes:
        figure (Any): 원 ax의 Figure; runtime으로 JSON 제외.
        axes (Any): 전달한 동일 Axes; runtime으로 JSON 제외.
        artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
        data (pd.DataFrame): 표시/계산한 수치와 원 label 순서.
        diagnostics (Dict[str, Any]): 표본 수·분모·선택 기준·추가 근거.
        parameters (Dict[str, Any]): 적용 Profile 설정.
        warnings (List[str]): 해석 가정·미표시/표본 제한 정보.
    """
    figure: Any
    axes: Any
    artists: List[Any]
    data: pd.DataFrame
    diagnostics: Dict[str, Any]
    parameters: Dict[str, Any]
    warnings: List[str]

# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate(profile):
    """전용 Profile과 공통 스타일을 확인합니다.

    Args:
        profile: ClusteringKDistanceProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, ClusteringKDistanceProfile)

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(values, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        ClusteringKDistanceOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    from sklearn.neighbors import NearestNeighbors
    raw=frame(values)
    if not 2<=profile.k<=len(raw):raise ValueError('k는 2 이상 표본 수 이하입니다.')
    if profile.eps is not None and (isinstance(profile.eps,bool) or not isinstance(profile.eps,(int,float)) or not np.isfinite(profile.eps) or profile.eps<0):raise ValueError('eps 오류')
    distances=NearestNeighbors(n_neighbors=profile.k,metric=profile.metric).fit(raw).kneighbors(raw,return_distance=True)[0][:,-1];ordered=np.sort(distances)
    ax.plot(np.arange(len(raw)),ordered,color=profile.color,alpha=profile.alpha)
    if profile.eps is not None:ax.axhline(profile.eps,color='C3',linestyle='--')
    ax.set_xlabel('Sorted sample rank');ax.set_ylabel('k-neighbor distance (self included)');data=pd.DataFrame({'rank':np.arange(len(raw)),'distance':ordered});diagnostics={'k':profile.k,'self_included':True,'fraction_within_eps':float((distances<=profile.eps).mean()) if profile.eps is not None else None};notes=['무릎은 eps 후보의 시각적 기준이며 자동 최적값이 아닙니다. scaling과 metric을 먼저 결정하세요.']
    return finish(ax, before, data, diagnostics, profile, ClusteringKDistanceOutputDTO, 'Clustering — K Distance', notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_k_distance(values, *, ax, profile=None, recipe_path=None, overrides=None) -> ClusteringKDistanceOutputDTO:
    """밀도 군집 eps 후보와 거리의 급격한 변화를 검토합니다.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        ClusteringKDistanceOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile=resolve_profile('plot-clustering-k-distance', profile, recipe_path=recipe_path, overrides=overrides)
    _validate(profile)
    before=start(ax)
    return _draw(values, ax, profile, before)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.clustering에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_k_distance(..., ax=axes[0, 0]); result.axes is axes[0, 0]
