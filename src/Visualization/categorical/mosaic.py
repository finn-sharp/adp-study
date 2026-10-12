"""
================================================================================
mosaic.py - 전체 구성과 범주 간 조건부 연관 패턴을 동시에 비교합니다.
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
class CategoricalMosaicProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        max_categories (int): 각 축의 최대 범주 수. 기본 12.
        annotate (bool): 셀 원 빈도를 표시. 기본 True.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8
    max_categories: int = 12
    annotate: bool = True

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CategoricalMosaicOutputDTO(PlotOutputDTO):
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
        profile: CategoricalMosaicProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, CategoricalMosaicProfile)

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(x, y, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        x: 첫 대응 원 벡터; contingency/mosaic는 범주, scatter/hexbin은 수치.
        y: 둘째 대응 원 벡터; Series 입력끼리는 인덱스가 같아야 함.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        CategoricalMosaicOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    from matplotlib.patches import Rectangle
    raw,removed=paired(x=x,y=y);table=pd.crosstab(levels(raw.x),levels(raw.y))
    if profile.max_categories<1 or max(table.shape)>profile.max_categories:raise ValueError('범주 한도 초과')
    left=0.;colors=['C'+str(i%10) for i in range(table.shape[1])]
    for i,(name,row) in enumerate(table.iterrows()):
        width=row.sum()/len(raw);bottom=0.
        for j,(other,count) in enumerate(row.items()):
            height=count/row.sum();ax.add_patch(Rectangle((left,bottom),width,height,facecolor=colors[j],alpha=profile.alpha,edgecolor='white',label=other if i==0 else None))
            if profile.annotate and width*height>.025:ax.text(left+width/2,bottom+height/2,str(int(count)),ha='center',va='center',fontsize=8)
            bottom+=height
        ax.text(left+width/2,-.03,name,ha='right',va='top',rotation=45,fontsize=8);left+=width
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xticks([]);ax.set_ylabel('Conditional Y fraction');ax.legend(fontsize=8)
    data=table;diagnostics={'n':len(raw),'missing_removed':removed,'area_denominator':'all valid rows'};notes=['면적은 공동 비율, 높이는 X 안에서의 Y 비율입니다.']
    return finish(ax, before, data, diagnostics, profile, CategoricalMosaicOutputDTO, 'Categorical — Mosaic', notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_mosaic(x, y, *, ax, profile=None, recipe_path=None, overrides=None) -> CategoricalMosaicOutputDTO:
    """전체 구성과 범주 간 조건부 연관 패턴을 동시에 비교합니다.

    Args:
        x: 첫 대응 원 벡터; contingency/mosaic는 범주, scatter/hexbin은 수치.
        y: 둘째 대응 원 벡터; Series 입력끼리는 인덱스가 같아야 함.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        CategoricalMosaicOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile=resolve_profile('plot-categorical-mosaic', profile, recipe_path=recipe_path, overrides=overrides)
    _validate(profile)
    before=start(ax)
    return _draw(x, y, ax, profile, before)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.categorical에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_mosaic(..., ax=axes[0, 0]); result.axes is axes[0, 0]
