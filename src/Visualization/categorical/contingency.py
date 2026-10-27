"""
================================================================================
contingency.py - 범주 조합의 집중·조건부 구성 차이와 희소 셀을 판단합니다.
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
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
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
from src.Visualization._common import (
    PlotOutputDTO,
    validate_profile,
    start,
    finish,
    vector,
    frame,
    paired,
    levels,
    heatmap,
    binary_scores,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class CategoricalContingencyProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        normalization (str): count/row/column/all 분모. 기본 'row'.
        annotate (bool): 셀 값 표시. 기본 True.
        max_categories (int): 각 축의 최대 범주 수. 기본 20.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    normalization: str = "row"
    annotate: bool = True
    max_categories: int = 20


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CategoricalContingencyOutputDTO(PlotOutputDTO):
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
        profile: CategoricalContingencyProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, CategoricalContingencyProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(x, y, ax, profile, before, cax):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        x: 첫 대응 원 벡터; contingency/mosaic는 범주, scatter/hexbin은 수치.
        y: 둘째 대응 원 벡터; Series 입력끼리는 인덱스가 같아야 함.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
        cax: 선택 색상 막대 축.
    Returns:
        CategoricalContingencyOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    raw, removed = paired(x=x, y=y)
    table = pd.crosstab(levels(raw.x), levels(raw.y))
    if max(table.shape) > profile.max_categories or profile.max_categories < 1:
        raise ValueError("범주 한도 초과")
    if profile.normalization == "count":
        data = table.astype(float)
    elif profile.normalization == "row":
        data = table.div(table.sum(axis=1), axis=0)
    elif profile.normalization == "column":
        data = table.div(table.sum(axis=0), axis=1)
    elif profile.normalization == "all":
        data = table / table.to_numpy().sum()
    else:
        raise ValueError("normalization 오류")
    heatmap(
        ax, data.values, data.index, data.columns, cax=cax, annotate=profile.annotate
    )
    ax.set_xlabel("Y category")
    ax.set_ylabel("X category")
    diagnostics = {
        "n": len(raw),
        "missing_removed": removed,
        "counts": table,
        "normalization": profile.normalization,
    }
    notes = ["비율과 원 빈도를 함께 보세요. 색 차이 자체는 유의성 검정이 아닙니다."]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        CategoricalContingencyOutputDTO,
        "Categorical — Contingency",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_contingency(
    x, y, *, ax, profile=None, recipe_path=None, overrides=None, cax=None
) -> CategoricalContingencyOutputDTO:
    """범주 조합의 집중·조건부 구성 차이와 희소 셀을 판단합니다.

    Args:
        x: 첫 대응 원 벡터; contingency/mosaic는 범주, scatter/hexbin은 수치.
        y: 둘째 대응 원 벡터; Series 입력끼리는 인덱스가 같아야 함.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
        cax: 명시적 색상 막대 Axes; None이면 색상 막대 생략.
    Returns:
        CategoricalContingencyOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-categorical-contingency",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax, cax)
    return _draw(x, y, ax, profile, before, cax)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.categorical에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_contingency(..., ax=axes[0, 0]); result.axes is axes[0, 0]
