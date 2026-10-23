"""
================================================================================
loadings.py - 성분을 구성하는 원 Feature와 부호·기여 패턴을 해석합니다.
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
class DecompositionLoadingsProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        annotate (bool): 각 성분 계수 표시. 기본 False.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    annotate: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class DecompositionLoadingsOutputDTO(PlotOutputDTO):
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
        profile: DecompositionLoadingsProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, DecompositionLoadingsProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(components, feature_names, ax, profile, before, cax):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        components: (성분, 원 Feature) 순서의 components_/loading 행렬.
        feature_names: 원 Feature의 고유 이름과 순서.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
        cax: 선택 색상 막대 축.
    Returns:
        DecompositionLoadingsOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    raw = frame(components)
    if feature_names is not None:
        if len(feature_names) != raw.shape[1] or len(
            set(str(v) for v in feature_names)
        ) != len(feature_names):
            raise ValueError("고유 Feature 이름/길이 오류")
        raw.columns = [str(v) for v in feature_names]
    raw.index = ["PC" + str(i + 1) for i in range(len(raw))]
    limit = float(np.max(np.abs(raw.values))) or 1.0
    heatmap(
        ax,
        raw.values,
        raw.index,
        raw.columns,
        cmap="coolwarm",
        cax=cax,
        vmin=-limit,
        vmax=limit,
        annotate=profile.annotate,
    )
    ax.set_xlabel("Original feature")
    ax.set_ylabel("Component")
    data = raw
    diagnostics = {"n_components": len(raw), "n_features": raw.shape[1]}
    notes = [
        "입력은 components_/명시 loading 행렬입니다. 성분의 전체 부호는 임의이며 서로 다른 적합 결과의 부호를 직접 비교하지 않습니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        DecompositionLoadingsOutputDTO,
        "Decomposition — Loadings",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_loadings(
    components,
    feature_names=None,
    *,
    ax,
    profile=None,
    recipe_path=None,
    overrides=None,
    cax=None,
) -> DecompositionLoadingsOutputDTO:
    """성분을 구성하는 원 Feature와 부호·기여 패턴을 해석합니다.

    Args:
        components: (성분, 원 Feature) 순서의 components_/loading 행렬.
        feature_names: 원 Feature의 고유 이름과 순서.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
        cax: 명시적 색상 막대 Axes; None이면 색상 막대 생략.
    Returns:
        DecompositionLoadingsOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-decomposition-loadings",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax, cax)
    return _draw(components, feature_names, ax, profile, before, cax)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.decomposition에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_loadings(..., ax=axes[0, 0]); result.axes is axes[0, 0]
