"""
================================================================================
explained_variance.py - 목표 누적분산에 필요한 성분 수와 차원 축소 손실을 비교합니다.
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
class DecompositionExplainedVarianceProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        target_fraction (float): 누적 설명분산 선택 기준 (0,1]. 기본 0.9.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    target_fraction: float = 0.9


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class DecompositionExplainedVarianceOutputDTO(PlotOutputDTO):
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
        profile: DecompositionExplainedVarianceProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, DecompositionExplainedVarianceProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(ratios, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        ratios: 성분 순서의 설명분산 비율; 비음수이고 합 ≤1.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        DecompositionExplainedVarianceOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    a, removed = vector(ratios, allow_missing=False)
    if (a < 0).any() or a.sum() > 1 + 1e-8 or not 0 < profile.target_fraction <= 1:
        raise ValueError("설명분산 비율/기준 오류")
    x = np.arange(1, len(a) + 1)
    cumulative = np.cumsum(a)
    ax.bar(x, a, color=profile.color, alpha=profile.alpha, label="Per component")
    ax.plot(x, cumulative, "o-", color="C3", label="Cumulative")
    ax.axhline(profile.target_fraction, color="black", linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance fraction")
    ax.legend(fontsize=8)
    data = pd.DataFrame({"component": x, "ratio": a, "cumulative_ratio": cumulative})
    indices = np.flatnonzero(cumulative >= profile.target_fraction)
    diagnostics = {
        "components_to_target": int(indices[0] + 1) if len(indices) else None,
        "covered_fraction": float(a.sum()),
    }
    notes = [
        "미제공 성분의 분산은 추정하지 않습니다. 설명분산 자체는 예측 품질이 아닙니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        DecompositionExplainedVarianceOutputDTO,
        "Decomposition — Explained Variance",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_explained_variance(
    ratios, *, ax, profile=None, recipe_path=None, overrides=None
) -> DecompositionExplainedVarianceOutputDTO:
    """목표 누적분산에 필요한 성분 수와 차원 축소 손실을 비교합니다.

    Args:
        ratios: 성분 순서의 설명분산 비율; 비음수이고 합 ≤1.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        DecompositionExplainedVarianceOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-decomposition-explained-variance",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax)
    return _draw(ratios, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.decomposition에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_explained_variance(..., ax=axes[0, 0]); result.axes is axes[0, 0]
