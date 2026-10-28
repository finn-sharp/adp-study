"""
================================================================================
count.py - 클래스 불균형·희귀 범주·범주 통합 필요성을 판단합니다.
================================================================================
설명:
    - 직접 실행(__main__) 시 고정 합성 입력으로 PNG 예시를 저장합니다.
    - Profile → JSON → 단일 ax 시각화 → 수치 근거·runtime DTO.
분석 가정:
    - plot 함수는 사용자가 전달한 2D Axes에만 그리며 figure 생성·show·save·예측 모델 재학습을 수행하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    직접 실행 PNG 예시 저장 추가
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
class CategoricalCountProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        max_categories (int): 표시할 최대 범주 수; 초과 범주는 진단에 보고. 기본 20.
        proportion (bool): 전체 유효 범주 표본 대비 비율. 기본 False.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    max_categories: int = 20
    proportion: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CategoricalCountOutputDTO(PlotOutputDTO):
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
        profile: CategoricalCountProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, CategoricalCountProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(categories, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        categories: 행별 원 범주 벡터; 문자열 충돌을 거부하고 대응 결측 행은 제거.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        CategoricalCountOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    raw, removed = paired(category=categories)
    raw["category"] = levels(raw.category)
    counts = raw.category.value_counts()
    if profile.max_categories < 1:
        raise ValueError("max_categories 오류")
    shown = counts.iloc[: profile.max_categories]
    height = shown / len(raw) if profile.proportion else shown
    ax.barh(np.arange(len(shown)), height, color=profile.color, alpha=profile.alpha)
    ax.set_yticks(np.arange(len(shown)))
    ax.set_yticklabels(shown.index)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of all valid rows" if profile.proportion else "Count")
    data = pd.DataFrame(
        {
            "category": counts.index,
            "count": counts.to_numpy(),
            "fraction": counts.to_numpy() / len(raw),
            "displayed": np.arange(len(counts)) < len(shown),
        }
    )
    diagnostics = {
        "n": len(raw),
        "missing_removed": removed,
        "omitted_rows": int(counts.iloc[len(shown) :].sum()),
        "n_categories": len(counts),
    }
    notes = (
        ["표시 한도를 초과한 범주를 묶거나 분모에서 제외하지 않습니다."]
        if len(shown) < len(counts)
        else []
    )
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        CategoricalCountOutputDTO,
        "Categorical — Count",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_count(
    categories, *, ax, profile=None, recipe_path=None, overrides=None
) -> CategoricalCountOutputDTO:
    """클래스 불균형·희귀 범주·범주 통합 필요성을 판단합니다.

    Args:
        categories: 행별 원 범주 벡터; 문자열 충돌을 거부하고 대응 결측 행은 제거.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        CategoricalCountOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-categorical-count", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    before = start(ax)
    return _draw(categories, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.categorical에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_count(..., ax=axes[0, 0]); result.axes is axes[0, 0]


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("categorical.count", plot_count)
