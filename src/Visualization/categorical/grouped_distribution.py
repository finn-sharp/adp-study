"""
================================================================================
grouped_distribution.py - 집단별 분포·중앙값·산포·소표본 차이를 비교합니다.
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
class CategoricalGroupedDistributionProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        kind (str): box/violin/strip; 같은 Axes에 범주별 수치 분포. 기본 'box'.
        max_categories (int): 최대 범주; 초과하면 오류로 명시적 선택 요구. 기본 20.
        whisker (float): box의 IQR whisker 배율. 기본 1.5.
        random_state (int): strip jitter seed. 기본 1234.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    kind: str = "box"
    max_categories: int = 20
    whisker: float = 1.5
    random_state: int = 1234


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CategoricalGroupedDistributionOutputDTO(PlotOutputDTO):
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
        profile: CategoricalGroupedDistributionProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, CategoricalGroupedDistributionProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(values, categories, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        categories: 행별 원 범주 벡터; 문자열 충돌을 거부하고 대응 결측 행은 제거.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        CategoricalGroupedDistributionOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    raw, removed = paired(value=values, category=categories)
    raw["value"] = vector(raw.value, allow_missing=False)[0]
    raw["category"] = levels(raw.category)
    names = list(pd.unique(raw.category))
    groups = [raw.loc[raw.category == name, "value"].to_numpy() for name in names]
    if (
        len(names) > profile.max_categories
        or profile.max_categories < 1
        or profile.whisker <= 0
        or profile.kind not in ("box", "violin", "strip")
    ):
        raise ValueError("범주 한도/분포 방법 오류")
    positions = np.arange(len(names))
    if profile.kind == "box":
        artist = ax.boxplot(
            groups, positions=positions, whis=profile.whisker, patch_artist=True
        )
        for box in artist["boxes"]:
            box.set_facecolor(profile.color)
            box.set_alpha(profile.alpha)
    elif profile.kind == "violin":
        if any(len(g) < 2 or np.ptp(g) == 0 for g in groups):
            raise ValueError("violin은 각 그룹에 2개 이상 비상수 관측이 필요합니다.")
        artist = ax.violinplot(groups, positions=positions, showmedians=True)
        for item in artist["bodies"]:
            item.set_facecolor(profile.color)
            item.set_alpha(profile.alpha)
    else:
        rng = np.random.RandomState(profile.random_state)
        for pos, g in zip(positions, groups):
            ax.scatter(
                pos + rng.uniform(-0.15, 0.15, len(g)),
                g,
                s=10,
                color=profile.color,
                alpha=profile.alpha,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    data = pd.DataFrame(
        [
            dict(
                category=n,
                n=len(g),
                median=float(np.median(g)),
                q25=float(np.quantile(g, 0.25)),
                q75=float(np.quantile(g, 0.75)),
                mean=float(g.mean()),
            )
            for n, g in zip(names, groups)
        ]
    )
    diagnostics = {
        "n": len(raw),
        "missing_removed": removed,
        "outliers_are_not_errors": True,
    }
    notes = ["그룹 표본 수를 함께 확인하세요. box 이상치는 자동 삭제 기준이 아닙니다."]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        CategoricalGroupedDistributionOutputDTO,
        "Categorical — Grouped Distribution",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_grouped_distribution(
    values, categories, *, ax, profile=None, recipe_path=None, overrides=None
) -> CategoricalGroupedDistributionOutputDTO:
    """집단별 분포·중앙값·산포·소표본 차이를 비교합니다.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        categories: 행별 원 범주 벡터; 문자열 충돌을 거부하고 대응 결측 행은 제거.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        CategoricalGroupedDistributionOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-categorical-grouped-distribution",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax)
    return _draw(values, categories, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.categorical에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_grouped_distribution(..., ax=axes[0, 0]); result.axes is axes[0, 0]
