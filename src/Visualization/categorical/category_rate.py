"""
================================================================================
category_rate.py - 범주별 발생률과 표본 수·불확실성으로 우선 검토 집단을 선택합니다.
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
class CategoricalCategoryRateProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        positive_label (Any): 양성 outcome 라벨; 원 의미를 명시. 기본 1.
        confidence (float): Wilson 이항 신뢰수준 (0,1). 기본 0.95.
        max_categories (int): 최대 범주 수. 기본 20.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    positive_label: Any = 1
    confidence: float = 0.95
    max_categories: int = 20


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CategoricalCategoryRateOutputDTO(PlotOutputDTO):
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
        profile: CategoricalCategoryRateProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, CategoricalCategoryRateProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(categories, outcome, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        categories: 행별 원 범주 벡터; 문자열 충돌을 거부하고 대응 결측 행은 제거.
        outcome: 행별 이진 outcome; positive_label의 의미를 명시.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        CategoricalCategoryRateOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    from scipy.stats import norm

    raw, removed = paired(category=categories, outcome=outcome)
    raw["category"] = levels(raw.category)
    unique = pd.unique(raw.outcome)
    if (
        len(unique) != 2
        or profile.positive_label not in unique
        or not 0 < profile.confidence < 1
        or profile.max_categories < 1
    ):
        raise ValueError("이진 outcome/신뢰수준 오류")
    raw["positive"] = (raw.outcome == profile.positive_label).astype(int)
    summary = raw.groupby("category", sort=False).positive.agg(["sum", "count"])
    if len(summary) > profile.max_categories:
        raise ValueError("범주 한도 초과")
    n = summary["count"].to_numpy()
    success = summary["sum"].to_numpy()
    rate = success / n
    z = norm.ppf((1 + profile.confidence) / 2)
    den = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / den
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / den
    low = np.clip(center - half, 0.0, 1.0)
    high = np.clip(center + half, 0.0, 1.0)
    low = np.minimum(low, rate)
    high = np.maximum(high, rate)
    positions = np.arange(len(summary))
    ax.errorbar(
        rate,
        positions,
        xerr=np.vstack([rate - low, high - rate]),
        fmt="o",
        color=profile.color,
        alpha=profile.alpha,
        capsize=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [str(c) + " (n=" + str(count) + ")" for c, count in zip(summary.index, n)]
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Positive fraction with Wilson interval")
    data = pd.DataFrame(
        {
            "category": summary.index,
            "positive_count": success,
            "n": n,
            "rate": rate,
            "lower": low,
            "upper": high,
        }
    )
    diagnostics = {
        "n": len(raw),
        "missing_removed": removed,
        "positive_label": profile.positive_label,
        "confidence": profile.confidence,
    }
    notes = [
        "독립 이항 관측의 개별 구간이며 다중비교 보정·인과·일반화 평가를 뜻하지 않습니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        CategoricalCategoryRateOutputDTO,
        "Categorical — Category Rate",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_category_rate(
    categories, outcome, *, ax, profile=None, recipe_path=None, overrides=None
) -> CategoricalCategoryRateOutputDTO:
    """범주별 발생률과 표본 수·불확실성으로 우선 검토 집단을 선택합니다.

    Args:
        categories: 행별 원 범주 벡터; 문자열 충돌을 거부하고 대응 결측 행은 제거.
        outcome: 행별 이진 outcome; positive_label의 의미를 명시.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        CategoricalCategoryRateOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-categorical-category-rate",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax)
    return _draw(categories, outcome, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.categorical에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_category_rate(..., ax=axes[0, 0]); result.axes is axes[0, 0]
