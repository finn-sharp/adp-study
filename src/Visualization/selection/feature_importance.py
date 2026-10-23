"""
================================================================================
feature_importance.py - 동일 평가 기준의 Feature 중요도와 반복 변동을 비교해 검토 우선순위를 정합니다.
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
class SelectionFeatureImportanceProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        max_features (int): 표시할 Feature 수; 큰 값 순. 기본 20.
        error_multiplier (float): 표시 오차=입력 deviations×배율; 통계 CI를 자동 의미하지 않음. 기본 1.0.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    max_features: int = 20
    error_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class SelectionFeatureImportanceOutputDTO(PlotOutputDTO):
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
        profile: SelectionFeatureImportanceProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, SelectionFeatureImportanceProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(feature_names, importances, deviations, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        feature_names: 원 Feature의 고유 이름과 순서.
        importances: Feature 순서의 동일 scoring 기준 중요도; 음수도 허용.
        deviations: 선택 비음수 반복 변동/오차; 자동 신뢰구간이 아님.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        SelectionFeatureImportanceOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    raw, removed = paired(feature=feature_names, importance=importances)
    raw["feature"] = levels(raw.feature)
    raw["importance"] = vector(raw.importance, allow_missing=False)[0]
    if (
        removed
        or raw.feature.duplicated().any()
        or profile.max_features < 1
        or profile.error_multiplier < 0
    ):
        raise ValueError("고유 Feature/표시 설정 오류")
    if deviations is not None:
        error, _ = vector(deviations, allow_missing=False)
        if len(error) != len(raw) or (error < 0).any():
            raise ValueError("deviations 길이/음수 오류")
        raw["deviation"] = error
    else:
        raw["deviation"] = 0.0
    data = raw.sort_values("importance", ascending=False, kind="stable").reset_index(
        drop=True
    )
    shown = data.iloc[: profile.max_features]
    pos = np.arange(len(shown))
    ax.barh(
        pos,
        shown.importance,
        xerr=shown.deviation * profile.error_multiplier,
        color=profile.color,
        alpha=profile.alpha,
        capsize=3,
    )
    ax.set_yticks(pos)
    ax.set_yticklabels(shown.feature)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Importance in the supplied scoring space")
    diagnostics = {
        "n_features": len(raw),
        "displayed_features": len(shown),
        "error_multiplier": profile.error_multiplier,
    }
    notes = [
        "입력 중요도의 정의·평가 split·오차 의미를 확인하세요. 상관 Feature의 중요도는 분산되며 크기는 인과 기여가 아닙니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        SelectionFeatureImportanceOutputDTO,
        "Selection — Feature Importance",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_feature_importance(
    feature_names,
    importances,
    deviations=None,
    *,
    ax,
    profile=None,
    recipe_path=None,
    overrides=None,
) -> SelectionFeatureImportanceOutputDTO:
    """동일 평가 기준의 Feature 중요도와 반복 변동을 비교해 검토 우선순위를 정합니다.

    Args:
        feature_names: 원 Feature의 고유 이름과 순서.
        importances: Feature 순서의 동일 scoring 기준 중요도; 음수도 허용.
        deviations: 선택 비음수 반복 변동/오차; 자동 신뢰구간이 아님.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        SelectionFeatureImportanceOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-selection-feature-importance",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax)
    return _draw(feature_names, importances, deviations, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.selection에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_feature_importance(..., ax=axes[0, 0]); result.axes is axes[0, 0]
