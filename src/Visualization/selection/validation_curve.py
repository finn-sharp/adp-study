"""
================================================================================
validation_curve.py - 학습량/분류 설정 변화의 train–validation 차이로 과적합과 개선 여지를 검토합니다.
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
class SelectionValidationCurveProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        x_label (str): 표시할 x축 의미. 기본 'Training samples / model setting'.
        score_label (str): 동일한 분류 평가 점수 이름. 기본 'Classification score (higher is better)'.
        error_multiplier (float): 입력 표준편차 등의 표시 배율; CI를 자동 추정하지 않음. 기본 1.0.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    x_label: str = "Training samples / model setting"
    score_label: str = "Classification score (higher is better)"
    error_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class SelectionValidationCurveOutputDTO(PlotOutputDTO):
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
        profile: SelectionValidationCurveProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, SelectionValidationCurveProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(
    x_values,
    train_scores,
    validation_scores,
    train_deviations,
    validation_deviations,
    ax,
    profile,
    before,
):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        x_values: 학습량/모델 설정의 고유 값; 주어진 순서 유지.
        train_scores: 설정별 학습 분류 점수 평균.
        validation_scores: 동일 split/CV의 검증 분류 점수 평균.
        train_deviations: 선택 학습 점수 변동; 비음수이며 동일 길이.
        validation_deviations: 선택 검증 점수 변동; 비음수이며 동일 길이.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        SelectionValidationCurveOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    raw, removed = paired(x=x_values, train=train_scores, validation=validation_scores)
    raw["train"] = vector(raw.train, allow_missing=False)[0]
    raw["validation"] = vector(raw.validation, allow_missing=False)[0]
    if removed or raw.x.duplicated().any() or profile.error_multiplier < 0:
        raise ValueError("고유 x값/오차 배율 오류")
    for name, errors in [
        ("train_deviation", train_deviations),
        ("validation_deviation", validation_deviations),
    ]:
        if errors is None:
            raw[name] = 0.0
        else:
            a, _ = vector(errors, allow_missing=False)
            if len(a) != len(raw) or (a < 0).any():
                raise ValueError("오차 길이/음수 오류")
            raw[name] = a
    positions = (
        vector(raw.x, allow_missing=False)[0]
        if pd.api.types.is_numeric_dtype(raw.x)
        else np.arange(len(raw))
    )
    for name, color in [("train", profile.color), ("validation", "C3")]:
        score = raw[name].to_numpy()
        error = raw[name + "_deviation"].to_numpy() * profile.error_multiplier
        ax.plot(positions, score, "o-", color=color, label=name, alpha=profile.alpha)
        ax.fill_between(
            positions, score - error, score + error, color=color, alpha=0.15
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(raw.x.astype(str), rotation=45, ha="right")
    ax.set_xlabel(profile.x_label)
    ax.set_ylabel(profile.score_label)
    ax.legend(fontsize=8)
    data = raw
    diagnostics = {
        "train_validation_gap": (raw.train - raw.validation).tolist(),
        "x_spacing": (
            "original numeric values"
            if pd.api.types.is_numeric_dtype(raw.x)
            else "categorical equal spacing"
        ),
    }
    notes = [
        "같은 split/CV·평가 지표의 사전 계산 값만 비교합니다. 표시 오차는 자동 CI가 아니며 테스트 자료로 설정을 선택하지 마세요."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        SelectionValidationCurveOutputDTO,
        "Selection — Validation Curve",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_validation_curve(
    x_values,
    train_scores,
    validation_scores,
    train_deviations=None,
    validation_deviations=None,
    *,
    ax,
    profile=None,
    recipe_path=None,
    overrides=None,
) -> SelectionValidationCurveOutputDTO:
    """학습량/분류 설정 변화의 train–validation 차이로 과적합과 개선 여지를 검토합니다.

    Args:
        x_values: 학습량/모델 설정의 고유 값; 주어진 순서 유지.
        train_scores: 설정별 학습 분류 점수 평균.
        validation_scores: 동일 split/CV의 검증 분류 점수 평균.
        train_deviations: 선택 학습 점수 변동; 비음수이며 동일 길이.
        validation_deviations: 선택 검증 점수 변동; 비음수이며 동일 길이.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        SelectionValidationCurveOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-selection-validation-curve",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax)
    return _draw(
        x_values,
        train_scores,
        validation_scores,
        train_deviations,
        validation_deviations,
        ax,
        profile,
        before,
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.selection에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_validation_curve(..., ax=axes[0, 0]); result.axes is axes[0, 0]


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("selection.validation_curve", plot_validation_curve)
