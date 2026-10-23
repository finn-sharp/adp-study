"""
================================================================================
threshold.py - 비용과 precision/recall trade-off를 근거로 검증 자료의 임계값 후보를 선택합니다.
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
class ClassificationThresholdProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        positive_label (Any): 큰 score가 의미하는 양성 라벨. 기본 1.
        false_positive_cost (float): 비음수 위양성 한 건 비용. 기본 1.0.
        false_negative_cost (float): 비음수 위음성 한 건 비용. 기본 1.0.
        max_thresholds (int): 전체 점수 후보의 표시 격자 한도, 2 이상. 기본 200.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    positive_label: Any = 1
    false_positive_cost: float = 1.0
    false_negative_cost: float = 1.0
    max_thresholds: int = 200


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ClassificationThresholdOutputDTO(PlotOutputDTO):
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
        profile: ClassificationThresholdProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, ClassificationThresholdProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(y_true, scores, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        y_true: 대응 참조 분류 라벨; 이진 score 차트는 정확히 두 클래스.
        scores: 큰 값이 positive_label인 유한 점수; 반드시 확률일 필요는 없음.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        ClassificationThresholdOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    y, s, removed = binary_scores(y_true, scores, profile.positive_label)
    if (
        min(profile.false_positive_cost, profile.false_negative_cost) < 0
        or profile.max_thresholds < 2
    ):
        raise ValueError("비용/격자 한도 오류")
    unique = np.unique(s)
    thresholds = np.r_[
        np.nextafter(unique.min(), -np.inf), unique, np.nextafter(unique.max(), np.inf)
    ]
    rows = []
    # 모든 고유 후보를 계산해 최소 비용 결정을 보존합니다. 표시만 제한합니다.
    order = np.argsort(s, kind="stable")
    ordered = s[order]
    cumulative = np.r_[0, np.cumsum(y[order])]
    total_positive = y.sum()
    total_negative = len(y) - total_positive
    for threshold in thresholds:
        left = np.searchsorted(ordered, threshold, side="left")
        fn = int(cumulative[left])
        tp = int(total_positive - fn)
        tn = int(left - fn)
        fp = int(total_negative - tn)
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / total_positive
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        cost = (
            fp * profile.false_positive_cost + fn * profile.false_negative_cost
        ) / len(y)
        rows.append(
            dict(
                threshold=threshold,
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                cost_per_sample=cost,
            )
        )
    data = pd.DataFrame(rows)
    display = np.unique(
        np.linspace(0, len(data) - 1, min(profile.max_thresholds, len(data))).astype(
            int
        )
    )
    shown = data.iloc[display]
    for name in ("precision", "recall", "f1"):
        ax.plot(shown.threshold, shown[name], label=name, alpha=profile.alpha)
    ax.set_xlabel("Decision threshold (score >= threshold)")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    best = data.loc[data.cost_per_sample.idxmin()]
    ax.axvline(
        best.threshold, color="black", linestyle="--", label="Minimum specified cost"
    )
    ax.legend(fontsize=8)
    diagnostics = {
        "n": len(y),
        "missing_removed": removed,
        "minimum_cost_threshold": float(best.threshold),
        "minimum_cost_per_sample": float(best.cost_per_sample),
        "tie_rule": "first ascending threshold",
        "n_candidates": len(data),
        "displayed_candidates": len(shown),
    }
    notes = [
        "이 자료에서 임계값을 선택하면 같은 자료의 성능은 선택 편향을 포함합니다. 선택용 검증 자료와 최종 테스트를 분리하세요. 비용은 평균 비용이며 통계 신뢰구간이 아닙니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        ClassificationThresholdOutputDTO,
        "Classification — Threshold",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_threshold(
    y_true, scores, *, ax, profile=None, recipe_path=None, overrides=None
) -> ClassificationThresholdOutputDTO:
    """비용과 precision/recall trade-off를 근거로 검증 자료의 임계값 후보를 선택합니다.

    Args:
        y_true: 대응 참조 분류 라벨; 이진 score 차트는 정확히 두 클래스.
        scores: 큰 값이 positive_label인 유한 점수; 반드시 확률일 필요는 없음.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        ClassificationThresholdOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-classification-threshold",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax)
    return _draw(y_true, scores, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.classification에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_threshold(..., ax=axes[0, 0]); result.axes is axes[0, 0]
