"""
================================================================================
calibration.py - 확률의 과신·과소신과 실제 발생률의 일치를 점검합니다.
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
class ClassificationCalibrationProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        positive_label (Any): probabilities가 의미하는 양성 라벨. 기본 1.
        n_bins (int): 양의 동일 폭 확률 구간 수. 기본 10.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    positive_label: Any = 1
    n_bins: int = 10


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ClassificationCalibrationOutputDTO(PlotOutputDTO):
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
        profile: ClassificationCalibrationProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, ClassificationCalibrationProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(y_true, probabilities, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        y_true: 대응 참조 분류 라벨; 이진 score 차트는 정확히 두 클래스.
        probabilities: positive_label의 [0,1] 확률; 같은 행의 y_true와 대응.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        ClassificationCalibrationOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    y, s, removed = binary_scores(
        y_true, probabilities, profile.positive_label, probability=True
    )
    if profile.n_bins < 1:
        raise ValueError("n_bins 오류")
    edges = np.linspace(0, 1, profile.n_bins + 1)
    ids = np.minimum(np.searchsorted(edges, s, side="right") - 1, profile.n_bins - 1)
    rows = []
    for i in range(profile.n_bins):
        mask = ids == i
        if mask.any():
            rows.append(
                dict(
                    bin=i,
                    left=edges[i],
                    right=edges[i + 1],
                    n=int(mask.sum()),
                    mean_probability=float(s[mask].mean()),
                    positive_fraction=float(y[mask].mean()),
                )
            )
    data = pd.DataFrame(rows)
    ax.plot(
        data.mean_probability,
        data.positive_fraction,
        "o-",
        color=profile.color,
        alpha=profile.alpha,
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    diagnostics = {
        "brier_score": float(np.mean((s - y) ** 2)),
        "ece": float(
            np.sum(data.n * np.abs(data.mean_probability - data.positive_fraction))
            / len(y)
        ),
        "n": len(y),
        "missing_removed": removed,
    }
    notes = [
        "빈 구간은 표시하지 않으며 n을 반환합니다. ECE는 구간 정의에 의존하고 보류 확률을 사용해야 합니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        ClassificationCalibrationOutputDTO,
        "Classification — Calibration",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_calibration(
    y_true, probabilities, *, ax, profile=None, recipe_path=None, overrides=None
) -> ClassificationCalibrationOutputDTO:
    """확률의 과신·과소신과 실제 발생률의 일치를 점검합니다.

    Args:
        y_true: 대응 참조 분류 라벨; 이진 score 차트는 정확히 두 클래스.
        probabilities: positive_label의 [0,1] 확률; 같은 행의 y_true와 대응.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        ClassificationCalibrationOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-classification-calibration",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax)
    return _draw(y_true, probabilities, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.classification에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_calibration(..., ax=axes[0, 0]); result.axes is axes[0, 0]


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("classification.calibration", plot_calibration)
