"""
================================================================================
qq.py - 꼬리·분포 가정의 이탈을 점검해 통계 검정과 변환을 선택합니다.
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
class NumericQqProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        distribution (str): SciPy 연속분포 이름; 기본 norm. 기본 'norm'.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    distribution: str = "norm"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class NumericQqOutputDTO(PlotOutputDTO):
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
        profile: NumericQqProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, NumericQqProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(values, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        NumericQqOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    from scipy import stats

    a, removed = vector(values, min_size=3)
    distribution = getattr(stats, profile.distribution, None)
    if not isinstance(distribution, stats.rv_continuous):
        raise ValueError("연속분포 이름이 필요합니다.")
    (theoretical, ordered), (slope, intercept, r) = stats.probplot(
        a, dist=distribution, fit=True
    )
    ax.scatter(theoretical, ordered, color=profile.color, alpha=profile.alpha, s=12)
    ax.plot(theoretical, slope * theoretical + intercept, color="C3", linestyle="--")
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Observed quantile")
    data = pd.DataFrame({"theoretical": theoretical, "observed": ordered})
    diagnostics = {
        "n": len(a),
        "missing_removed": removed,
        "reference_slope": float(slope),
        "reference_intercept": float(intercept),
        "quantile_r": float(r),
    }
    notes = [
        "기준선은 분위수 비교용이며 회귀 모델 결과 또는 정규성 검정 p 값이 아닙니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        NumericQqOutputDTO,
        "Numeric — Qq",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_qq(
    values, *, ax, profile=None, recipe_path=None, overrides=None
) -> NumericQqOutputDTO:
    """꼬리·분포 가정의 이탈을 점검해 통계 검정과 변환을 선택합니다.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        NumericQqOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-numeric-qq", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    before = start(ax)
    return _draw(values, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.numeric에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_qq(..., ax=axes[0, 0]); result.axes is axes[0, 0]


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("numeric.qq", plot_qq)
