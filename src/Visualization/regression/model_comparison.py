"""
================================================================================
model_comparison.py - 기준·전처리·대안 회귀의 검증 점수 비교
================================================================================
설명:
    - 직접 실행(__main__) 시 고정 합성 입력으로 PNG 예시를 저장합니다.
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 호출자가 만든 단일 ax만 사용합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-23
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    직접 실행 PNG 예시 저장 추가
    2026-10-23  김재현(Finn)     v1.0.0    응용 분석 모듈·모의10회 실행 연결
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
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.Visualization._common import (
    PlotOutputDTO,
    start,
    finish,
    validate_profile as validate_plot_profile,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class RegressionModelComparisonProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목 기본 ''.
        color (str): 검증 점수 색 기본 'C0'.
        alpha (float): 표시 투명도 [0,1] 기본 0.8.
        metric (str): 동일 검증 행·원단위 점수 열 이름 기본 'cv_r2'.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    metric: str = "cv_r2"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class RegressionModelComparisonOutputDTO(PlotOutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        figure (Any): 원 Figure; runtime 제외.
        axes (Any): 동일 ax; runtime 제외.
        artists (List[Any]): 추가 artist; runtime 제외.
        data (pd.DataFrame): 입력 비교표 복사, 실제 점수 유지.
        diagnostics (Dict[str,Any]): 점수 이름·행 수; 선택/통계 검정을 대신하지 않음.
        parameters (Dict[str,Any]): Profile 전체.
        warnings (List[str]): 다른 응답 척도·분할 점수 혼용 금지.
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
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
    Returns:
        object: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    validate_plot_profile(profile, RegressionModelComparisonProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(comparison, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        comparison: 전용 명세의 comparison 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        RegressionModelComparisonOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    data = pd.DataFrame(comparison).copy()
    if (
        "name" not in data
        or profile.metric not in data
        or data.empty
        or not np.isfinite(data[profile.metric].to_numpy(dtype=float)).all()
    ):
        raise ValueError("name 및 유한 점수 열이 필요합니다.")
    before = start(ax)
    ax.barh(
        data.name.astype(str),
        data[profile.metric],
        color=profile.color,
        alpha=profile.alpha,
    )
    ax.set_xlabel(profile.metric)
    return finish(
        ax,
        before,
        data,
        {"metric": profile.metric, "n_models": len(data)},
        profile,
        RegressionModelComparisonOutputDTO,
        "Regression CV comparison",
        [
            "같은 분할·같은 원 반응 단위의 점수만 함께 비교하세요. 그림이 모델 선택이나 검정을 자동 수행하지 않습니다."
        ],
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def plot_model_comparison(
    comparison, *, ax, profile=None, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        comparison: 전용 명세의 comparison 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        RegressionModelComparisonOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "plot-regression-model-comparison",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    return _run(comparison, ax, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("regression.model_comparison", plot_model_comparison)
