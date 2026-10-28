"""
================================================================================
residual_order.py - 잔차 관측/시간 순서 탐색
================================================================================
설명:
    - 직접 실행(__main__) 시 고정 합성 입력으로 PNG 예시를 저장합니다.
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 그리기 함수는 사용자 ax만 사용하며 모형을 다시 적합하거나 관측을 삭제하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    직접 실행 PNG 예시 저장 추가
    2026-10-12  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
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
    validate_profile,
    start,
    finish,
    paired,
    vector,
)
from src.stats.regression._common import ols_result, numeric_data
from statsmodels.stats.outliers_influence import OLSInfluence


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class ResidualOrderProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
        color (str): 선/표식 색. 기본 'C0'.
        alpha (float): 표식 투명도 [0,1]. 기본 0.8.
        ordered (bool): True는 시간 순·동일 간격임을 호출자가 명시. 기본 False.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    ordered: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ResidualOrderOutputDTO(PlotOutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        figure (Any): 원 ax Figure; runtime JSON 제외.
        axes (Any): 전달한 동일 Axes; JSON 제외.
        artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
        data (pd.DataFrame): 표시 수치·행 위치·원 label 또는 계수 표.
        diagnostics (Dict[str,Any]): 판단 기준·모형 좌표계·표본 수.
        parameters (Dict[str,Any]): 적용 Profile.
        warnings (List[str]): 진단 가정·선택 후 해석 정보.
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
    validate_profile(profile, ResidualOrderProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(model, order, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        order: 전용 명세의 order 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        ResidualOrderOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    before = start(ax)
    result = ols_result(model)
    residual = np.asarray(result.resid)
    x = np.arange(len(residual)) if order is None else np.asarray(order)
    if x.ndim != 1 or len(x) != len(residual) or pd.isna(x).any():
        raise ValueError("관측 순서 길이/결측 오류")
    if profile.ordered and not pd.Index(x).is_monotonic_increasing:
        raise ValueError("명시 시간 순서가 오름차순이 아닙니다.")
    ax.plot(x, residual, color=profile.color, alpha=profile.alpha)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel(
        "Time / ordered observations" if profile.ordered else "Input observation order"
    )
    ax.set_ylabel("Residual")
    data = pd.DataFrame({"order": x, "residual": residual})
    diagnostics = {"n": len(residual), "ordered": profile.ordered}
    notes = [
        "시간 순·간격이 명시되지 않으면 입력 행 패턴일 뿐입니다. 자기상관 또는 독립성의 자동 검정이 아닙니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        ResidualOrderOutputDTO,
        "Regression — residual order",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def plot_residual_order(
    model, order=None, *, ax, profile=None, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        order: 전용 명세의 order 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        ResidualOrderOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "plot-regression-residual-order",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    return _run(model, order, ax, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("regression.residual_order", plot_residual_order)
