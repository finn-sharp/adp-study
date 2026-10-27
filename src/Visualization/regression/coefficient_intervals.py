"""
================================================================================
coefficient_intervals.py - 회귀계수·t·p·신뢰구간
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
class CoefficientIntervalsProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
        color (str): 선/표식 색. 기본 'C0'.
        alpha (float): 표식 투명도 [0,1]. 기본 0.8.
        confidence (float): 계수 구간 신뢰수준 (0,1). 기본 0.95.
        include_intercept (bool): 절편을 함께 표시할지 여부. 기본 False.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    confidence: float = 0.95
    include_intercept: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CoefficientIntervalsOutputDTO(PlotOutputDTO):
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
    validate_profile(profile, CoefficientIntervalsProfile)
    if not 0 < profile.confidence < 1:
        raise ValueError("신뢰수준 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(model, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        CoefficientIntervalsOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    before = start(ax)
    result = ols_result(model)
    interval = np.asarray(result.conf_int(alpha=1 - profile.confidence))
    data = pd.DataFrame(
        {
            "term": result.model.exog_names,
            "coefficient": np.asarray(result.params),
            "standard_error": np.asarray(result.bse),
            "t_statistic": np.asarray(result.tvalues),
            "p_value": np.asarray(result.pvalues),
            "lower": interval[:, 0],
            "upper": interval[:, 1],
        }
    )
    shown = data if profile.include_intercept else data.loc[data.term != "Intercept"]
    if shown.empty:
        raise ValueError("표시할 계수가 없습니다.")
    pos = np.arange(len(shown))
    ax.errorbar(
        shown.coefficient,
        pos,
        xerr=np.vstack(
            [shown.coefficient - shown.lower, shown.upper - shown.coefficient]
        ),
        fmt="o",
        color=profile.color,
        alpha=profile.alpha,
        capsize=3,
    )
    ax.set_yticks(pos)
    ax.set_yticklabels(shown.term)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Coefficient with conditional interval")
    diagnostics = {
        "confidence": profile.confidence,
        "cov_type": result.cov_type,
        "df_resid": float(result.df_resid),
    }
    notes = [
        "계수는 변환 Feature/y 좌표계입니다. 규모를 비교하려면 단위·표준화를 확인하고 선택 후 구간을 무조건적 유의성으로 읽지 마세요."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        CoefficientIntervalsOutputDTO,
        "Regression — coefficient intervals",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def plot_coefficient_intervals(
    model, *, ax, profile=None, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        CoefficientIntervalsOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "plot-regression-coefficient-intervals",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    return _run(model, ax, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("regression.coefficient_intervals", plot_coefficient_intervals)
