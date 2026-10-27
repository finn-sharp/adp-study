"""
================================================================================
glm_residuals.py - 분포별 GLM 적합평균과 Pearson/deviance 잔차
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 호출한 ax에만 그리며 분포별 잔차를 OLS 잔차로 혼동하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-23
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-23  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
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
class GLMResidualsProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목 기본 ''.
        color (str): 점 색상 기본 'C0'.
        alpha (float): 투명도 [0,1] 기본 0.6.
        residual (str): pearson/deviance 잔차 선택 기본 'pearson'.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.6
    residual: str = "pearson"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class GLMResidualsOutputDTO(PlotOutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        figure (Any): 원 Figure; JSON 제외.
        axes (Any): 동일 ax; JSON 제외.
        artists (List[Any]): 추가 artist; JSON 제외.
        data (pd.DataFrame): 적합 GLM의 평균·선택 잔차·leverage·Cook·원행.
        diagnostics (Dict[str,Any]): family·link·선택 잔차·조건부 분산 요약.
        parameters (Dict[str,Any]): 적용 Profile.
        warnings (List[str]): 잔차 정규성의 자동 요구·영향값 자동 삭제 금지.
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
    validate_plot_profile(profile, GLMResidualsProfile)
    if profile.residual not in ("pearson", "deviance"):
        raise ValueError("잔차 종류 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(result, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        result: 전용 명세의 result 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        GLMResidualsOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    if (
        not hasattr(result, "diagnostics")
        or not hasattr(result, "fit_metrics")
        or result.fit_metrics.get("family") not in ("logistic", "gamma", "poisson")
    ):
        raise TypeError("전용 GLM DTO가 필요합니다.")
    data = result.diagnostics.copy()
    column = profile.residual + "_residual"
    before = start(ax)
    ax.scatter(data.fitted_mean, data[column], color=profile.color, alpha=profile.alpha)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="Fitted conditional mean", ylabel=profile.residual + " residual")
    return finish(
        ax,
        before,
        data,
        dict(
            family=result.fit_metrics["family"],
            link=result.fit_metrics["link"],
            residual=profile.residual,
            pearson_dispersion=result.fit_metrics["pearson_dispersion"],
        ),
        profile,
        GLMResidualsOutputDTO,
        "Training GLM residuals",
        [
            "이항/Poisson/Gamma 잔차에 OLS 정규성을 일괄 요구하지 않습니다. 분포·평균 링크·분산과 영향도를 함께 검토하세요."
        ],
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def plot_glm_residuals(result, *, ax, profile=None, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        result: 전용 명세의 result 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        GLMResidualsOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "plot-regression-glm-residuals",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    return _run(result, ax, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
