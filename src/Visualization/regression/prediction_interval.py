"""
================================================================================
prediction_interval.py - 평균 CI와 새 관측 PI의 분리 표시
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 그리기 함수는 사용자 ax만 사용하며 모형을 다시 적합하거나 관측을 삭제하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
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
class PredictionIntervalProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
        color (str): 선/표식 색. 기본 'C0'.
        alpha (float): 표식 투명도 [0,1]. 기본 0.8.
        confidence (float): 평균 CI/새 관측 PI 신뢰수준. 기본 0.95.
        max_points (int): 표시 행 한도; 반환은 전체 새 행. 기본 200.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    confidence: float = 0.95
    max_points: int = 200


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class PredictionIntervalOutputDTO(PlotOutputDTO):
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
    validate_profile(profile, PredictionIntervalProfile)
    if not 0 < profile.confidence < 1 or profile.max_points < 1:
        raise ValueError("구간 설정 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(model, X_new, observed, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        X_new: 전용 명세의 X_new 입력.
        observed: 전용 명세의 observed 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        PredictionIntervalOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    before = start(ax)
    result = ols_result(model)
    raw, _ = numeric_data(X_new)
    expected = [name for name in result.model.exog_names if name != "Intercept"]
    if raw.columns.tolist() != expected:
        raise ValueError("새 설계 Feature 이름/순서가 다릅니다.")
    design = raw.copy()
    if "Intercept" in result.model.exog_names:
        design.insert(0, "Intercept", 1.0)
    data = result.get_prediction(design).summary_frame(alpha=1 - profile.confidence)
    data.insert(0, "position", np.arange(len(data)))
    if observed is not None:
        truth, removed = vector(observed, allow_missing=False)
        if len(truth) != len(data):
            raise ValueError("새 참값 길이 오류")
        data["observed"] = truth
    shown = data.iloc[: profile.max_points]
    x = shown.position.to_numpy()
    ax.errorbar(
        x,
        shown["mean"],
        yerr=np.vstack(
            [shown["mean"] - shown.obs_ci_lower, shown.obs_ci_upper - shown["mean"]]
        ),
        fmt="o",
        markersize=3,
        color=profile.color,
        alpha=profile.alpha,
        capsize=1,
        label="Prediction interval",
    )
    ax.errorbar(
        x,
        shown["mean"],
        yerr=np.vstack(
            [shown["mean"] - shown.mean_ci_lower, shown.mean_ci_upper - shown["mean"]]
        ),
        fmt="none",
        color="C3",
        capsize=2,
        label="Mean confidence interval",
    )
    if observed is not None:
        ax.scatter(x, shown.observed, color="black", s=10, label="Observed")
    ax.set_xlabel("New observation position")
    ax.set_ylabel("Response in fitted model scale")
    ax.legend(fontsize=8)
    diagnostics = {
        "confidence": profile.confidence,
        "n_new": len(data),
        "displayed": len(shown),
        "cov_type": result.cov_type,
    }
    notes = [
        "평균 CI와 관측 PI를 구별합니다. 좌표계는 적합 y와 같고 로그 inverse/smearing을 자동 적용하지 않습니다. robust 공분산의 관측 PI도 일정 오차분산 근사임을 확인하세요."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        PredictionIntervalOutputDTO,
        "Regression — prediction interval",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def plot_prediction_interval(
    model, X_new, observed=None, *, ax, profile=None, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        X_new: 전용 명세의 X_new 입력.
        observed: 전용 명세의 observed 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        PredictionIntervalOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "plot-regression-prediction-interval",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    return _run(model, X_new, observed, ax, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
