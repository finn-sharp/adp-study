"""
================================================================================
observed_predicted.py - 원 단위 관측–예측과 R²·RMSE
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
class ObservedPredictedProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목; 기본은 진단 pane 이름. 기본 ''.
        color (str): 선/표식 색. 기본 'C0'.
        alpha (float): 표식 투명도 [0,1]. 기본 0.8.
        sample_role (str): train/validation/test 출처 선언. 기본 'test'.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    sample_role: str = "test"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ObservedPredictedOutputDTO(PlotOutputDTO):
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
    validate_profile(profile, ObservedPredictedProfile)
    if profile.sample_role not in ("train", "validation", "test"):
        raise ValueError("표본 역할 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(observed, predictions, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        observed: 전용 명세의 observed 입력.
        predictions: 전용 명세의 predictions 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        ObservedPredictedOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    before = start(ax)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    raw, removed = paired(observed=observed, predicted=predictions)
    raw["observed"] = vector(raw.observed, allow_missing=False)[0]
    raw["predicted"] = vector(raw.predicted, allow_missing=False)[0]
    if len(raw) < 2:
        raise ValueError("R² 비교 관측 수 부족")
    a = raw.observed.to_numpy()
    b = raw.predicted.to_numpy()
    ax.scatter(a, b, color=profile.color, alpha=profile.alpha, s=16)
    low = min(a.min(), b.min())
    high = max(a.max(), b.max())
    ax.plot([low, high], [low, high], "--", color="C3")
    ax.set_xlabel("Observed response (original units)")
    ax.set_ylabel("Predicted response (original units)")
    data = raw
    diagnostics = {
        "n": len(a),
        "missing_removed": removed,
        "sample_role": profile.sample_role,
        "rmse": float(np.sqrt(mean_squared_error(a, b))),
        "mae": float(mean_absolute_error(a, b)),
        "r2": float(r2_score(a, b)),
    }
    notes = [
        "identity 기준선입니다. 입력은 같은 y 단위·대응 행이어야 하며 train 점수는 일반화 성능이 아닙니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        ObservedPredictedOutputDTO,
        "Regression — observed predicted",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def plot_observed_predicted(
    observed, predictions, *, ax, profile=None, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        observed: 전용 명세의 observed 입력.
        predictions: 전용 명세의 predictions 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        ObservedPredictedOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "plot-regression-observed-predicted",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    return _run(observed, predictions, ax, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("regression.observed_predicted", plot_observed_predicted)
