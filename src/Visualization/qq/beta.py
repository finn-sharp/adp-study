"""
================================================================================
beta.py - beta 분포 전용 Q-Q 비교
================================================================================
설명:
    - 직접 실행(__main__) 시 고정 합성 입력으로 PNG 예시를 저장합니다.
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 사용자가 전달한 ax에만 그립니다. 분포 fitting은 시각 진단이며 정식 p 검정이 아닙니다.
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
from scipy import stats
from src.Visualization._common import PlotOutputDTO, validate_profile, start, vector
from src.Visualization.qq._common import render_qq


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class QQBetaProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
        color (str): 관측 표식 색. 기본 'C0'.
        alpha (float): 표식 투명도 [0,1]. 기본 0.8.
        reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
        fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
        loc (float): 명시 위치; 양수 지지 분포 fit에서는 고정. 기본 0.0.
        scale (float): 양수 명시 척도. 기본 1.0.
        shape_a (float): 양수 beta a; fit에서는 추정. 기본 2.0.
        shape_b (float): 양수 beta b; fit에서는 추정. 기본 2.0.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    reference: str = "identity"
    fit_parameters: bool = False
    loc: float = 0.0
    scale: float = 1.0
    shape_a: float = 2.0
    shape_b: float = 2.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class QQBetaOutputDTO(PlotOutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        figure (Any): 원 ax Figure; runtime JSON 제외.
        axes (Any): 사용자가 전달한 동일 Axes; JSON 제외.
        artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
        data (pd.DataFrame): probability/theoretical/observed, 관측 오름차순.
        diagnostics (Dict[str,Any]): 실제 분포 인자·표본 수·지지 위반·기준선 정의.
        parameters (Dict[str,Any]): 전용 Profile 설정.
        warnings (List[str]): 분포·동률·같은 자료 fitting·해석 정보.
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
    validate_profile(profile, QQBetaProfile)
    if profile.scale <= 0 or profile.reference not in ("identity", "quartile", "none"):
        raise ValueError("QQ 척도/기준선 오류")
    if profile.shape_a <= 0:
        raise ValueError("shape_a는 양수입니다.")
    if profile.shape_b <= 0:
        raise ValueError("shape_b는 양수입니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _distribution(values, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        values: 전용 명세의 values 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        tuple: frozen 분포, 실제 사용 인자.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    a, removed = vector(values, min_size=3)
    if profile.fit_parameters and np.ptp(a) == 0:
        raise ValueError("상수 자료의 분포 fit은 퇴화합니다.")
    if profile.fit_parameters:
        if ((a <= profile.loc) | (a >= profile.loc + profile.scale)).any():
            raise ValueError("Beta fit은 열린 loc..loc+scale 안의 관측이 필요합니다.")
        shape_a, shape_b, loc, scale = stats.beta.fit(
            a, floc=profile.loc, fscale=profile.scale
        )
    else:
        shape_a, shape_b, loc, scale = (
            profile.shape_a,
            profile.shape_b,
            profile.loc,
            profile.scale,
        )
    params = {
        "a": float(shape_a),
        "b": float(shape_b),
        "loc": float(loc),
        "scale": float(scale),
    }
    if params["scale"] <= 0 or not all(np.isfinite(v) for v in params.values()):
        raise ValueError("추정 인자 오류")
    return stats.beta(**params), params


def _run(values, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        values: 전용 명세의 values 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        QQBetaOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    before = start(ax)
    distribution, params = _distribution(values, profile)
    return render_qq(
        values, distribution, params, profile, ax, before, QQBetaOutputDTO, "beta"
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def plot_beta(values, *, ax, profile=None, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        values: 전용 명세의 values 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        QQBetaOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "plot-qq-beta", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(values, ax, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("qq.beta", plot_beta)
