"""
================================================================================
binomial.py - binomial 이산 분포 전용 Q-Q 비교
================================================================================
설명:
    - 직접 실행(__main__) 시 고정 합성 입력으로 PNG 예시를 저장합니다.
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 이산 분포의 동률/계단 구조를 정상 구조로 보존합니다. p 검정이 아닙니다.
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
class QQBinomialProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목; 비우면 분포별 Q-Q 제목. 기본 ''.
        color (str): 관측 표식 색. 기본 'C0'.
        alpha (float): 표식 투명도 [0,1]. 기본 0.8.
        reference (str): identity/quartile/none 기준선; 자동 회귀 모형 아님. 기본 'identity'.
        fit_parameters (bool): 같은 관측에서 명시 분포 인자를 추정할지 여부. 기본 False.
        loc (int): 명시 정수 지지 위치. 기본 0.
        n_trials (int): 알려진 양수 시행 수; fit에서도 고정. 기본 10.
        success_probability (float): 성공 확률 [0,1]; fit에서 평균/n. 기본 0.5.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    reference: str = "identity"
    fit_parameters: bool = False
    loc: int = 0
    n_trials: int = 10
    success_probability: float = 0.5


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class QQBinomialOutputDTO(PlotOutputDTO):
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
    validate_profile(profile, QQBinomialProfile)
    if profile.reference not in ("identity", "quartile", "none"):
        raise ValueError("QQ 기준선 오류")
    if not 0 <= profile.success_probability <= 1:
        raise ValueError("확률 범위 오류")
    if profile.n_trials < 1:
        raise ValueError("시행 수 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _distribution(values, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        values: 전용 명세의 values 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        tuple: 명시 이산 frozen 분포, 사용 인자.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    a, removed = vector(values, min_size=3)
    if (a != np.floor(a)).any():
        raise ValueError("이산 QQ는 정수 관측을 사용하세요.")
    p = (
        float((a - profile.loc).mean() / profile.n_trials)
        if profile.fit_parameters
        else profile.success_probability
    )
    if not 0 <= p <= 1:
        raise ValueError("Binomial fit 지지 범위 오류")
    params = {"n": profile.n_trials, "p": p, "loc": profile.loc}
    return stats.binom(**params), params


def _run(values, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        values: 전용 명세의 values 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        QQBinomialOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    before = start(ax)
    distribution, params = _distribution(values, profile)
    return render_qq(
        values,
        distribution,
        params,
        profile,
        ax,
        before,
        QQBinomialOutputDTO,
        "binomial",
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def plot_binomial(values, *, ax, profile=None, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        values: 전용 명세의 values 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        QQBinomialOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "plot-qq-binomial", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(values, ax, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.


if __name__ == "__main__":
    from examples.visualization_gallery import save_example

    save_example("qq.binomial", plot_binomial)
