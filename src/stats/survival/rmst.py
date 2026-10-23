"""
================================================================================
rmst.py - 제한평균생존시간 추정
================================================================================
설명:
    - 기법별 Profile·JSON 레시피·공개 실행기·전용 Output DTO를 제공합니다.
분석 가정:
    - 독립 관측·비정보성 우측 중도절단. tau는 자료를 보기 전에 지정합니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-20
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-20  김재현(Finn)     v1.0.0    도메인 공통화·전용 명세·실행 예제 추가
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
from typing import Optional, Dict, Any, List
import numpy as np
from src.output import OutputDTO
from src.Recipes import resolve_profile
from .kaplan_meier import run_kaplan_meier, KaplanMeierProfile


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class RMSTProfile:
    """Attributes:
    tau (float): 적분 상한, 기본 24. 0<tau<=최대 관측시간.
    """

    tau: float = 24.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class RMSTOutputDTO(OutputDTO):
    """Attributes:
    method (str): Kaplan-Meier step integral.
    tau (float): 사용한 공통 적분 상한.
    estimate (float): 0~tau 생존곡선 아래 면적, 시간과 같은 단위.
    warnings (List[str]): 추론·외삽 제한.
    """

    method: str
    tau: float
    estimate: float
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate_inputs(profile):
    """Args: profile: RMSTProfile.
    Returns: None.
    Raises: TypeError/ValueError: 잘못된 Profile 또는 양수가 아닌 유한 tau.
    """
    if not isinstance(profile, RMSTProfile):
        raise TypeError("RMSTProfile이 필요합니다.")
    if (
        isinstance(profile.tau, bool)
        or not isinstance(profile.tau, (int, float))
        or not np.isfinite(profile.tau)
        or profile.tau <= 0
    ):
        raise ValueError("tau는 양의 유한값이어야 합니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _integrate(km, tau):
    """Args: km: 검증된 KM DTO; tau: 관측 범위 내 적분 상한.
    Returns: float: 우연속 계단함수의 정확한 면적.
    Raises: ValueError: 최대 관측시간을 넘은 tau.
    """
    if tau > km.times[-1]:
        raise ValueError("최대 관측시간 밖 RMST는 외삽하지 않습니다.")
    knots = np.unique(np.r_[0.0, km.times[km.times < tau], tau])
    positions = np.searchsorted(km.times, knots[:-1], side="right")
    return float(np.dot(np.diff(knots), np.r_[1.0, km.survival][positions]))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_rmst(
    durations,
    event_observed,
    profile: Optional[RMSTProfile] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RMSTOutputDTO:
    """Args:
        durations: 대상별 관측시간, 1차원 비음수 유한 수치.
        event_observed: 같은 길이의 사건1/절단0.
        profile: RMSTProfile. JSON 옵션과 동시 지정 불가.
        recipe_path: 사용자 JSON, 기본 survival-rmst.json.
        overrides: 일회성 설정.
    Returns: RMSTOutputDTO: 기술적 제한평균. p값·신뢰구간은 제공하지 않음.
    Raises: TypeError/ValueError: 입력·설정·외삽 오류. OSError: JSON 읽기 오류.
    """
    p = resolve_profile(
        "survival-rmst", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_inputs(p)
    km = run_kaplan_meier(durations, event_observed, KaplanMeierProfile())
    return RMSTOutputDTO(
        "Kaplan-Meier step integral",
        float(p.tau),
        _integrate(km, p.tau),
        ["기술통계이며 집단 차이의 유의성 또는 인과 효과가 아닙니다."],
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(run_rmst([10, 20, 30], [1, 0, 1], overrides={"tau": 24}).to_json(indent=2))
