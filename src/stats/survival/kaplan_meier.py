"""
================================================================================
kaplan_meier.py - 우측 중도절단 자료의 Kaplan-Meier 생존함수 추정
================================================================================
설명:
    - 기존 Profile·JSON 레시피·공개 실행기·전용 DTO 계약을 사용합니다.
분석 가정:
    - 독립 관측과 비정보성 우측 중도절단. 모든 대상은 시각 0부터 관측합니다.
    - 동일 시각의 사건·절단은 모두 직전 위험집합에 포함합니다.
    - 좌측/구간 절단, 지연 진입, 가중치, 경쟁위험은 지원하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-20
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-20  김재현(Finn)     v1.0.0    승인된 생존분석 모듈·Greenwood 신뢰구간 추가
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
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import norm
from src.output import OutputDTO
from src.Recipes import resolve_profile


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class KaplanMeierProfile:
    """입력 설정. 관측 자료는 설정 JSON과 분리합니다.

    Attributes:
        label (str): 집단 이름. 공백이 아닌 문자열, 기본 'sample'.
        alpha (float): 점별 log-log Greenwood 신뢰구간 유의수준, 기본 0.05.
        evaluation_times (Optional[List[float]]): 조회할 시각. 기본 None.
            유한한 비음수 값이며 최대 관측시간 이하. 입력 순서·중복을 보존합니다.
    """
    alpha: float = 0.05
    label: str = "sample"
    evaluation_times: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class KaplanMeierOutputDTO(OutputDTO):
    """생존 추정 결과. 모든 표 배열은 times의 오름차순과 대응합니다.

    Attributes:
        method (str): 추정 방법 이름.
        parameters (Dict[str, Any]): 적용한 Profile 전체.
        sample_size (int): 관측 대상 수.
        times (np.ndarray): 사건 또는 절단이 발생한 고유 관측 시각.
        at_risk (np.ndarray): 해당 시각 직전 위험집합 수.
        events (np.ndarray): 해당 시각 사건 수.
        censored (np.ndarray): 해당 시각 중도절단 수.
        survival (np.ndarray): 사건 반영 후 우연속 S(t)=P(T>t) 추정값.
        confidence_lower (np.ndarray): times에 대응하는 점별 신뢰구간 하한.
        confidence_upper (np.ndarray): times에 대응하는 점별 신뢰구간 상한.
        cumulative_failure (np.ndarray): 1-S(t). 경쟁위험 누적발생함수가 아닙니다.
        median_survival_time (Optional[float]): S(t)<=0.5인 최초 시각.
            관찰 중 미도달이면 None이며 JSON null로 저장합니다.
        evaluation_times (Optional[np.ndarray]): 요청 조회 시각, 미요청 None.
        survival_at_times (Optional[np.ndarray]): 요청 시각의 S(t), 미요청 None.
        warnings (List[str]): 해석과 지원 범위의 제한.
    """
    method: str
    parameters: Dict[str, Any]
    sample_size: int
    times: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    censored: np.ndarray
    survival: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    cumulative_failure: np.ndarray
    median_survival_time: Optional[float]
    evaluation_times: Optional[np.ndarray]
    survival_at_times: Optional[np.ndarray]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate_inputs(durations, event_observed, profile):
    """관측시간·사건 지표·설정을 검증합니다.

    Args:
        durations: 결측 없는 비음수 유한 수치형 1차원 관측시간.
        event_observed: 동일 길이의 0/1 또는 bool. 사건=1, 절단=0.
        profile: KaplanMeierProfile.
    Returns:
        tuple: 정규화한 시간, 사건 지표, 선택 조회 시각.
    Raises:
        TypeError: Profile 또는 배열 자료형 오류.
        ValueError: 차원·길이·값·조회 범위 오류.
    """
    if not isinstance(profile, KaplanMeierProfile):
        raise TypeError("KaplanMeierProfile이 필요합니다.")
    if isinstance(profile.alpha, bool) or not isinstance(profile.alpha, (int, float)) or not 0 < profile.alpha < 1:
        raise ValueError("alpha는 0과 1 사이여야 합니다.")
    if not isinstance(profile.label, str) or not profile.label.strip():
        raise ValueError("label은 공백이 아닌 문자열이어야 합니다.")
    t, e = np.asarray(durations), np.asarray(event_observed)
    if t.dtype.kind not in "iuf" or e.dtype.kind not in "biuf":
        raise TypeError("시간은 수치형, 사건 지표는 수치형 또는 bool이어야 합니다.")
    if t.ndim != 1 or e.ndim != 1 or t.size == 0 or t.size != e.size:
        raise ValueError("비어 있지 않은 동일 길이의 1차원 배열이 필요합니다.")
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("시간은 비음수 유한값이어야 합니다.")
    if not np.isfinite(e).all() or not np.isin(e, [0, 1]).all():
        raise ValueError("사건 지표는 0 또는 1이어야 합니다.")
    query = None
    if profile.evaluation_times is not None:
        query = np.asarray(profile.evaluation_times)
        if query.dtype.kind not in "iuf":
            raise TypeError("조회 시각은 수치형이어야 합니다.")
        if query.ndim != 1 or not np.isfinite(query).all():
            raise ValueError("조회 시각은 유한한 1차원 배열이어야 합니다.")
        if ((query < 0) | (query > t.max())).any():
            raise ValueError("조회 시각은 0 이상 최대 관측시간 이하여야 합니다.")
        query = query.astype(float)
    return t.astype(float), e.astype(int), query


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _fit_kaplan_meier(t, e):
    """고유 관측 시각별 위험집합과 곱한계 추정값을 계산합니다.

    Args:
        t: 검증된 관측시간 배열.
        e: 검증된 사건 지표 배열.
    Returns:
        tuple: 시각, 위험집합, 사건, 절단, 생존 추정값 배열.
    Raises:
        ValueError: 빈 배열. 공개 실행기의 검증을 거쳐 사용합니다.
    """
    times, inverse, removed = np.unique(t, return_inverse=True, return_counts=True)
    events = np.bincount(inverse, weights=e, minlength=len(times)).astype(int)
    censored = removed - events
    at_risk = len(t) - np.r_[0, np.cumsum(removed[:-1])]
    survival = np.cumprod(1.0 - events / at_risk)
    return times, at_risk, events, censored, survival


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_kaplan_meier(durations, event_observed,
                    profile: Optional[KaplanMeierProfile] = None,
                    recipe_path: Optional[str] = None,
                    overrides: Optional[Dict[str, Any]] = None) -> KaplanMeierOutputDTO:
    """JSON 또는 명시적 Profile로 생존함수를 추정합니다.

    Args:
        durations: 대상별 관측시간. 같은 단위를 사용합니다.
        event_observed: 사건=1, 우측 중도절단=0. 행 순서는 durations와 동일.
        profile: 명시적 설정. 지정하면 recipe_path/overrides와 함께 사용할 수 없습니다.
        recipe_path: 사용자 레시피. 미지정 시 등록된 기본 JSON.
        overrides: JSON 위에 적용할 일회성 설정.
    Returns:
        KaplanMeierOutputDTO: 위험집합·생존함수·중앙수명·선택 시각 추정값.
    Raises:
        TypeError/ValueError: 입력·설정·혼합 호출 오류.
        FileNotFoundError/OSError: 레시피 파일 읽기 실패.
    Notes:
        추정 전용입니다. p-value나 H0/H1을 임의로 생성하지 않습니다.
        신뢰구간은 점별 구간이며 동시 신뢰대역이 아닙니다. RMST·검정·Cox는 별도 모듈입니다.
    """
    p = resolve_profile("survival-kaplan-meier", profile=profile,
                        recipe_path=recipe_path, overrides=overrides)
    t, e, query = _validate_inputs(durations, event_observed, p)
    times, risk, events, censored, survival = _fit_kaplan_meier(t, e)
    reached = np.flatnonzero(survival <= 0.5)
    median = float(times[reached[0]]) if reached.size else None
    values = None
    if query is not None:
        positions = np.searchsorted(times, query, side="right")
        values = np.r_[1.0, survival][positions]
    increments = np.zeros(len(times))
    mask = (events > 0) & (risk > events)
    increments[mask] = events[mask] / (risk[mask] * (risk[mask] - events[mask]))
    greenwood = np.cumsum(increments)
    lower, upper = survival.copy(), survival.copy()
    interior = (survival > 0) & (survival < 1)
    log_s = np.log(survival[interior])
    se = np.sqrt(greenwood[interior]) / np.abs(log_s)
    z = norm.ppf(1 - p.alpha / 2)
    lower[interior] = np.exp(-np.exp(np.log(-log_s) + z * se))
    upper[interior] = np.exp(-np.exp(np.log(-log_s) - z * se))
    notes = ["독립 관측과 비정보성 우측 중도절단을 가정합니다."]
    if median is None:
        notes.append("관찰 기간에 중앙수명에 도달하지 않았습니다.")
    return KaplanMeierOutputDTO(
        "Kaplan-Meier", asdict(p), len(t), times, risk, events, censored,
        survival, lower, upper, 1.0 - survival, median, query, values, notes)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 정식 이관 후 기본 registry/JSON을 사용하는 실행 예제입니다.
    result = run_kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 0, 1],
                             overrides={"evaluation_times": [0, 1, 2, 3, 4]})
    print(result.to_json(indent=2))
