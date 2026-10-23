"""
================================================================================
logrank.py - 다집단 로그순위 검정과 Holm 쌍별 비교
================================================================================
설명:
    - 기법별 Profile·JSON 레시피·공개 실행기·전용 Output DTO를 제공합니다.
분석 가정:
    - 독립 집단·비정보성 우측 중도절단·충분한 사건 수. 교차 곡선에서 검정력이 낮을 수 있습니다.
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
from itertools import combinations
import numpy as np
from scipy.stats import chi2
from src.output import OutputDTO
from src.Recipes import resolve_profile
from .kaplan_meier import run_kaplan_meier, KaplanMeierProfile


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class LogrankProfile:
    """Attributes:
    alpha (float): 유의수준, 기본 0.05, 0<alpha<1.
    pairwise (bool): 모든 쌍 비교와 Holm 보정을 반환, 기본 False.
        전체 검정 기각 후 수행하는 절차는 호출자가 사전 규칙에 따라 제어합니다.
    """

    alpha: float = 0.05
    pairwise: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class LogrankOutputDTO(OutputDTO):
    """Attributes:
    statistic (float): (k-1)차원 초기하 공분산의 카이제곱 통계량.
    p_value (float): 점근 카이제곱 상측 p값.
    reject_h0 (bool): p_value<alpha.
    alpha (float): 실제 유의수준.
    degrees_of_freedom (int): 집단수-1.
    null_hypothesis (str): 모든 집단의 생존함수가 동일.
    alternative_hypothesis (str): 적어도 한 집단의 생존함수가 다름.
    groups (List[Any]): 정렬한 집단 라벨.
    pairwise (List[Dict[str, Any]]): 모든 쌍의 가설·통계량·원/보정 p값·판정.
    warnings (List[str]): 점근 검정·연관성 해석 제한.
    """

    statistic: float
    p_value: float
    reject_h0: bool
    alpha: float
    degrees_of_freedom: int
    null_hypothesis: str
    alternative_hypothesis: str
    groups: List[Any]
    pairwise: List[Dict[str, Any]]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate_inputs(t, e, groups, p):
    """Args: t/e: 시간·사건; groups: 동질 타입 집단 라벨; p: Profile.
    Returns: tuple: 검증한 시간·사건·그룹·고유 라벨.
    Raises: TypeError/ValueError: 입력·가설검정 가능 조건 오류.
    """
    if not isinstance(p, LogrankProfile):
        raise TypeError("LogrankProfile이 필요합니다.")
    if (
        isinstance(p.alpha, bool)
        or not isinstance(p.alpha, (int, float))
        or not 0 < p.alpha < 1
    ):
        raise ValueError("alpha 오류")
    if not isinstance(p.pairwise, bool):
        raise TypeError("pairwise는 bool입니다.")
    km = run_kaplan_meier(t, e, KaplanMeierProfile())
    import pandas as pd

    g = np.asarray(groups)
    if g.ndim != 1 or len(g) != km.sample_size or pd.isna(g).any():
        raise ValueError("groups 길이·결측 오류")
    levels = np.unique(g)
    if len(levels) < 2 or np.sum(e) == 0:
        raise ValueError("두 집단 이상과 사건이 필요합니다.")
    return np.asarray(t, float), np.asarray(e, int), g, levels


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _logrank(t, e, g, levels):
    """Args: t/e/g: 검증된 관측; levels: 집단 라벨.
    Returns: tuple: 통계량·p값. 위험집합 조건부 다변량 초기하 공분산 사용.
    Raises: ValueError: 정보 부족으로 공분산 rank가 k-1 미만.
    """
    k = len(levels)
    u = np.zeros(k)
    v = np.zeros((k, k))
    for time in np.unique(t[e == 1]):
        risk = np.array([np.sum((g == x) & (t >= time)) for x in levels], float)
        events = np.array(
            [np.sum((g == x) & (t == time) & (e == 1)) for x in levels], float
        )
        n, d = risk.sum(), events.sum()
        prop = risk / n
        u += events - d * prop
        if n > 1:
            v += d * (n - d) / (n - 1) * (np.diag(prop) - np.outer(prop, prop))
    cov = v[:-1, :-1]
    if np.linalg.matrix_rank(cov) < k - 1:
        raise ValueError("로그순위 검정 정보가 부족합니다.")
    statistic = max(0.0, float(u[:-1] @ np.linalg.solve(cov, u[:-1])))
    return statistic, float(chi2.sf(statistic, k - 1))


def _holm(values):
    """Args: values: 모든 사전 지정 비교의 p값.
    Returns: ndarray: 원 비교 순서의 Holm 보정 p값.
    Raises: ValueError: 비어 있는 목록은 사용하지 않음.
    """
    p = np.asarray(values)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    )
    return adjusted


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_logrank(
    durations,
    event_observed,
    groups,
    profile: Optional[LogrankProfile] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> LogrankOutputDTO:
    """Args:
        durations: 비음수 유한 1차원 관측시간.
        event_observed: 동일 길이 사건1/절단0.
        groups: 동일 길이의 결측 없는 동질 타입 집단 라벨.
        profile: 명시적 Profile. JSON 옵션과 혼용 불가.
        recipe_path: 사용자 JSON. 기본 survival-logrank.json.
        overrides: 일회성 설정.
    Returns: LogrankOutputDTO: 전체 검정 및 요청한 모든 쌍의 Holm 결과.
    Raises: TypeError/ValueError: 설정·입력·검정 정보 부족. OSError: JSON 오류.
    """
    p = resolve_profile(
        "survival-logrank", profile, recipe_path=recipe_path, overrides=overrides
    )
    t, e, g, levels = _validate_inputs(durations, event_observed, groups, p)
    statistic, pv = _logrank(t, e, g, levels)
    pairs = []
    if p.pairwise:
        for left, right in combinations(levels, 2):
            mask = np.isin(g, [left, right])
            s, q = _logrank(t[mask], e[mask], g[mask], [left, right])
            pairs.append(
                dict(
                    left=left,
                    right=right,
                    statistic=s,
                    p_value=q,
                    degrees_of_freedom=1,
                    null_hypothesis="S_left(t)=S_right(t) for all t",
                    alternative_hypothesis="different for some t",
                )
            )
        for row, adj in zip(pairs, _holm([r["p_value"] for r in pairs])):
            row.update(
                adjusted_p_value=float(adj),
                reject_h0=bool(adj < p.alpha),
                alpha=p.alpha,
            )
    return LogrankOutputDTO(
        statistic,
        pv,
        bool(pv < p.alpha),
        p.alpha,
        len(levels) - 1,
        "all group survival functions are equal for all t",
        "at least one survival function differs",
        levels.tolist(),
        pairs,
        [
            "점근 검정이며 비기각은 동일성의 증명이 아닙니다.",
            "비무작위 집단 차이는 인과 효과가 아닙니다.",
        ],
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        run_logrank(
            [1, 2, 3, 2, 4, 5], [1, 1, 0, 1, 0, 1], ["A"] * 3 + ["B"] * 3
        ).to_json(indent=2)
    )
