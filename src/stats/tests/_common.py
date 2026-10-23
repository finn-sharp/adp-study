"""
================================================================================
_common.py - 통계 입력 검증과 수치 보조 함수
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 각 공개 모듈의 표본 설계·정의역·정확/근사 조건을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    통계 PDF 전체 범위 모듈화·입력 검증·명세
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
import inspect
import math
import warnings as pywarnings
import numpy as np
from scipy import stats
from scipy.special import gammaln
from src.stats.tests._validation import alpha


def number(value, name, positive=False):
    """유한한 실수 설정을 검증합니다.

    Args:
        value: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        name: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        positive: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(name + "는 유한한 실수여야 합니다.")
    if positive and value <= 0:
        raise ValueError(name + "는 양수여야 합니다.")
    return float(value)


def integer(value, name, minimum=0):
    """bool 제외 정수 조건을 검증합니다.

    Args:
        value: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        name: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        minimum: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(name + " 정수 범위를 확인하세요.")
    return int(value)


def numeric(x, ndim, minimum, policy="raise"):
    """shape·NaN 정책을 검증하고 2차원은 행 전체를 제거합니다.

    Args:
        x: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        ndim: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        minimum: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        policy: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    if np.iscomplexobj(x):
        raise ValueError("실수 자료가 필요합니다.")
    a = np.asarray(x, dtype=float)
    if a.ndim != ndim or (ndim == 2 and a.shape[1] == 0) or np.isinf(a).any():
        raise ValueError("자료 shape/무한대를 확인하세요.")
    if policy not in ("raise", "omit"):
        raise ValueError("nan_policy는 raise/omit입니다.")
    if np.isnan(a).any():
        if policy == "raise":
            raise ValueError("결측값이 있습니다.")
        a = a[~np.isnan(a)] if ndim == 1 else a[~np.isnan(a).any(axis=1)]
    if len(a) < minimum:
        raise ValueError("유효 관측 수가 부족합니다.")
    return a


def paired(x, y, policy):
    """대응 표본의 원 위치를 맞추고 쌍별 결측을 제거합니다.

    Args:
        x: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        y: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        policy: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    a = np.asarray(x)
    b = np.asarray(y)
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError("대응 x/y는 같은 길이의 1차원 자료입니다.")
    rows = numeric(np.column_stack([a, b]), 2, 2, policy)
    return rows[:, 0], rows[:, 1]


def table_data(x, square=False):
    """양의 주변합을 가진 정수 도수표를 검증합니다.

    Args:
        x: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        square: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    a = numeric(x, 2, 2)
    if (
        a.shape[1] < 2
        or (square and a.shape[0] != a.shape[1])
        or (a < 0).any()
        or (a != np.floor(a)).any()
    ):
        raise ValueError("정수 도수표 shape/값 오류")
    if (a.sum(axis=0) == 0).any() or (a.sum(axis=1) == 0).any():
        raise ValueError("주변합은 양수여야 합니다.")
    if a.sum() > 2**53:
        raise ValueError("정확하게 표현할 수 있는 총빈도를 초과합니다.")
    return a.astype(np.int64)


def tail(s, alt, distribution):
    """좌/우/양측 p를 안정적인 CDF/SF로 계산합니다.

    Args:
        s: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        alt: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        distribution: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    if alt == "less":
        return float(distribution.cdf(s))
    if alt == "greater":
        return float(distribution.sf(s))
    return float(min(1.0, 2 * distribution.sf(abs(s))))


def mean_interval(value, se, alpha_, alt, dist):
    """대립 방향과 일치하는 평균/차이 구간을 계산합니다.

    Args:
        value: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        se: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        alpha_: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        alt: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        dist: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    q = dist.ppf(1 - alpha_ / 2 if alt == "two-sided" else 1 - alpha_)
    if alt == "greater":
        return (float(value - q * se), float("inf"))
    if alt == "less":
        return (float("-inf"), float(value + q * se))
    return (float(value - q * se), float(value + q * se))


def result(
    key, p, s, pvalue, h0, method, extras, details=None, notes=None, reject=None
):
    """전용 DTO에 전달할 필드를 구성하고 유효 p를 확인합니다.

    Args:
        key: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        p: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        s: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        pvalue: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        h0: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        method: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        extras: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        details: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        notes: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        reject: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    if pvalue is not None and (not np.isfinite(pvalue) or not 0 <= pvalue <= 1):
        raise ValueError("미정의 p 값입니다.")
    return dict(
        test_type=key,
        statistic=float(s),
        p_value=None if pvalue is None else float(pvalue),
        reject_h0=bool(pvalue < p.alpha) if reject is None else bool(reject),
        alpha=float(p.alpha),
        null_hypothesis=h0,
        alternative=getattr(p, "alternative", "omnibus"),
        method_used=method,
        details={} if details is None else details,
        warnings=[] if notes is None else notes,
        **extras,
    )


def covariance(a):
    """정규 다변량 추론에 필요한 양의 정부호 공분산을 확인합니다.

    Args:
        a: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    if len(a) <= a.shape[1]:
        raise ValueError("각 다변량 표본의 n은 특성 수 p보다 커야 합니다.")
    cov = np.atleast_2d(np.cov(a, rowvar=False, ddof=1))
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("특이/비정부호 공분산은 지원하지 않습니다.") from exc
    return cov


def log_table_probability(t):
    """고정 주변합 도수표의 조건부 log 확률을 계산합니다.

    Args:
        t: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    return float(
        gammaln(t.sum(axis=0) + 1).sum()
        + gammaln(t.sum(axis=1) + 1).sum()
        - gammaln(t.sum() + 1)
        - gammaln(t + 1).sum()
    )


def enumerate_tables(rows, cols, max_tables):
    """고정 주변합 표를 생성하며 한도 초과는 명시적으로 오류입니다.

    Args:
        rows: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        cols: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        max_tables: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    count = [0]
    r = len(rows)
    c = len(cols)

    def fill_row(total, remaining, j, prefix):
        """fill_row의 공유 계산을 수행합니다.

        Args:
            total: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
            remaining: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
            j: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
            prefix: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        Returns:
            Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
        Raises:
            ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
        """
        if j == c - 1:
            if 0 <= total <= remaining[j]:
                yield prefix + [int(total)]
        else:
            lo = max(0, int(total - sum(remaining[j + 1 :])))
            for val in range(lo, min(int(total), int(remaining[j])) + 1):
                yield from fill_row(total - val, remaining, j + 1, prefix + [val])

    def recurse(i, remaining, built):
        """recurse의 공유 계산을 수행합니다.

        Args:
            i: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
            remaining: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
            built: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        Returns:
            Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
        Raises:
            ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
        """
        if i == r - 1:
            if sum(remaining) == rows[i]:
                count[0] += 1
                if count[0] > max_tables:
                    raise ValueError("max_tables 초과: Monte Carlo 모드를 선택하세요.")
                yield np.asarray(built + [list(remaining)], dtype=np.int64)
        else:
            for row in fill_row(rows[i], remaining, 0, []):
                yield from recurse(
                    i + 1, [remaining[j] - row[j] for j in range(c)], built + [row]
                )

    yield from recurse(0, list(cols), [])


def sample_table(rows, cols, rng):
    """행별 다변량 초기하 분포로 정확한 조건부 무작위 표를 생성합니다.

    Args:
        rows: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        cols: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        rng: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    remaining = np.array(cols, dtype=np.int64)
    t = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for i in range(len(rows) - 1):
        needed = int(rows[i])
        for j in range(len(cols) - 1):
            pop = int(remaining[j:].sum())
            good = int(remaining[j])
            v = (
                int(rng.hypergeometric(good, pop - good, needed))
                if needed and pop
                else 0
            )
            t[i, j] = v
            remaining[j] -= v
            needed -= v
        t[i, -1] = needed
        remaining[-1] -= needed
    t[-1] = remaining
    return t
