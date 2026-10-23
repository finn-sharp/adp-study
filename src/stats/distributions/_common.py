"""
================================================================================
_common.py - 통계 분석 공유 수치 구현
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
from dataclasses import asdict
import numpy as np
from scipy import stats
from src.stats.tests._common import integer


def query_array(x, probability=False):
    """조회점을 1차원으로 만들고 확률 정의역을 확인합니다.

    Args:
        x: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        probability: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    if x is None:
        return None
    if np.iscomplexobj(x):
        raise ValueError("실수 조회점 필요")
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.ndim != 1 or len(a) == 0 or np.isnan(a).any():
        raise ValueError("조회 배열 shape/NaN 오류")
    if probability and (not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()):
        raise ValueError("q는 0과1 사이입니다.")
    return a


def evaluate_distribution(name, p, queries, rv, discrete):
    """점확률/밀도와 꼬리·구간을 분리한 결과를 생성합니다.

    Args:
        name: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        p: 해당 계산의 검증된 자료/설정. 공개 모듈의 기법별 계약 참조.
        queries: x/interval/q/n_samples 조회 사전.
        rv: 기법 모듈이 만든 SciPy frozen distribution.
        discrete: 이산 분포 여부.
    Returns:
        Any: 함수 설명에 해당하는 검증 자료·계산값·결과 사전.
    Raises:
        ValueError/TypeError: 자료·설정·정의역·계산 한도 오류.
    """
    x = query_array(queries["x"])
    q = query_array(queries["q"], True)
    size = integer(queries["n_samples"], "n_samples")
    interval = queries["interval"]
    prob = None
    if interval is not None:
        v = np.asarray(interval, dtype=float)
        if v.shape != (2,) or np.isnan(v).any() or v[0] > v[1]:
            raise ValueError("유효한 [low,high] 구간 필요")
        lo, hi = map(float, v)
        interval = (lo, hi)
        if discrete:
            # For integer support, closed interval P(ceil(lo) <= X <= floor(hi)).
            left = np.ceil(lo) - 1
            right = np.floor(hi)
        else:
            left = lo
            right = hi
        if left >= right and (not discrete or np.ceil(lo) > np.floor(hi)):
            prob = 0.0
        else:
            # Use SF difference in the upper tail to avoid catastrophic cancellation.
            prob = (
                float(rv.sf(left) - rv.sf(right))
                if rv.cdf(left) > 0.5
                else float(rv.cdf(right) - rv.cdf(left))
            )
            prob = float(np.clip(prob, 0, 1))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mean, var = rv.stats(moments="mv")
    if name == "hypergeometric" and p.N == 1:
        mean = p.n * p.K
        var = 0.0
    defined = {"mean": bool(np.isfinite(mean)), "variance": bool(np.isfinite(var))}
    notes = []
    if not all(defined.values()):
        notes.append(
            "모수 조건에서 평균/분산이 유한하게 정의되지 않습니다. 해당 필드는 None입니다."
        )
    return dict(
        distribution_name=name,
        distribution_kind="discrete" if discrete else "continuous",
        parameters=asdict(p),
        evaluation_points=x,
        pmf=rv.pmf(x) if discrete and x is not None else None,
        pdf=rv.pdf(x) if not discrete and x is not None else None,
        cdf=rv.cdf(x) if x is not None else None,
        survival=rv.sf(x) if x is not None else None,
        interval=interval,
        interval_probability=prob,
        quantile_probabilities=q,
        quantiles=rv.ppf(q) if q is not None else None,
        inverse_survival=rv.isf(q) if q is not None else None,
        mean=float(mean) if defined["mean"] else None,
        variance=float(var) if defined["variance"] else None,
        moment_defined=defined,
        samples=(
            rv.rvs(size=size, random_state=np.random.RandomState(p.random_state))
            if size
            else None
        ),
        distribution=rv,
        warnings=notes,
    )
