"""분포별 QQ의 plotting position·Axes·수치 반환 공통 계약; 분포 사양은 개별 파일 소유."""

import numpy as np
import pandas as pd
from src.Visualization._common import vector, finish


def render_qq(values, distribution, fitted_parameters, profile, ax, before, cls, name):
    """명시 분포의 분위수와 관측 순서통계량을 비교합니다.

    Args:
        values: 유한 관측; 단변량 NaN은 제거.
        distribution: 개별 파일이 구성한 SciPy frozen 분포.
        fitted_parameters: 실제 사용 분포 인자 사전.
        profile: reference/fit_parameters/color/alpha/title 설정.
        ax: 사용자 Axes.
        before: 기존 artist 식별자.
        cls: 전용 Plot DTO.
        name: 분포 이름.
    Returns:
        PlotOutputDTO: 원 축·분위수·지지 범위 위반·사용 인자.
    Raises:
        ValueError: 비유한 이론 분위수·기준선 오류.
    """
    a, removed = vector(values, min_size=3)
    ordered = np.sort(a)
    probability = (np.arange(len(a)) + 0.5) / len(a)
    theoretical = np.asarray(distribution.ppf(probability), dtype=float)
    if not np.isfinite(theoretical).all():
        raise ValueError("이론 분위수가 유한하지 않습니다.")
    low, high = distribution.support()
    outside = int(((a < low) | (a > high)).sum())
    notes = [
        "Q-Q는 시각적 적합 진단이며 p 값이나 분포 검정의 자동 결론을 제공하지 않습니다."
    ]
    if outside:
        notes.append(str(outside) + "개 관측이 명시 분포 지지 범위 밖입니다.")
    if profile.fit_parameters:
        notes.append(
            "같은 관측에서 분포 인자를 추정했습니다. 이를 독립적인 검증 자료 적합도로 해석하지 않습니다."
        )
    ax.scatter(theoretical, ordered, color=profile.color, alpha=profile.alpha, s=14)
    slope = None
    intercept = None
    if profile.reference == "identity":
        slope = 1.0
        intercept = 0.0
    elif profile.reference == "quartile":
        q = np.asarray(distribution.ppf([0.25, 0.75]))
        observed = np.quantile(a, [0.25, 0.75])
        if q[1] > q[0]:
            slope = float((observed[1] - observed[0]) / (q[1] - q[0]))
            intercept = float(observed[0] - slope * q[0])
        else:
            notes.append("이론 사분위수가 같아 quartile 기준선을 생략했습니다.")
    elif profile.reference != "none":
        raise ValueError("reference는 identity/quartile/none입니다.")
    if slope is not None:
        x = np.array([theoretical.min(), theoretical.max()])
        ax.plot(x, intercept + slope * x, color="C3", linestyle="--")
    if hasattr(distribution.dist, "pmf"):
        notes.append(
            "이산 분포의 계단·동률은 정상적인 구조입니다. 직선 근접만으로 분포/독립 가정을 판정하지 않습니다."
        )
    ax.set_xlabel(name + " theoretical quantile")
    ax.set_ylabel("Observed quantile")
    data = pd.DataFrame(
        {"probability": probability, "theoretical": theoretical, "observed": ordered}
    )
    diagnostics = {
        "n": len(a),
        "missing_removed": removed,
        "distribution": name,
        "distribution_parameters": fitted_parameters,
        "plotting_position": "(i + 0.5) / n, i=0..n-1",
        "outside_support": outside,
        "reference_slope": slope,
        "reference_intercept": intercept,
    }
    return finish(ax, before, data, diagnostics, profile, cls, "Q-Q — " + name, notes)
