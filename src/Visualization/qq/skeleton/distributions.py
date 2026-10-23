"""distributions UCI 스켈레톤.

원 자료 → JSON/Profile → 학습 전용 파이프라인 → 수치/진단/사용자 Axes.
작성자: 김재현(Finn), 2026-10-12. 변경 이력: v1.0.0 최초 추가.
라이선스: Proprietary; UCI 원본은 CC BY 4.0. import에서 다운로드·학습·그림 생성 없음.
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
from io import BytesIO
from src.datasets.uci import load_tabular, load_bike, dataset_bytes
from src.datasets.adult import load_adult
from src.datasets.optdigits import load_optdigits
from src.Recipes import load_profile
from src.Visualization.skeleton._common import render_panels
import importlib
import numpy as np
import pandas as pd


def load_data(cache_dir=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
    Returns:
        dict: 분포별 명시 원 UCI Feature; 동일 자료의 fitting은 탐색.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    wine = load_tabular("regression", cache_dir=cache_dir)
    bike = load_bike(cache_dir)
    raw_bike = pd.read_csv(BytesIO(dataset_bytes("bike", cache_dir)))
    adult, _ = load_adult(cache_dir)
    digits, labels, future, future_labels = load_optdigits(cache_dir)
    positive = wine.iloc[:, 0]
    data = {
        name: positive
        for name in ("exponential", "gamma", "lognormal", "weibull", "chi_square")
    }
    data.update({name: wine["pH"] for name in ("normal", "student_t", "logistic")})
    data.update(
        beta=raw_bike.hum,
        uniform=raw_bike.temp,
        poisson=bike.cnt,
        binomial=digits.reshape(len(digits), -1)[:, 28],
        bernoulli=(adult.income == ">50K").astype(int),
    )
    return data


def load_recipe(overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        overrides: 전용 명세의 overrides 입력.
    Returns:
        dict: 분포별 Profile; 지지/위치/시행 수 명시.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    names = [
        "normal",
        "student_t",
        "exponential",
        "gamma",
        "lognormal",
        "weibull",
        "beta",
        "uniform",
        "chi_square",
        "logistic",
        "poisson",
        "binomial",
        "bernoulli",
    ]
    updates = overrides or {}
    if set(updates) - set(names):
        raise ValueError("분포 이름 오류")
    profiles = {}
    for name in names:
        settings = {
            "fit_parameters": name not in ("beta", "uniform"),
            "reference": "identity",
        }
        if name == "binomial":
            settings["n_trials"] = 16
        settings.update(updates.get(name, {}))
        profiles[name] = load_profile(
            "plot-qq-" + name.replace("_", "-"), overrides=settings
        )
    return profiles


def build_pipeline(data, profiles=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        data: 전용 명세의 data 입력.
        profiles: 전용 명세의 profiles 입력.
    Returns:
        list: 각 ax에 독립 분포 QQ 연결.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profiles = load_recipe() if profiles is None else profiles
    calls = []
    for name, profile in profiles.items():
        function = getattr(
            importlib.import_module("src.Visualization.qq." + name), "plot_" + name
        )
        calls.append(
            (
                name,
                lambda ax, name=name, function=function, profile=profile: function(
                    data[name], ax=ax, profile=profile
                ),
            )
        )
    return calls


def main(cache_dir=None, axes=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
        axes: 전용 명세의 axes 입력.
        overrides: 전용 명세의 overrides 입력.
    Returns:
        EDADashboardOutputDTO: 13개 독립 QQ pane·실제 분포 인자.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    data = load_data(cache_dir)
    profiles = load_recipe(overrides)
    return render_panels(build_pipeline(data, profiles), axes=axes, ncols=3)


if __name__ == "__main__":
    result = main()
    print(
        {
            name: panel.diagnostics["distribution_parameters"]
            for name, panel in result.panels.items()
        }
    )
