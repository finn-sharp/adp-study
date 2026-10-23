"""relationships UCI subplot 스켈레톤.

원 자료 → pane별 Profile → 분석 결과/그리기 함수 연결 → 명시적 subplot 조합.
각 pane의 수치와 가정을 반환합니다. 작성자: 김재현(Finn), 2026-10-12.
변경 이력: v1.0.0 최초 추가. 라이선스: Proprietary; UCI 원 자료는 CC BY 4.0.
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
import numpy as np
from src.Recipes import load_profile
from ._common import render_panels
from src.datasets.uci import load_tabular
from src.Visualization.relationships.scatter import plot_scatter
from src.Visualization.relationships.hexbin import plot_hexbin
from src.Visualization.relationships.correlation import plot_correlation

RECIPES = {
    "scatter": "plot-relationships-scatter",
    "hexbin": "plot-relationships-hexbin",
    "correlation": "plot-relationships-correlation",
}


def load_data(cache_dir=None):
    """학습·그림 생성 없이 공식 UCI 원 자료를 읽습니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시.
    Returns:
        object: 분야별 원 관측 자료.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·자료·다운로드·축 오류.
    """
    return load_tabular("classification", cache_dir=cache_dir)


def load_recipe(overrides=None):
    """pane별 기존 JSON/Profile을 읽습니다.

    Args:
        overrides: pane 이름별 레시피 overrides 사전 또는 None.
    Returns:
        dict: pane 이름 → 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·자료·다운로드·축 오류.
    """
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - set(RECIPES):
        raise ValueError("pane override 이름 오류")
    return {
        name: load_profile(key, overrides=overrides.get(name))
        for name, key in RECIPES.items()
    }


def build_pipeline(data, profiles=None):
    """원 자료에서 필요한 분석 수치를 준비하고 단일 ax 호출을 연결합니다.

    Args:
        data: load_data의 원 자료.
        profiles: pane 이름별 전용 Profile 또는 None.
    Returns:
        list: (pane 이름, ax callable) 목록.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·자료·다운로드·축 오류.
    """
    profiles = load_recipe() if profiles is None else profiles
    from src.stats.correlation.numeric.pearson import run_pearson_matrix

    X = data.drop(columns="target")
    correlation = run_pearson_matrix(X)
    return [
        (
            "scatter",
            lambda ax: plot_scatter(
                data.sepal_length,
                data.petal_length,
                data.target,
                ax=ax,
                profile=profiles["scatter"],
            ),
        ),
        (
            "hexbin",
            lambda ax: plot_hexbin(
                data.sepal_length, data.petal_length, ax=ax, profile=profiles["hexbin"]
            ),
        ),
        (
            "correlation",
            lambda ax: plot_correlation(
                correlation.correlation_matrix,
                correlation.adjusted_p_value_matrix,
                ax=ax,
                profile=profiles["correlation"],
            ),
        ),
    ]


def main(cache_dir=None, axes=None, overrides=None):
    """사용자의 subplot 축에 연결하거나 예제 레이아웃을 생성합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시.
        axes: 기존 subplot Axes 배열 또는 None; None에서만 예제 레이아웃 생성.
        overrides: pane 이름별 레시피 overrides 사전 또는 None.
    Returns:
        EDADashboardOutputDTO: 같은 축과 pane별 수치.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·자료·다운로드·축 오류.
    """
    data = load_data(cache_dir)
    profiles = load_recipe(overrides)
    calls = build_pipeline(data, profiles)
    return render_panels(calls, axes=axes, ncols=2)


if __name__ == "__main__":
    result = main()
    print({name: list(panel.diagnostics) for name, panel in result.panels.items()})
