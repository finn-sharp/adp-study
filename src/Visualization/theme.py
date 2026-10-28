"""
================================================================================
theme.py - 흰 바탕·절제된 팔레트·얇은 격자의 분석 차트 테마. 명시한 컨텍스트와 축에만 적용합니다.
================================================================================
설명:
    - 흰 바탕·절제된 팔레트·얇은 격자의 분석 차트 테마. 명시한 컨텍스트와 축에만 적용합니다.
분석 가정:
    - 명시한 2D Axes의 장식만 변경하며 figure 생성·저장·자료 재계산을 수행하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    분석 보고서 테마·지역 스타일 컨텍스트 추가
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

PALETTE = (
    "#2E746E",
    "#DC9947",
    "#91AA77",
    "#C96F68",
    "#8C82AA",
    "#557C98",
    "#BDA66A",
    "#6E9491",
)
INK = "#293D46"
MUTED = "#788891"
RULE = "#E3E9ED"


@lru_cache(maxsize=1)
def _font_name():
    """저장소의 한글 폰트를 명시 등록합니다. import만으로 등록하지 않습니다."""
    from matplotlib import font_manager

    path = Path(__file__).resolve().parents[2] / "기출문제/공통/NanumGothic-Regular.ttf"
    font_manager.fontManager.addfont(str(path))
    return font_manager.FontProperties(fname=str(path)).get_name()


@contextmanager
def analysis_theme():
    """지역 rc_context로 분석 보고서 스타일을 적용하고 원 설정을 복원합니다.

    Args:
        None: 저장소 한글 폰트와 기본 팔레트 사용.
    Yields:
        None: 호출자가 Figure/Axes와 plot 함수를 실행하는 구간.
    Raises:
        OSError/ImportError: 폰트·Matplotlib 의존성 오류.
    """
    import matplotlib as mpl
    from cycler import cycler

    with mpl.rc_context(
        {
            "font.family": _font_name(),
            "font.size": 10,
            "axes.unicode_minus": False,
            "text.color": INK,
            "axes.labelcolor": "#63757F",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.prop_cycle": cycler(color=PALETTE),
            "axes.edgecolor": "#CAD5DA",
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.labelsize": 9,
            "axes.labelpad": 9,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK,
            "legend.handlelength": 2.0,
            "legend.borderaxespad": 0.6,
            "patch.linewidth": 0.6,
        }
    ):
        yield


def _colormaps():
    """크기·부호의 의미를 보존하는 지역 색상표를 구성합니다."""
    from matplotlib.colors import LinearSegmentedColormap

    sequential = LinearSegmentedColormap.from_list(
        "adp_sequential", ["#EFF5F2", "#AED0BD", "#4E9484", "#245E5B"]
    )
    diverging = LinearSegmentedColormap.from_list(
        "adp_diverging", ["#397B87", "#F5F7F6", "#C87962"]
    )
    sequential.set_bad("#F5F7F8")
    diverging.set_bad("#F5F7F8")
    return sequential, diverging


def style_axes(ax, kind=""):
    """지정 축의 장식·범례·색상표만 정리하고 수치·척도·기준선 위치는 보존합니다.

    Args:
        ax: 호출자가 소유한 기존 2D Axes.
        kind: 선택 시각화 종류. 예: numeric.histogram, classification.confusion.
    Returns:
        Axes: 동일 축. 다른 축과 figure를 생성하지 않음.
    Raises:
        TypeError: 잘못된 축.
    """
    from matplotlib.axes import Axes
    from matplotlib.colors import to_hex
    import numpy as np

    if not isinstance(ax, Axes) or ax.name != "rectilinear":
        raise TypeError("기존 2D 직교 Axes가 필요합니다.")
    sequential, diverging = _colormaps()
    mappables = list(ax.images) + [
        c for c in ax.collections if c.get_array() is not None
    ]
    for artist in mappables:
        name = artist.get_cmap().name
        if name in ("viridis", "plasma", "YlGnBu", "Blues"):
            artist.set_cmap(sequential)
        elif name in ("coolwarm", "bwr", "RdBu", "RdBu_r"):
            artist.set_cmap(diverging)
    heatmap = bool(ax.images)
    ax.set_axisbelow(True)
    ax.grid(False)
    if ax.axison and not heatmap and "mosaic" not in kind and "hexbin" not in kind:
        horizontal = any(
            name in kind
            for name in (
                "category_rate",
                "feature_importance",
                "vif",
                "model_comparison",
                "coefficient_intervals",
            )
        )
        ax.grid(
            True, axis="x" if horizontal else "y", color=RULE, linewidth=0.6, alpha=0.8
        )
    for side, spine in ax.spines.items():
        spine.set_visible(ax.axison and not heatmap and side in ("left", "bottom"))
        spine.set_color("#CAD5DA")
        spine.set_linewidth(0.65)
    ax.tick_params(axis="both", which="both", length=0, colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color("#63757F")
    ax.yaxis.label.set_color("#63757F")
    for line in ax.lines:
        try:
            color = to_hex(line.get_color())
        except (ValueError, TypeError):
            continue
        if color == "#000000":
            line.set_color("#788891")
            if line.get_linestyle() in ("--", ":", "-."):
                line.set_linewidth(1.0)
                line.set_alpha(0.75)
        elif color == "#ff0000":
            line.set_color("#C96F68")
    if "silhouette" in kind:
        for i, collection in enumerate(ax.collections):
            if collection.get_array() is None:
                collection.set_facecolor(PALETTE[i % len(PALETTE)])
    if "networks.graph" in kind:
        for collection in ax.collections:
            if hasattr(collection, "get_sizes"):
                collection.set_facecolor("#BCD8CB")
                collection.set_edgecolor("white")
    # Annotation contrast is recomputed after a heatmap color change.
    for image in ax.images:
        values = image.get_array()
        if values.ndim != 2:
            continue
        for text in ax.texts:
            x, y = text.get_position()
            i, j = int(round(y)), int(round(x))
            if not (0 <= i < values.shape[0] and 0 <= j < values.shape[1]):
                continue
            value = values[i, j]
            if np.ma.is_masked(value) or not np.isfinite(value):
                continue
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text.set_color("white" if luminance < 0.5 else INK)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_color("#52656F")
            text.set_fontsize(8)
    return ax
