"""
================================================================================
plots.py - 데이터 시각화와 모델 평가 그래프
================================================================================
설명: 데이터시각화 PDF의 기본 차트·히트맵을 Figure/Axes 반환 함수로 구현합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 rc_context 기반 설정과 재사용 차트 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


@dataclass
class PlotProfile:
    """그래프 설정.

    Attributes:
        kind: line, scatter, bar, hist, box, heatmap.
        title: 제목.
        xlabel: x축 이름.
        ylabel: y축 이름.
        figsize: 그림 크기 (인치).
        font_family: 설치된 선택적 폰트 이름. None이면 기본 폰트.
        font_size: 글자 크기.
        bins: 히스토그램 구간 수.
        cmap: 히트맵 색상표.
    """
    kind: str = "line"
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[float, float] = (8, 4)
    font_family: Optional[str] = None
    font_size: float = 10
    bins: int = 20
    cmap: str = "viridis"


@dataclass
class PlotOutputDTO(OutputDTO):
    """그래프 결과.

    Attributes:
        figure: matplotlib Figure. 저장은 figure.savefig, 해제는 plt.close(figure).
        axes: matplotlib Axes.
        kind, title, xlabel, ylabel: 요청한 차트 유형과 제목·축 라벨.
    """
    figure: Any
    axes: Any
    kind: str = "line"
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
PlotResult = PlotOutputDTO


def run_plot(x: Any, y: Any = None, profile: Optional[PlotProfile] = None) -> PlotOutputDTO:
    """기본 그래프를 생성합니다. show/save를 자동 수행하지 않습니다.

    Args:
        x: 단일 분포 데이터 또는 x축 데이터. heatmap에서는 2차원 행렬.
        y: line/scatter/bar의 y값. line에서 None이면 x를 y값으로 사용.
        profile: 그래프 설정.
    Returns:
        PlotOutputDTO: Figure와 Axes. 전역 rcParams를 변경하지 않습니다.
    Raises:
        ValueError: 미지원 kind, 필요한 y 누락, 설치되지 않은 폰트 또는 잘못된 행렬.
    """
    p = profile if profile is not None else PlotProfile()
    if p.kind not in ("line", "scatter", "bar", "hist", "box", "heatmap"):
        raise ValueError("지원하지 않는 그래프 kind입니다.")
    if p.kind in ("scatter", "bar") and y is None:
        raise ValueError("scatter/bar에는 x와 y가 모두 필요합니다.")
    rc = {"font.size": p.font_size, "axes.unicode_minus": False}
    if p.font_family is not None:
        if p.font_family not in {f.name for f in font_manager.fontManager.ttflist}:
            raise ValueError("요청한 폰트가 설치되어 있지 않습니다: " + p.font_family)
        rc["font.family"] = p.font_family
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=p.figsize)
        try:
            if p.kind == "line":
                ax.plot(x) if y is None else ax.plot(x, y)
            elif p.kind == "scatter":
                ax.scatter(x, y, alpha=0.7)
            elif p.kind == "bar":
                ax.bar(x, y)
            elif p.kind == "hist":
                ax.hist(x, bins=p.bins, edgecolor="black")
            elif p.kind == "box":
                ax.boxplot(x)
            else:
                arr = np.asarray(x, dtype=float)
                if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
                    raise ValueError("heatmap은 유한한 2차원 수치 행렬이어야 합니다.")
                image = ax.imshow(arr, cmap=p.cmap, aspect="auto")
                fig.colorbar(image, ax=ax)
                if hasattr(x, "columns"):
                    ax.set_xticks(range(len(x.columns))); ax.set_xticklabels(x.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(x.index))); ax.set_yticklabels(x.index)
            ax.set(title=p.title, xlabel=p.xlabel, ylabel=p.ylabel)
            fig.tight_layout()
        except Exception:
            plt.close(fig)
            raise
    return PlotOutputDTO(fig, ax, p.kind, p.title, p.xlabel, p.ylabel)


@dataclass
class PairPlotOutputDTO(OutputDTO):
    """쌍별 시각화 출력 DTO.

    Attributes:
        figure: matplotlib Figure(runtime), 저장은 figure.savefig.
        axes: Axes 배열(runtime).
        grid: 원본 seaborn PairGrid(runtime).
        feature_names: 입력 열 이름, 원래 순서.
        hue: 요청한 그룹 열 또는 None.
    Notes:
        이전 grid.fig 접근을 fig 속성으로 지원합니다. 원본 PairGrid 메서드는 grid로 접근합니다.
    """
    figure: Any
    axes: Any
    grid: Any
    feature_names: list
    hue: Optional[str]

    @property
    def fig(self):
        return self.figure


def run_pairplot(data: Any, hue: Optional[str] = None) -> PairPlotOutputDTO:
    """수치 특성 간 쌍별 분포·산점도를 생성합니다.

    Args:
        data: pandas DataFrame.
        hue: 선택적 그룹 열 이름.
    Returns:
        PairPlotOutputDTO: Figure/Axes/PairGrid 및 입력 열·hue 정보. 저장/표시는 호출자가 수행합니다.
    Raises:
        ValueError: 잘못된 데이터/그룹 설정.
        ImportError: seaborn 미설치.
    """
    import seaborn as sns
    with plt.rc_context():
        grid = sns.pairplot(data=data, hue=hue, diag_kind="hist")
    return PairPlotOutputDTO(grid.fig, grid.axes, grid, [str(c) for c in data.columns], hue)
