"""
================================================================================
dendrogram.py - 계층 병합 거리와 절단 후보로 군집 구조를 검토합니다.
================================================================================
설명:
    - Profile → JSON → 단일 ax 시각화 → 수치 근거·runtime DTO.
분석 가정:
    - 사용자가 전달한 2D Axes에만 그리며 figure 생성·show·save·예측 모델 재학습을 수행하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    Axes 조합 EDA·의사결정 시각화 추가
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
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.Recipes import resolve_profile
from src.Visualization._common import (
    PlotOutputDTO,
    validate_profile,
    start,
    finish,
    vector,
    frame,
    paired,
    levels,
    heatmap,
    binary_scores,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class ClusteringDendrogramProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        method (str): single/complete/average/weighted/centroid/median/ward. 기본 'ward'.
        metric (str): SciPy pdist 거리; ward/centroid/median은 euclidean. 기본 'euclidean'.
        input_kind (str): observations/distance/linkage; 거리 행렬 명시. 기본 'observations'.
        cut_height (Optional[float]): 선택 비음수 절단 높이. 기본 None.
        max_samples (int): O(n²) 계층 계산/표시의 최대 표본 수. 기본 1000.
        truncate_last (int): 마지막 병합 단위 표시 수; 0이면 원 잎 전체. 기본 30.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    method: str = "ward"
    metric: str = "euclidean"
    input_kind: str = "observations"
    cut_height: Optional[float] = None
    max_samples: int = 1000
    truncate_last: int = 30


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ClusteringDendrogramOutputDTO(PlotOutputDTO):
    """그린 축과 의사결정 수치.

    Attributes:
        figure (Any): 원 ax의 Figure; runtime으로 JSON 제외.
        axes (Any): 전달한 동일 Axes; runtime으로 JSON 제외.
        artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
        data (pd.DataFrame): 표시/계산한 수치와 원 label 순서.
        diagnostics (Dict[str, Any]): 표본 수·분모·선택 기준·추가 근거.
        parameters (Dict[str, Any]): 적용 Profile 설정.
        warnings (List[str]): 해석 가정·미표시/표본 제한 정보.
    """

    figure: Any
    axes: Any
    artists: List[Any]
    data: pd.DataFrame
    diagnostics: Dict[str, Any]
    parameters: Dict[str, Any]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate(profile):
    """전용 Profile과 공통 스타일을 확인합니다.

    Args:
        profile: ClusteringDendrogramProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, ClusteringDendrogramProfile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(values, labels, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        labels: 관측 순서의 라벨 또는 명시 클래스 순서; 길이/포함 범위를 검증.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        ClusteringDendrogramOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    if (
        profile.method
        not in (
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        )
        or profile.max_samples < 2
        or profile.truncate_last < 0
    ):
        raise ValueError("계층 설정 오류")
    if profile.cut_height is not None and (
        isinstance(profile.cut_height, bool)
        or not isinstance(profile.cut_height, (int, float))
        or not np.isfinite(profile.cut_height)
        or profile.cut_height < 0
    ):
        raise ValueError("절단 높이 오류")
    raw = frame(values)
    if profile.input_kind == "linkage":
        Z = raw.values
        hierarchy.is_valid_linkage(Z, throw=True)
        n = len(Z) + 1
    elif profile.input_kind == "observations":
        n = len(raw)
        if n < 2 or n > profile.max_samples:
            raise ValueError("표본 한도 오류")
        if (
            profile.method in ("ward", "centroid", "median")
            and profile.metric != "euclidean"
        ):
            raise ValueError("해당 linkage는 euclidean 관측 거리가 필요합니다.")
        Z = hierarchy.linkage(raw.values, method=profile.method, metric=profile.metric)
    elif profile.input_kind == "distance":
        n = len(raw)
        if (
            raw.shape != (n, n)
            or n < 2
            or n > profile.max_samples
            or (raw.values < 0).any()
            or not np.allclose(raw.values, raw.values.T)
            or not np.allclose(np.diag(raw), 0)
        ):
            raise ValueError("대칭 비음수 거리와 영 대각이 필요합니다.")
        if profile.method in ("ward", "centroid", "median"):
            raise ValueError(
                "사전 거리에는 average/complete/single/weighted를 사용하세요."
            )
        Z = hierarchy.linkage(
            squareform(raw.values, checks=True), method=profile.method
        )
    else:
        raise ValueError("input_kind 오류")
    if n > profile.max_samples:
        raise ValueError("표본 한도 초과")
    if labels is not None and (len(labels) != n or any(pd.isna(v) for v in labels)):
        raise ValueError("잎 label 길이/결측 오류")
    kwargs = (
        {"truncate_mode": "lastp", "p": profile.truncate_last, "show_leaf_counts": True}
        if profile.truncate_last
        else {}
    )
    result = hierarchy.dendrogram(
        Z,
        labels=None if labels is None else [str(v) for v in labels],
        ax=ax,
        color_threshold=profile.cut_height,
        **kwargs,
    )
    if profile.cut_height is not None:
        ax.axhline(profile.cut_height, color="C3", linestyle="--")
    ax.set_xlabel("Leaf / contracted cluster")
    ax.set_ylabel("Merge distance")
    data = pd.DataFrame(Z, columns=["left", "right", "distance", "n_leaves"])
    diagnostics = {
        "n_samples": n,
        "displayed_leaves": result["ivl"],
        "leaf_order": result["leaves"],
        "cluster_labels": (
            hierarchy.fcluster(Z, profile.cut_height, criterion="distance")
            if profile.cut_height is not None
            else None
        ),
        "method": profile.method,
        "metric": profile.metric,
    }
    notes = [
        "거리/스케일/연결 기준에 민감합니다. 절단 높이는 탐색 기준이며 최적 군집 수를 자동 결정하지 않습니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        ClusteringDendrogramOutputDTO,
        "Clustering — Dendrogram",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_dendrogram(
    values, labels=None, *, ax, profile=None, recipe_path=None, overrides=None
) -> ClusteringDendrogramOutputDTO:
    """계층 병합 거리와 절단 후보로 군집 구조를 검토합니다.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        labels: 관측 순서의 라벨 또는 명시 클래스 순서; 길이/포함 범위를 검증.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        ClusteringDendrogramOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-clustering-dendrogram",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    before = start(ax)
    return _draw(values, labels, ax, profile, before)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.clustering에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_dendrogram(..., ax=axes[0, 0]); result.axes is axes[0, 0]
