"""
================================================================================
silhouette.py - 음의 silhouette·군집별 산포와 평균으로 군집 분리를 비교합니다.
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
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
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
class ClusteringSilhouetteProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        metric (str): sklearn 거리 또는 precomputed. 기본 'euclidean'.
        noise_label (Any): 제외할 밀도 군집 noise 라벨; None은 미제외. 기본 -1.
        max_samples (int): 전체 쌍 거리 계산의 최대 유효 표본. 기본 5000.
    """

    title: str = ""
    color: str = "C0"
    alpha: float = 0.8
    metric: str = "euclidean"
    noise_label: Any = -1
    max_samples: int = 5000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ClusteringSilhouetteOutputDTO(PlotOutputDTO):
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
        profile: ClusteringSilhouetteProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, ClusteringSilhouetteProfile)


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
        ClusteringSilhouetteOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    from sklearn.metrics import silhouette_samples

    if (
        isinstance(values, pd.DataFrame)
        and isinstance(labels, pd.Series)
        and not values.index.equals(labels.index)
    ):
        raise ValueError("관측과 군집 라벨 인덱스가 다릅니다.")
    raw = frame(values)
    lab = np.asarray(labels)
    if lab.ndim != 1 or len(lab) != len(raw) or pd.isna(lab).any():
        raise ValueError("대응 라벨 오류")
    keep = (
        np.ones(len(lab), dtype=bool)
        if profile.noise_label is None
        else pd.Series(lab).ne(profile.noise_label).to_numpy(dtype=bool)
    )
    if profile.metric == "precomputed":
        if (
            raw.shape != (len(raw), len(raw))
            or (raw.values < 0).any()
            or not np.allclose(raw.values, raw.values.T)
            or not np.allclose(np.diag(raw), 0)
        ):
            raise ValueError("precomputed 거리 행렬 오류")
        X = raw.values[np.ix_(keep, keep)]
    else:
        X = raw.values[keep]
    lab = lab[keep]
    unique = pd.unique(lab)
    if (
        not 2 <= len(unique) < len(lab)
        or len(lab) > profile.max_samples
        or profile.max_samples < 3
    ):
        raise ValueError("silhouette 군집/표본 수/한도 오류")
    scores = silhouette_samples(X, lab, metric=profile.metric)
    bottom = 0
    for i, label in enumerate(unique):
        selected = np.sort(scores[lab == label])
        ax.fill_betweenx(
            np.arange(bottom, bottom + len(selected)),
            0,
            selected,
            alpha=profile.alpha,
            color="C" + str(i % 10),
        )
        ax.text(-0.98, bottom + len(selected) / 2, str(label), fontsize=8)
        bottom += len(selected) + 3
    mean = float(scores.mean())
    ax.axvline(mean, color="C3", linestyle="--", label="Mean=" + format(mean, ".3f"))
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Samples grouped by cluster")
    ax.legend(fontsize=8)
    data = pd.DataFrame(
        {
            "original_position": np.flatnonzero(keep),
            "cluster": lab,
            "silhouette": scores,
        }
    )
    diagnostics = {
        "mean_silhouette": mean,
        "negative_fraction": float((scores < 0).mean()),
        "noise_excluded": int((~keep).sum()),
        "n": len(lab),
    }
    notes = [
        "noise 제외 후 군집 구조를 평가합니다. 서로 다른 거리/Feature 공간의 점수를 직접 비교하지 않습니다."
    ]
    return finish(
        ax,
        before,
        data,
        diagnostics,
        profile,
        ClusteringSilhouetteOutputDTO,
        "Clustering — Silhouette",
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_silhouette(
    values, labels, *, ax, profile=None, recipe_path=None, overrides=None
) -> ClusteringSilhouetteOutputDTO:
    """음의 silhouette·군집별 산포와 평균으로 군집 분리를 비교합니다.

    Args:
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        labels: 관측 순서의 라벨 또는 명시 클래스 순서; 길이/포함 범위를 검증.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        ClusteringSilhouetteOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile = resolve_profile(
        "plot-clustering-silhouette",
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
# 예: result = plot_silhouette(..., ax=axes[0, 0]); result.axes is axes[0, 0]
