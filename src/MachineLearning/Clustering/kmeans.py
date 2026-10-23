"""
================================================================================
kmeans.py - K-means 군집화
================================================================================
설명:
    - K-means 군집화을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 전 자료 적합은 탐색 분석용이며 별도 일반화 성능을 뜻하지 않습니다.
    - 결측값과 범주형 인코딩은 실행 전에 처리합니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.0.0    모델별 레시피·전체 명세·실행 예제 작성
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.output import OutputDTO
from src.Recipes import load_profile, resolve_profile
from src.MachineLearning.common import validate_features, make_scaler
from src.MachineLearning.analysis_common import (
    positive_int,
    finite_number,
    validate_scaling,
    feature_names,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class KMeansProfile:
    """K-means 군집화 입력 조건.

    Attributes:
        scaling (str): 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard".
        random_state (int): 재현용 0 이상 정수 시드. 기본 1234.
        n_clusters (int): 요청 군집 수. 2 이상이며 서로 다른 입력 행 수 이하. 기본 3.
        n_init (int): 초기화 반복 횟수. 양의 정수. 기본 10.
        max_iter (int): 최대 반복 횟수. 양의 정수. 기본 300.
    """

    scaling: str = "standard"
    random_state: int = 1234
    n_clusters: int = 3
    n_init: int = 10
    max_iter: int = 300


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class KMeansOutputDTO(OutputDTO):
    """K-means 군집화 결과. 배열은 원 입력 행 순서.

    Attributes:
        model (Any): 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고.
        labels (np.ndarray): (n_samples,) 군집 라벨. 밀도 군집의 -1은 잡음.
        feature_names (List[str]): 원 입력 열 순서의 이름.
        n_clusters (int): 잡음 -1을 제외한 실제 군집 수.
        noise_mask (np.ndarray): (n_samples,) 잡음 여부.
        silhouette (Optional[float]): 잡음을 제외한 silhouette. 두 군집 미만 또는 각 행이 다른 군집이면 None.
        centers (np.ndarray): (n_clusters, n_features) 스케일링된 좌표계의 군집 중심.
        inertia (float): 군집 중심까지 거리 제곱 합.
        n_iter (int): 실제 반복 횟수.
    """

    model: Any
    labels: np.ndarray
    feature_names: List[str]
    n_clusters: int
    noise_mask: np.ndarray
    silhouette: Optional[float]
    centers: np.ndarray
    inertia: float
    n_iter: int


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(X: Any, profile: KMeansProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        profile: KMeansProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, KMeansProfile):
        raise TypeError("KMeansProfile이 필요합니다.")
    arr = validate_features(X)
    validate_scaling(profile.scaling, profile.random_state)
    positive_int(profile.random_state, "random_state", 0)
    positive_int(profile.n_clusters, "n_clusters", 2)
    positive_int(profile.n_init, "n_init")
    positive_int(profile.max_iter, "max_iter")
    if profile.n_clusters > len(np.unique(arr, axis=0)):
        raise ValueError("서로 다른 행 수보다 군집 수가 많습니다.")
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_model(X: Any, profile: KMeansProfile) -> KMeansOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        profile: 검증된 KMeansProfile.
    Returns:
        KMeansOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    from sklearn.cluster import KMeans

    estimator = KMeans(
        n_clusters=profile.n_clusters,
        n_init=profile.n_init,
        max_iter=profile.max_iter,
        random_state=profile.random_state,
    )
    arr = validate_features(X)
    model = Pipeline([("scaler", make_scaler(profile.scaling)), ("model", estimator)])
    labels = model.fit_predict(arr)
    from src.MachineLearning.analysis_common import cluster_summary

    summary = cluster_summary(
        (
            model.named_steps["scaler"].transform(arr)
            if profile.scaling != "none"
            else arr
        ),
        labels,
    )
    fitted = model.named_steps["model"]
    return KMeansOutputDTO(
        model=model,
        labels=labels,
        feature_names=feature_names(X),
        **summary,
        centers=fitted.cluster_centers_.copy(),
        inertia=float(fitted.inertia_),
        n_iter=int(fitted.n_iter_),
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_kmeans(
    X: Any,
    profile: Optional[KMeansProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> KMeansOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        profile: 명시적 KMeansProfile. None이면 kmeans.json을 읽음.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        KMeansOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile(
        "kmeans", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_inputs(X, profile)
    return _run_model(X, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.3, random_state=1234)
    result = run_kmeans(X, load_profile("kmeans"))
    print(result.to_json(indent=2)[:500])
