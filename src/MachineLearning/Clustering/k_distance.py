"""
================================================================================
k_distance.py - DBSCAN eps 선택용 k-distance 진단
================================================================================
설명:
    - DBSCAN eps 선택용 k-distance 진단을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 진단 도구이며 자동 최적 eps 또는 이상 탐지 임계값을 주장하지 않습니다.
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
from src.MachineLearning.analysis_common import positive_int, finite_number, validate_scaling, feature_names



# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class KDistanceProfile:
    """DBSCAN eps 선택용 k-distance 진단 입력 조건.

    Attributes:
        scaling (str): 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard".
        k (int): 조회 이웃 수. 자기 자신 포함하므로 DBSCAN min_samples와 같은 정의. 2 이상 표본 수 이하. 기본 5.
    """

    scaling: str = "standard"
    k: int = 5


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class KDistanceOutputDTO(OutputDTO):
    """DBSCAN eps 선택용 k-distance 진단 결과. 배열은 원 입력 행 순서.

    Attributes:
        model (Any): 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고.
        feature_names (List[str]): 원 입력 열 순서의 이름.
        distances (np.ndarray): (n_samples,) 자기 자신 포함 k번째 이웃까지 거리.
        sorted_indices (np.ndarray): 거리 오름차순 정렬의 원 행 위치.
        sorted_distances (np.ndarray): 오름차순 거리. 곡선의 굽은 부분을 eps 후보로 검토.
    """

    model: Any
    feature_names: List[str]
    distances: np.ndarray
    sorted_indices: np.ndarray
    sorted_distances: np.ndarray


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate_inputs(X: Any, profile: KDistanceProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        profile: KDistanceProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, KDistanceProfile):
        raise TypeError("KDistanceProfile이 필요합니다.")
    arr = validate_features(X)
    validate_scaling(profile.scaling)
    positive_int(profile.k, "k", 2)
    if profile.k > len(arr): raise ValueError("k는 표본 수 이하여야 합니다.")
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_model(X: Any, profile: KDistanceProfile) -> KDistanceOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        profile: 검증된 KDistanceProfile.
    Returns:
        KDistanceOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    from sklearn.neighbors import NearestNeighbors
    arr = validate_features(X)
    model = Pipeline([("scaler", make_scaler(profile.scaling)), ("model", NearestNeighbors(n_neighbors=profile.k))])
    model.fit(arr)
    scaled = model.named_steps["scaler"].transform(arr) if profile.scaling != "none" else arr
    # 명시적 X 조회는 자기 자신을 포함합니다. X=None 조회는 자기 자신을 제외하므로 사용하지 않음.
    distances = model.named_steps["model"].kneighbors(scaled)[0][:, -1]
    order = np.argsort(distances, kind="stable")
    return KDistanceOutputDTO(model, feature_names(X), distances, order, distances[order])


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_k_distance(X: Any, profile: Optional[KDistanceProfile] = None, *,
              recipe_path: Optional[str] = None,
              overrides: Optional[Dict[str, Any]] = None) -> KDistanceOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        profile: 명시적 KDistanceProfile. None이면 k-distance.json을 읽음.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        KDistanceOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile("k-distance", profile, recipe_path=recipe_path, overrides=overrides)
    _validate_inputs(X, profile)
    return _run_model(X, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.3, random_state=1234)
    result = run_k_distance(X, load_profile("k-distance"))
    print(result.to_json(indent=2)[:500])
