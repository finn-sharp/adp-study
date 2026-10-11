"""
================================================================================
dbscan.py - DBSCAN 밀도 군집화
================================================================================
설명:
    - DBSCAN 밀도 군집화을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 유클리드 거리 기반이며 새로운 자료의 predict/transform을 지원하지 않습니다.
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
class DBSCANProfile:
    """DBSCAN 밀도 군집화 입력 조건.

    Attributes:
        scaling (str): 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard".
        eps (float): 이웃 반경. 유한한 양수. 기본 0.5.
        min_samples (int): 핵심점 최소 이웃 수. 자기 자신 포함. 2 이상. 기본 5.
    """

    scaling: str = "standard"
    eps: float = 0.5
    min_samples: int = 5


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class DBSCANOutputDTO(OutputDTO):
    """DBSCAN 밀도 군집화 결과. 배열은 원 입력 행 순서.

    Attributes:
        model (Any): 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고.
        labels (np.ndarray): (n_samples,) 군집 라벨. 밀도 군집의 -1은 잡음.
        feature_names (List[str]): 원 입력 열 순서의 이름.
        n_clusters (int): 잡음 -1을 제외한 실제 군집 수.
        noise_mask (np.ndarray): (n_samples,) 잡음 여부.
        silhouette (Optional[float]): 잡음을 제외한 silhouette. 두 군집 미만 또는 각 행이 다른 군집이면 None.
        core_indices (np.ndarray): 원 입력 기준 핵심점 행 위치.
    """

    model: Any
    labels: np.ndarray
    feature_names: List[str]
    n_clusters: int
    noise_mask: np.ndarray
    silhouette: Optional[float]
    core_indices: np.ndarray


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate_inputs(X: Any, profile: DBSCANProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        profile: DBSCANProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, DBSCANProfile):
        raise TypeError("DBSCANProfile이 필요합니다.")
    arr = validate_features(X)
    validate_scaling(profile.scaling)
    finite_number(profile.eps, "eps", positive=True)
    positive_int(profile.min_samples, "min_samples", 2)
    if profile.min_samples > len(arr): raise ValueError("min_samples는 표본 수 이하여야 합니다.")
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_model(X: Any, profile: DBSCANProfile) -> DBSCANOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        profile: 검증된 DBSCANProfile.
    Returns:
        DBSCANOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    from sklearn.cluster import DBSCAN
    estimator = DBSCAN(eps=profile.eps, min_samples=profile.min_samples, metric="euclidean", n_jobs=1)
    arr = validate_features(X)
    model = Pipeline([("scaler", make_scaler(profile.scaling)), ("model", estimator)])
    labels = model.fit_predict(arr)
    from src.MachineLearning.analysis_common import cluster_summary
    summary = cluster_summary(model.named_steps["scaler"].transform(arr) if profile.scaling != "none" else arr, labels)
    return DBSCANOutputDTO(model=model, labels=labels, feature_names=feature_names(X), **summary,
                          core_indices=model.named_steps["model"].core_sample_indices_.copy())


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_dbscan(X: Any, profile: Optional[DBSCANProfile] = None, *,
              recipe_path: Optional[str] = None,
              overrides: Optional[Dict[str, Any]] = None) -> DBSCANOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        profile: 명시적 DBSCANProfile. None이면 dbscan.json을 읽음.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        DBSCANOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile("dbscan", profile, recipe_path=recipe_path, overrides=overrides)
    _validate_inputs(X, profile)
    return _run_model(X, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.3, random_state=1234)
    result = run_dbscan(X, load_profile("dbscan"))
    print(result.to_json(indent=2)[:500])
