"""
================================================================================
mds.py - MDS 다차원 척도법
================================================================================
설명:
    - MDS 다차원 척도법을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 입력 X의 유클리드 거리를 보존하는 metric/nonmetric 적합입니다. precomputed 거리 입력과 transform은 지원하지 않습니다.
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
class MDSProfile:
    """MDS 다차원 척도법 입력 조건.

    Attributes:
        scaling (str): 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard".
        random_state (int): 재현용 0 이상 정수 시드. 기본 1234.
        n_components (int): 출력 축 수. 알고리즘별 표본·특성·클래스 제약을 만족해야 함. 기본 2.
        metric (bool): True는 metric SMACOF, False는 nonmetric SMACOF. 기본 True.
        n_init (int): 초기화 반복. 양의 정수. 기본 4.
        max_iter (int): 최대 반복. 양의 정수. 기본 300.
        eps (float): 수렴 허용오차. 양수. 기본 0.001.
    """

    scaling: str = "standard"
    random_state: int = 1234
    n_components: int = 2
    metric: bool = True
    n_init: int = 4
    max_iter: int = 300
    eps: float = 0.001


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class MDSOutputDTO(OutputDTO):
    """MDS 다차원 척도법 결과. 배열은 원 입력 행 순서.

    Attributes:
        model (Any): 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고.
        embedding (np.ndarray): (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표.
        feature_names (List[str]): 원 입력 열 순서의 이름.
        supports_transform (bool): 새 자료의 model.transform 지원 여부. False이면 재적합 필요.
        stress (float): 라이브러리의 raw SMACOF stress. 구버전에서는 최종 갱신 전 값일 수 있으며 nonmetric은 disparity 기준.
        n_iter (int): 최적 초기화에서 실제 반복 횟수.
        distance_reconstruction_error (float): 최종 좌표와 전처리 입력의 유클리드 거리 제곱 차이 합. nonmetric 목적 함수와는 다름.
    """

    model: Any
    embedding: np.ndarray
    feature_names: List[str]
    supports_transform: bool
    stress: float
    n_iter: int
    distance_reconstruction_error: float


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(X: Any, profile: MDSProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        profile: MDSProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, MDSProfile):
        raise TypeError("MDSProfile이 필요합니다.")
    arr = validate_features(X)
    validate_scaling(profile.scaling, profile.random_state)
    positive_int(profile.random_state, "random_state", 0)
    positive_int(profile.n_components, "n_components")
    positive_int(profile.n_init, "n_init")
    positive_int(profile.max_iter, "max_iter")
    finite_number(profile.eps, "eps", positive=True)
    if not isinstance(profile.metric, bool):
        raise ValueError("metric은 bool이어야 합니다.")
    if profile.n_components >= len(arr):
        raise ValueError("n_components는 표본 수 미만이어야 합니다.")
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_model(X: Any, profile: MDSProfile) -> MDSOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        profile: 검증된 MDSProfile.
    Returns:
        MDSOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    import inspect
    from sklearn.manifold import MDS

    kwargs = (
        {"normalized_stress": False}
        if "normalized_stress" in inspect.signature(MDS).parameters
        else {}
    )
    estimator = MDS(
        n_components=profile.n_components,
        metric=profile.metric,
        n_init=profile.n_init,
        max_iter=profile.max_iter,
        eps=profile.eps,
        dissimilarity="euclidean",
        random_state=profile.random_state,
        n_jobs=1,
        **kwargs,
    )
    arr = validate_features(X)
    model = Pipeline([("scaler", make_scaler(profile.scaling)), ("model", estimator)])
    embedding = model.fit_transform(arr)
    if not np.isfinite(embedding).all():
        raise ValueError("변환 좌표가 비유한 값입니다. 입력 rank와 설정을 확인하세요.")
    from scipy.spatial.distance import pdist

    scaled = (
        model.named_steps["scaler"].transform(arr) if profile.scaling != "none" else arr
    )
    error = float(np.sum((pdist(scaled) - pdist(embedding)) ** 2))
    return MDSOutputDTO(
        model,
        embedding,
        feature_names(X),
        False,
        float(estimator.stress_),
        int(estimator.n_iter_),
        error,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_mds(
    X: Any,
    profile: Optional[MDSProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> MDSOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        profile: 명시적 MDSProfile. None이면 mds.json을 읽음.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        MDSOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile(
        "mds", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_inputs(X, profile)
    return _run_model(X, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    result = run_mds(X, load_profile("mds"))
    print(result.to_json(indent=2)[:500])
