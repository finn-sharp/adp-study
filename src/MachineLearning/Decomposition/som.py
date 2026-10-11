"""
================================================================================
som.py - SOM 자기조직화 지도
================================================================================
설명:
    - SOM 자기조직화 지도을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 2차원 직사각형 격자의 온라인 Kohonen SOM입니다. 군집 수는 별도로 추정하지 않습니다.
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class SOMProfile:
    """SOM 자기조직화 지도 입력 조건.

    Attributes:
        scaling (str): 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard".
        random_state (int): 재현용 0 이상 정수 시드. 기본 1234.
        rows (int): 격자 행 수. 양의 정수. 기본 5.
        columns (int): 격자 열 수. 양의 정수. 기본 5.
        n_iter (int): 온라인 갱신 횟수. 양의 정수. 기본 1000.
        learning_rate (float): 초기 학습률. 0 초과 1 이하. 기본 0.5.
        sigma (float): 초기 이웃 폭. 유한한 양수. 기본 1.5.
    """

    scaling: str = "standard"
    random_state: int = 1234
    rows: int = 5
    columns: int = 5
    n_iter: int = 1000
    learning_rate: float = 0.5
    sigma: float = 1.5


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class SOMOutputDTO(OutputDTO):
    """SOM 자기조직화 지도 결과. 배열은 원 입력 행 순서.

    Attributes:
        model (Any): 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고.
        embedding (np.ndarray): (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표.
        feature_names (List[str]): 원 입력 열 순서의 이름.
        supports_transform (bool): 새 자료의 model.transform 지원 여부. False이면 재적합 필요.
        weights (np.ndarray): (rows, columns, n_features) 코드북. 스케일링 좌표계.
        quantization_error (float): 입력과 BMU 코드북 사이 평균 유클리드 거리.
        topographic_error (float): 최상위 두 BMU가 4-이웃으로 인접하지 않은 관측 비율. 1개 노드이면 0.
    """

    model: Any
    embedding: np.ndarray
    feature_names: List[str]
    supports_transform: bool
    weights: np.ndarray
    quantization_error: float
    topographic_error: float


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate_inputs(X: Any, profile: SOMProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        profile: SOMProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, SOMProfile):
        raise TypeError("SOMProfile이 필요합니다.")
    arr = validate_features(X)
    validate_scaling(profile.scaling, profile.random_state)
    positive_int(profile.random_state, "random_state", 0)
    for key in ("rows", "columns", "n_iter"): positive_int(getattr(profile, key), key)
    finite_number(profile.learning_rate, "learning_rate", positive=True)
    finite_number(profile.sigma, "sigma", positive=True)
    if profile.learning_rate > 1: raise ValueError("learning_rate는 1 이하여야 합니다.")
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

class SOMTransformer(TransformerMixin, BaseEstimator):
    """신규 입력을 최적 일치 노드(BMU) 좌표로 변환하는 SOM."""

    def __init__(self, rows=5, columns=5, n_iter=1000, learning_rate=0.5, sigma=1.5, random_state=1234):
        """코드북 적합 조건을 저장합니다.

        Args:
            rows: 격자 행 수.
            columns: 격자 열 수.
            n_iter: 온라인 갱신 횟수.
            learning_rate: 초기 학습률.
            sigma: 초기 이웃 폭.
            random_state: 난수 시드.
        Returns:
            None: 설정 저장.
        Raises:
            None: 적합 시 설정 검증.
        """
        self.rows = rows
        self.columns = columns
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y=None):
        """표본 기반 초기화 후 Gaussian 이웃을 온라인 갱신합니다.

        Args:
            X: 유한한 수치 행렬.
            y: 미사용. sklearn 변환기 호환용.
        Returns:
            SOMTransformer: 적합 self.
        Raises:
            ValueError: 자료 또는 설정 오류.
        """
        arr = _validate_inputs(X, SOMProfile(scaling="none", **self.get_params()))
        rng = np.random.RandomState(self.random_state)
        self.n_features_in_ = arr.shape[1]
        self.coordinates_ = np.array([(r, c) for r in range(self.rows) for c in range(self.columns)])
        weights = arr[rng.randint(len(arr), size=len(self.coordinates_))].copy()
        for step in range(self.n_iter):
            sample = arr[rng.randint(len(arr))]
            bmu = np.argmin(((weights - sample) ** 2).sum(axis=1))
            decay = np.exp(-step / self.n_iter)
            radius = self.sigma * decay
            neighborhood = np.exp(-((self.coordinates_ - self.coordinates_[bmu]) ** 2).sum(axis=1) / (2 * radius ** 2))
            weights += self.learning_rate * decay * neighborhood[:, None] * (sample - weights)
        self.weights_ = weights.reshape(self.rows, self.columns, self.n_features_in_)
        return self

    def _distances(self, X):
        """새 관측과 코드북 사이 거리를 계산합니다.

        Args:
            X: 학습 열 순서의 특성. 단일 행도 허용.
        Returns:
            np.ndarray: (n_samples, rows*columns) 거리 행렬.
        Raises:
            NotFittedError: 미적합 코드북.
            ValueError: 열 수 불일치 또는 입력 오류.
        """
        check_is_fitted(self, "weights_")
        arr = validate_features(X, min_samples=1)
        if arr.shape[1] != self.n_features_in_: raise ValueError("학습 특성 수와 다릅니다.")
        from scipy.spatial.distance import cdist
        return cdist(arr, self.weights_.reshape(-1, self.n_features_in_))

    def transform(self, X):
        """각 행을 2차원 BMU 격자 좌표로 변환합니다.

        Args:
            X: 학습과 같은 열 순서의 특성.
        Returns:
            np.ndarray: (n_samples, 2) 정수 좌표. 같은 노드는 같은 좌표.
        Raises:
            NotFittedError/ValueError: 코드북 또는 입력 오류.
        """
        return self.coordinates_[self._distances(X).argmin(axis=1)].copy()

    def quantization_error(self, X):
        """평균 BMU 유클리드 거리를 계산합니다.

        Args:
            X: 코드북과 동일 좌표계의 특성.
        Returns:
            float: 평균 최소 거리.
        Raises:
            NotFittedError/ValueError: 코드북 또는 입력 오류.
        """
        return float(self._distances(X).min(axis=1).mean())

    def topographic_error(self, X):
        """두 최상위 BMU의 4-이웃 비인접 비율을 계산합니다.

        Args:
            X: 코드북과 동일 좌표계의 특성.
        Returns:
            float: 0~1 비율. 코드북 노드 1개이면 0.
        Raises:
            NotFittedError/ValueError: 코드북 또는 입력 오류.
        """
        distances = self._distances(X)
        if distances.shape[1] == 1: return 0.0
        best = np.argsort(distances, axis=1, kind="stable")[:, :2]
        delta = np.abs(self.coordinates_[best[:, 0]] - self.coordinates_[best[:, 1]]).sum(axis=1)
        return float(np.mean(delta != 1))


def _run_model(X: Any, profile: SOMProfile) -> SOMOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        profile: 검증된 SOMProfile.
    Returns:
        SOMOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    arr = validate_features(X)
    estimator = SOMTransformer(rows=profile.rows, columns=profile.columns, n_iter=profile.n_iter,
                               learning_rate=profile.learning_rate, sigma=profile.sigma, random_state=profile.random_state)
    model = Pipeline([("scaler", make_scaler(profile.scaling)), ("model", estimator)])
    embedding = model.fit_transform(arr)
    if not np.isfinite(embedding).all(): raise ValueError("변환 좌표가 비유한 값입니다. 입력 rank와 설정을 확인하세요.")
    scaled = model.named_steps["scaler"].transform(arr) if profile.scaling != "none" else arr
    return SOMOutputDTO(model, embedding, feature_names(X), True, estimator.weights_.copy(),
                        estimator.quantization_error(scaled), estimator.topographic_error(scaled))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_som(X: Any, profile: Optional[SOMProfile] = None, *,
              recipe_path: Optional[str] = None,
              overrides: Optional[Dict[str, Any]] = None) -> SOMOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        profile: 명시적 SOMProfile. None이면 som.json을 읽음.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        SOMOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile("som", profile, recipe_path=recipe_path, overrides=overrides)
    _validate_inputs(X, profile)
    return _run_model(X, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = run_som(X, load_profile("som"))
    print(result.to_json(indent=2)[:500])
