"""
================================================================================
tsne.py - t-SNE 시각화용 차원 축소
================================================================================
설명:
    - t-SNE 시각화용 차원 축소을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.
분석 가정:
    - 시각화 목적의 transductive 적합입니다. transform 및 새 관측 예측을 지원하지 않습니다.
    - 결측값과 범주형 인코딩은 실행 전에 처리합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-11
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-11  김재현(Finn)     v1.0.0    모델별 레시피·전체 명세·실행 예제 작성
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
class TSNEProfile:
    """t-SNE 시각화용 차원 축소 입력 조건.

    Attributes:
        scaling (str): 전처리 none/standard/minmax. 탐색용 실행에서는 입력 전체에 적합. 기본 "standard".
        random_state (int): 재현용 0 이상 정수 시드. 기본 1234.
        n_components (int): 출력 축 수. 알고리즘별 표본·특성·클래스 제약을 만족해야 함. 기본 2.
        perplexity (float): 유효 근방 크기. 유한한 양수이며 표본 수 미만. 기본 30.0.
        learning_rate (float): 최적화 학습률. 유한한 양수. 기본 200.0.
        max_iter (int): 최대 반복. 250 이상. 기본 1000.
    """

    scaling: str = "standard"
    random_state: int = 1234
    n_components: int = 2
    perplexity: float = 30.0
    learning_rate: float = 200.0
    max_iter: int = 1000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class TSNEOutputDTO(OutputDTO):
    """t-SNE 시각화용 차원 축소 결과. 배열은 원 입력 행 순서.

    Attributes:
        model (Any): 적합 Pipeline. 기본 JSON/사전에서 제외. 메서드 지원 범위는 모듈 명세 참고.
        embedding (np.ndarray): (n_samples, n_components) 좌표. SOM은 2차원 BMU 격자 좌표.
        feature_names (List[str]): 원 입력 열 순서의 이름.
        supports_transform (bool): 새 자료의 model.transform 지원 여부. False이면 재적합 필요.
        kl_divergence (float): 적합 시 KL 발산. 다른 perplexity의 좌표와 직접 성능 비교 불가.
    """

    model: Any
    embedding: np.ndarray
    feature_names: List[str]
    supports_transform: bool
    kl_divergence: float


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(X: Any, profile: TSNEProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        profile: TSNEProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, TSNEProfile):
        raise TypeError("TSNEProfile이 필요합니다.")
    arr = validate_features(X)
    validate_scaling(profile.scaling, profile.random_state)
    positive_int(profile.random_state, "random_state", 0)
    positive_int(profile.n_components, "n_components")
    finite_number(profile.perplexity, "perplexity", positive=True)
    finite_number(profile.learning_rate, "learning_rate", positive=True)
    positive_int(profile.max_iter, "max_iter", 250)
    if profile.perplexity >= len(arr):
        raise ValueError("perplexity는 표본 수 미만이어야 합니다.")
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_model(X: Any, profile: TSNEProfile) -> TSNEOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        profile: 검증된 TSNEProfile.
    Returns:
        TSNEOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    import inspect
    from sklearn.manifold import TSNE

    iteration_key = (
        "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    )
    estimator = TSNE(
        n_components=profile.n_components,
        perplexity=profile.perplexity,
        learning_rate=profile.learning_rate,
        init="random",
        method="exact",
        random_state=profile.random_state,
        **{iteration_key: profile.max_iter},
    )
    arr = validate_features(X)
    model = Pipeline([("scaler", make_scaler(profile.scaling)), ("model", estimator)])
    embedding = model.fit_transform(arr)
    if not np.isfinite(embedding).all():
        raise ValueError("변환 좌표가 비유한 값입니다. 입력 rank와 설정을 확인하세요.")
    return TSNEOutputDTO(
        model, embedding, feature_names(X), False, float(estimator.kl_divergence_)
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_tsne(
    X: Any,
    profile: Optional[TSNEProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TSNEOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        profile: 명시적 TSNEProfile. None이면 tsne.json을 읽음.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        TSNEOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile(
        "tsne", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_inputs(X, profile)
    return _run_model(X, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    result = run_tsne(X, load_profile("tsne"))
    print(result.to_json(indent=2)[:500])
