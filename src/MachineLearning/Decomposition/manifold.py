"""
================================================================================
manifold.py - MDS, Isomap, LLE, t-SNE 탐색적 임베딩
================================================================================
설명: 14장 차원 축소 비교 예제를 탐색적 분석으로 분리합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 임베딩 실행기 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, Optional
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
from sklearn.pipeline import Pipeline
from src.MachineLearning.common import make_scaler, validate_features


@dataclass
class EmbeddingProfile:
    """탐색적 임베딩 조건.

    Attributes:
        method: mds, isomap, lle, tsne.
        n_components: 축 수.
        scaling: none, standard, minmax.
        random_state: 시드.
        n_neighbors: Isomap/LLE 이웃 수 (표본 수 미만).
        perplexity: t-SNE perplexity (양수, 표본 수 미만).
    """
    method: str = "tsne"
    n_components: int = 2
    scaling: str = "standard"
    random_state: int = 1234
    n_neighbors: int = 5
    perplexity: float = 30.0


@dataclass
class EmbeddingOutputDTO(OutputDTO):
    """탐색적 임베딩 결과.

    Attributes:
        model: 적합 Pipeline. MDS/t-SNE는 새 데이터 transform을 지원하지 않습니다.
        coordinates: 입력 행 순서의 축소 좌표. 예측 성능 지표가 아닙니다.
    """
    model: Any
    coordinates: Any


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
EmbeddingResult = EmbeddingOutputDTO


def run_embedding(X: Any, profile: Optional[EmbeddingProfile] = None) -> EmbeddingOutputDTO:
    """수치 특성을 탐색적 저차원 좌표로 변환합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        profile: 임베딩 설정.
    Returns:
        EmbeddingOutputDTO: 적합 모델과 좌표.
    Raises:
        ValueError: 잘못된 방법, perplexity/이웃 수 또는 입력.

    Notes:
        같은 데이터로 좌표 학습·분류·평가한 정확도를 일반화 성능으로 반환하지 않습니다.
    """
    p = profile if profile is not None else EmbeddingProfile()
    arr = validate_features(X)
    if p.method in ("isomap", "lle") and not 1 <= p.n_neighbors < len(arr):
        raise ValueError("n_neighbors는 1 이상 표본 수 미만이어야 합니다.")
    if p.method == "mds":
        estimator = MDS(n_components=p.n_components, random_state=p.random_state)
    elif p.method == "isomap":
        estimator = Isomap(n_components=p.n_components, n_neighbors=p.n_neighbors, eigen_solver="arpack")
    elif p.method == "lle":
        estimator = LocallyLinearEmbedding(n_components=p.n_components, n_neighbors=p.n_neighbors,
                                          eigen_solver="arpack", random_state=p.random_state)
    elif p.method == "tsne":
        if not 0 < p.perplexity < len(arr):
            raise ValueError("perplexity는 양수이며 표본 수 미만이어야 합니다.")
        estimator = TSNE(n_components=p.n_components, perplexity=p.perplexity,
                         random_state=p.random_state, init="random", learning_rate=200.0)
    else:
        raise ValueError("method는 mds, isomap, lle, tsne 중 하나여야 합니다.")
    pipeline = Pipeline([("scaler", make_scaler(p.scaling)), ("model", estimator)])
    return EmbeddingOutputDTO(pipeline, pipeline.fit_transform(arr))
