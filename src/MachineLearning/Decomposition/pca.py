"""
================================================================================
pca.py - PCA/요인분석 및 PCA-로지스틱 Pipeline
================================================================================
설명: 14장 주성분 분석과 통계분석 PDF의 차원 축소를 모듈화합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 차원 축소 및 학습 폴드 내부 PCA 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""

__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.MachineLearning.common import make_scaler, run_supervised, validate_features, SupervisedResult
from src.MachineLearning.Linear.logistic import LogisticProfile


@dataclass
class DecompositionProfile:
    """탐색적 차원 축소 조건.

    Attributes:
        method: pca 또는 factor_analysis.
        n_components: 축 수. PCA는 0~1 설명분산 비율도 지원 (svd_solver='full').
        scaling: none, standard, minmax. 전달한 전체 데이터에서 적합.
        whiten: PCA 축의 분산 정규화 여부. 요인분석에는 사용하지 않음.
        random_state: 시드.
    """
    method: str = "pca"
    n_components: Any = 2
    scaling: str = "standard"
    whiten: bool = False
    random_state: int = 1234


@dataclass
class DecompositionOutputDTO(OutputDTO):
    """차원 축소 결과.

    Attributes:
        model: 적합 Pipeline. 새 입력에는 model.transform을 사용합니다.
        transformed: 입력 행 순서의 축소 좌표.
        components: 축별 원 특성 계수 (스케일링한 특성 공간).
        explained_variance_ratio: PCA 설명분산 비율. 요인분석은 None.
        cumulative_variance_ratio: PCA 누적 설명분산 비율. 요인분석은 None.
    """
    model: Any
    transformed: np.ndarray
    components: np.ndarray
    explained_variance_ratio: Optional[np.ndarray]
    cumulative_variance_ratio: Optional[np.ndarray]


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
DecompositionResult = DecompositionOutputDTO


@dataclass
class PCAClassificationOutputDTO(SupervisedResult):
    """PCAClassification 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def run_decomposition(X: Any, profile: Optional[DecompositionProfile] = None) -> DecompositionOutputDTO:
    """전체 입력의 탐색적 차원 축소를 수행합니다.

    Args:
        X: 결측값 없는 수치형 2차원 행렬.
        profile: 차원 축소 조건. None이면 표준화 PCA 2축.
    Returns:
        DecompositionOutputDTO: Pipeline, 좌표, 성분 및 선택적 설명분산.
    Raises:
        ValueError: 잘못된 method·축 수 또는 데이터.

    Notes:
        지도 학습 평가에는 전체 데이터에서 이 함수를 호출하지 말고 run_pca_classification을 사용합니다.
    """
    p = profile if profile is not None else DecompositionProfile()
    arr = validate_features(X)
    if p.method == "pca":
        estimator = PCA(n_components=p.n_components, whiten=p.whiten, svd_solver="full", random_state=p.random_state)
    elif p.method == "factor_analysis":
        estimator = FactorAnalysis(n_components=p.n_components, random_state=p.random_state)
    else:
        raise ValueError("method는 pca 또는 factor_analysis여야 합니다.")
    pipeline = Pipeline([("scaler", make_scaler(p.scaling)), ("model", estimator)])
    transformed = pipeline.fit_transform(arr)
    ratio = getattr(estimator, "explained_variance_ratio_", None)
    return DecompositionOutputDTO(pipeline, transformed, estimator.components_, ratio,
                               np.cumsum(ratio) if ratio is not None else None)


@dataclass
class PCAClassificationProfile(LogisticProfile):
    """PCA와 로지스틱 분류의 조건.

    Attributes:
        n_components: 학습 데이터에서 적합할 PCA 축 수 또는 설명분산 비율.
        param_grid: 상속 공통 필드. pca__n_components와 model__C를 동시에 탐색할 수 있음.
        기타: LogisticProfile 및 SupervisedProfile 필드를 상속합니다.
    """
    n_components: Any = 2


def run_pca_classification(X: Any, y: Any,
                           profile: Optional[PCAClassificationProfile] = None) -> PCAClassificationOutputDTO:
    """스케일러 → PCA → 로지스틱 분류를 학습 데이터/CV 폴드 내부에서 적합합니다.

    Args:
        X: 결측값 없는 수치형 2차원 특성.
        y: 1차원 분류 라벨.
        profile: PCA·분류·CV 설정.
    Returns:
        PCAClassificationOutputDTO: 보류 테스트 지표와 적합 Pipeline.
    Raises:
        ValueError: 분류 이외 task, 잘못된 축 수 또는 학습 조건.
    """
    p = profile if profile is not None else PCAClassificationProfile()
    if p.task != "classification":
        raise ValueError("PCA 분류 실행기는 classification만 지원합니다.")
    estimator = LogisticRegression(C=p.C, penalty=p.penalty, solver=p.solver,
              max_iter=p.max_iter, class_weight=p.class_weight, l1_ratio=p.l1_ratio,
              random_state=p.random_state)
    return PCAClassificationOutputDTO.from_result(run_supervised(X, y, estimator, p,
                          extra_steps=[("pca", PCA(n_components=p.n_components, svd_solver="full"))]))


if __name__ == "__main__":
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    print(run_pca_classification(X, y).test_metrics)
