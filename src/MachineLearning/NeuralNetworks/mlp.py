"""
================================================================================
mlp.py - 다층 퍼셉트론 회귀·분류
================================================================================
설명:
    - 11장 다층 퍼셉트론 모델 및 지도학습 PDF를 기존 Profile → 실행 함수 → Result 구조로 재구성합니다.
    - 공통 분할·스케일링·교차검증·평가는 common.py에 위임합니다.
작성 정보:
    - 작성일: 2026-10-10
    - 버전: 1.0.0
변경 이력:
    2026-10-10  v1.0.0  모델 실행기, 입력 조건 및 결과 명세 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 모듈 전용 결과 DTO 및 저장용 변환 추가
"""

__version__ = "1.1.0"
__status__ = "Development"

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from sklearn.neural_network import MLPRegressor, MLPClassifier
from src.MachineLearning.common import SupervisedProfile, SupervisedResult, run_supervised


@dataclass
class MLPProfile(SupervisedProfile):
    """다층 퍼셉트론 회귀·분류 조건. 공통 필드는 SupervisedProfile 명세를 따릅니다.

    Attributes:
        task: regression, classification만 지원합니다.
        scaling: 기본 standard.
        hidden_layer_sizes: 은닉층별 뉴런 수 (기본 (20, 10)).
        solver: lbfgs, sgd 또는 adam (기본 "adam").
        activation: 은닉층 활성화 함수 (기본 "relu").
        alpha: L2 규제 강도 (기본 0.0001).
        max_iter: 최대 반복 수 (기본 1000).
        early_stopping: 학습 데이터 내 검증 분할로 조기 종료. 테스트 데이터는 사용하지 않음 (기본 False).
    """

    task: str = 'classification'
    scaling: str = 'standard'
    hidden_layer_sizes: Tuple[int, ...] = (20, 10)
    solver: str = "adam"
    activation: str = "relu"
    alpha: float = 0.0001
    max_iter: int = 1000
    early_stopping: bool = False


@dataclass
class MLPOutputDTO(SupervisedResult):
    """MLP 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def run_mlp(X: Any, y: Any, profile: Optional[MLPProfile] = None) -> MLPOutputDTO:
    """다층 퍼셉트론 회귀·분류의 학습·테스트 평가 및 선택적 튜닝을 수행합니다.

    Args:
        X: 결측치 없는 수치형 (n_samples, n_features) 행렬 또는 DataFrame.
        y: 길이 n_samples의 1차원 타깃.
        profile: MLPProfile. None이면 기본 설정을 사용합니다.
    Returns:
        MLPOutputDTO: 적합 Pipeline, 학습/테스트 지표, 예측, 선택적 CV 결과.
    Raises:
        ValueError: 입력·task·모델 인자가 유효하지 않거나 표본 수가 부족한 경우.
    """
    profile = profile if profile is not None else MLPProfile()
    if profile.task not in ('regression', 'classification'):
        raise ValueError("이 모델은 regression, classification만 지원합니다.")
    estimator_class = MLPClassifier if profile.task == "classification" else MLPRegressor
    model = estimator_class(hidden_layer_sizes=profile.hidden_layer_sizes, solver=profile.solver, activation=profile.activation, alpha=profile.alpha, max_iter=profile.max_iter, early_stopping=profile.early_stopping, random_state=profile.random_state)
    return MLPOutputDTO.from_result(run_supervised(X, y, model, profile))


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = run_mlp(X, y)
    print(result.test_metrics)
