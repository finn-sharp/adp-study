"""
================================================================================
xgboost.py - XGBoost 회귀·분류
================================================================================
설명:
    - 지도학습 PDF의 XGBoost 예제를 기존 Profile → 실행 함수 → Result 구조로 재구성합니다.
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
from src.MachineLearning.common import SupervisedProfile, SupervisedResult, run_supervised


@dataclass
class XGBoostProfile(SupervisedProfile):
    """XGBoost 회귀·분류 조건. 공통 필드는 SupervisedProfile 명세를 따릅니다.

    Attributes:
        task: regression, classification만 지원합니다.
        scaling: 기본 none.
        n_estimators: 부스팅 단계 수 (기본 100).
        max_depth: 트리 최대 깊이 (기본 3).
        learning_rate: 학습률 (기본 0.1).
        subsample: 표본 비율 (기본 1.0).
        colsample_bytree: 특성 비율 (기본 1.0).
    """

    task: str = 'classification'
    scaling: str = 'none'
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0


@dataclass
class XGBoostOutputDTO(SupervisedResult):
    """XGBoost 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def run_xgboost(X: Any, y: Any, profile: Optional[XGBoostProfile] = None) -> XGBoostOutputDTO:
    """XGBoost 회귀·분류의 학습·테스트 평가 및 선택적 튜닝을 수행합니다.

    Args:
        X: 결측치 없는 수치형 (n_samples, n_features) 행렬 또는 DataFrame.
        y: 길이 n_samples의 1차원 타깃.
        profile: XGBoostProfile. None이면 기본 설정을 사용합니다.
    Returns:
        XGBoostOutputDTO: 적합 Pipeline, 학습/테스트 지표, 예측, 선택적 CV 결과.
    Raises:
        ValueError: 입력·task·모델 인자가 유효하지 않거나 표본 수가 부족한 경우.
        ImportError: 선택적 xgboost 패키지가 설치되지 않은 경우.
    """
    profile = profile if profile is not None else XGBoostProfile()
    if profile.task not in ('regression', 'classification'):
        raise ValueError("이 모델은 regression, classification만 지원합니다.")
    try:
        from xgboost import XGBRegressor, XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost 패키지를 먼저 설치하세요.") from exc
    estimator_class = XGBClassifier if profile.task == "classification" else XGBRegressor
    if profile.task == "classification":
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        target = np.asarray(y)
        if target.ndim != 1 or not np.array_equal(target, LabelEncoder().fit_transform(target)):
            raise ValueError("XGBoost 분류 라벨은 0부터 시작하는 연속 정수여야 합니다.")
    model = estimator_class(n_estimators=profile.n_estimators, max_depth=profile.max_depth, learning_rate=profile.learning_rate, subsample=profile.subsample, colsample_bytree=profile.colsample_bytree, random_state=profile.random_state, n_jobs=profile.n_jobs)
    if profile.task == "regression":
        model.set_params(objective="reg:squarederror")
    return XGBoostOutputDTO.from_result(run_supervised(X, y, model, profile))


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = run_xgboost(X, y)
    print(result.test_metrics)
