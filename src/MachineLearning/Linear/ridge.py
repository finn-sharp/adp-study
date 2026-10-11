"""
================================================================================
ridge.py - L2 규제 회귀
================================================================================
설명:
    - 05장 릿지 회귀 모델를 기존 Profile → 실행 함수 → Result 구조로 재구성합니다.
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
from sklearn.linear_model import Ridge
from src.MachineLearning.common import SupervisedProfile, SupervisedResult, run_supervised


@dataclass
class RidgeProfile(SupervisedProfile):
    """L2 규제 회귀 조건. 공통 필드는 SupervisedProfile 명세를 따릅니다.

    Attributes:
        task: regression만 지원합니다.
        scaling: 기본 standard.
        alpha: L2 규제 강도 (0 이상) (기본 1.0).
        solver: 선형 시스템 풀이 방식 (기본 "auto").
    """

    task: str = 'regression'
    scaling: str = 'standard'
    alpha: float = 1.0
    solver: str = "auto"


@dataclass
class RidgeOutputDTO(SupervisedResult):
    """Ridge 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def run_ridge(X: Any, y: Any, profile: Optional[RidgeProfile] = None) -> RidgeOutputDTO:
    """L2 규제 회귀의 학습·테스트 평가 및 선택적 튜닝을 수행합니다.

    Args:
        X: 결측치 없는 수치형 (n_samples, n_features) 행렬 또는 DataFrame.
        y: 길이 n_samples의 1차원 타깃.
        profile: RidgeProfile. None이면 기본 설정을 사용합니다.
    Returns:
        RidgeOutputDTO: 적합 Pipeline, 학습/테스트 지표, 예측, 선택적 CV 결과.
    Raises:
        ValueError: 입력·task·모델 인자가 유효하지 않거나 표본 수가 부족한 경우.
    """
    profile = profile if profile is not None else RidgeProfile()
    if profile.task not in ('regression',):
        raise ValueError("이 모델은 regression만 지원합니다.")
    model = Ridge(alpha=profile.alpha, solver=profile.solver, random_state=profile.random_state)
    return RidgeOutputDTO.from_result(run_supervised(X, y, model, profile))


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    result = run_ridge(X, y)
    print(result.test_metrics)
