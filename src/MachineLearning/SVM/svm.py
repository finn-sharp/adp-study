"""
================================================================================
svm.py - 서포트 벡터 회귀·분류
================================================================================
설명:
    - 10장 서포트 벡터 머신 모델 및 지도학습 PDF를 기존 Profile → 실행 함수 → Result 구조로 재구성합니다.
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
from sklearn.svm import SVR, SVC
from src.MachineLearning.common import SupervisedProfile, SupervisedResult, run_supervised


@dataclass
class SVMProfile(SupervisedProfile):
    """서포트 벡터 회귀·분류 조건. 공통 필드는 SupervisedProfile 명세를 따릅니다.

    Attributes:
        task: regression, classification만 지원합니다.
        scaling: 기본 standard.
        C: 오차 페널티 계수 (양수) (기본 1.0).
        kernel: linear, poly, rbf 또는 sigmoid. precomputed는 지원하지 않음 (기본 "rbf").
        gamma: 커널 계수. scale, auto 또는 양수 (기본 "scale").
        epsilon: 회귀에서 epsilon 튜브 너비 (기본 0.1).
        probability: 분류 확률 추정 여부. 활성화하면 추가 비용 발생 (기본 False).
    """

    task: str = 'classification'
    scaling: str = 'standard'
    C: float = 1.0
    kernel: str = "rbf"
    gamma: Any = "scale"
    epsilon: float = 0.1
    probability: bool = False


@dataclass
class SVMOutputDTO(SupervisedResult):
    """SVM 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def run_svm(X: Any, y: Any, profile: Optional[SVMProfile] = None) -> SVMOutputDTO:
    """서포트 벡터 회귀·분류의 학습·테스트 평가 및 선택적 튜닝을 수행합니다.

    Args:
        X: 결측치 없는 수치형 (n_samples, n_features) 행렬 또는 DataFrame.
        y: 길이 n_samples의 1차원 타깃.
        profile: SVMProfile. None이면 기본 설정을 사용합니다.
    Returns:
        SVMOutputDTO: 적합 Pipeline, 학습/테스트 지표, 예측, 선택적 CV 결과.
    Raises:
        ValueError: 입력·task·모델 인자가 유효하지 않거나 표본 수가 부족한 경우.
    """
    profile = profile if profile is not None else SVMProfile()
    if profile.task not in ('regression', 'classification'):
        raise ValueError("이 모델은 regression, classification만 지원합니다.")
    if profile.kernel == "precomputed":
        raise ValueError("precomputed 커널은 자동 행 분할 실행기에서 지원하지 않습니다.")
    model = SVC(C=profile.C, kernel=profile.kernel, gamma=profile.gamma, random_state=profile.random_state, probability=profile.probability) if profile.task == "classification" else SVR(C=profile.C, kernel=profile.kernel, gamma=profile.gamma, epsilon=profile.epsilon)
    return SVMOutputDTO.from_result(run_supervised(X, y, model, profile))


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = run_svm(X, y)
    print(result.test_metrics)
