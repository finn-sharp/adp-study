"""
================================================================================
naive_bayes.py - 가우시안 나이브 베이즈 분류
================================================================================
설명:
    - 지도학습 PDF의 GaussianNB 예제를 기존 Profile → 실행 함수 → Result 구조로 재구성합니다.
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from src.MachineLearning.common import SupervisedProfile, SupervisedResult, run_supervised


@dataclass
class NaiveBayesProfile(SupervisedProfile):
    """가우시안 나이브 베이즈 분류 조건. 공통 필드는 SupervisedProfile 명세를 따릅니다.

    Attributes:
        task: classification만 지원합니다.
        scaling: 기본 none.
        var_smoothing: 분산 안정화를 위한 양의 값 (기본 1e-9).
        variant: gaussian, multinomial, bernoulli. multinomial은 비음수 특성이 필요합니다.
        alpha: multinomial/bernoulli 평활화 강도 (양수).
        binarize: bernoulli 이진화 임계값. None이면 이미 0/1인 특성이 필요합니다.
    """

    task: str = 'classification'
    scaling: str = 'none'
    var_smoothing: float = 1e-9
    variant: str = 'gaussian'
    alpha: float = 1.0
    binarize: Optional[float] = 0.0


@dataclass
class NaiveBayesOutputDTO(SupervisedResult):
    """NaiveBayes 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def run_naive_bayes(X: Any, y: Any, profile: Optional[NaiveBayesProfile] = None) -> NaiveBayesOutputDTO:
    """가우시안 나이브 베이즈 분류의 학습·테스트 평가 및 선택적 튜닝을 수행합니다.

    Args:
        X: 결측치 없는 수치형 (n_samples, n_features) 행렬 또는 DataFrame.
        y: 길이 n_samples의 1차원 타깃.
        profile: NaiveBayesProfile. None이면 기본 설정을 사용합니다.
    Returns:
        NaiveBayesOutputDTO: 적합 Pipeline, 학습/테스트 지표, 예측, 선택적 CV 결과.
    Raises:
        ValueError: 입력·task·모델 인자가 유효하지 않거나 표본 수가 부족한 경우.
    """
    profile = profile if profile is not None else NaiveBayesProfile()
    if profile.task not in ('classification',):
        raise ValueError("이 모델은 classification만 지원합니다.")
    if profile.variant == 'gaussian':
        model = GaussianNB(var_smoothing=profile.var_smoothing)
    elif profile.variant == 'multinomial':
        model = MultinomialNB(alpha=profile.alpha)
    elif profile.variant == 'bernoulli':
        model = BernoulliNB(alpha=profile.alpha, binarize=profile.binarize)
    else:
        raise ValueError("variant는 gaussian, multinomial, bernoulli 중 하나여야 합니다.")
    return NaiveBayesOutputDTO.from_result(run_supervised(X, y, model, profile))


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = run_naive_bayes(X, y)
    print(result.test_metrics)
