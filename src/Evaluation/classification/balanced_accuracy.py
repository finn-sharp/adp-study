"""
================================================================================
balanced_accuracy.py - classification balanced_accuracy 평가
================================================================================
설명:
    - 독립 평가 실행기; 학습·분할·그림 생성 없이 입력 예측만 평가합니다.
분석 가정:
    - 정답과 예측은 같은 행/타깃 순서입니다. 평가 자료는 모델 선택과 구분합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-13
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-13  김재현(Finn)      v1.0.0    독립 메트릭·JSON·Profile·Output DTO
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# 1. Imports
from dataclasses import dataclass, field
from typing import Any
from src.output import OutputDTO
from src.Recipes.loader import resolve_profile
from src.Evaluation._common import validate_profile, evaluate, example_inputs


# 2. Input Profile
@dataclass
class ClassificationBalancedAccuracyProfile:
    """평가 설정.

    Attributes:
        average (str): binary/micro/macro/weighted/None 집계; None은 클래스별 배열.
        positive_label (Any): binary 지표·순위 확률의 양성 클래스.
        labels (Any): 확률 열의 클래스 순서. 행렬 확률은 반드시 지정.
        zero_division (int): precision/recall/F의 0 분모 대체값: 0 또는 1.
        beta (float): F-beta의 유한 양수 beta.
        top_fractions (Any): Lift/Gain 선별 비율 (0,1]; ceil(n*f), 동률은 원 행 순서.
        n_bins (int): calibration 동일 너비 bin 수, 양의 정수.
    """

    average: str = "weighted"
    positive_label: Any = 1
    labels: Any = None
    zero_division: int = 0
    beta: float = 1.0
    top_fractions: Any = field(default_factory=lambda: [0.1, 0.2, 0.3])
    n_bins: int = 10


# 3. Output DTO
@dataclass
class ClassificationBalancedAccuracyOutputDTO(OutputDTO):
    """직렬화 가능한 독립 평가 결과.

    Attributes:
        metric_name (str): 평가 목적 또는 메트릭 이름.
        value (Any): 스칼라·클래스별 배열·곡선 값·묶음 지표 사전.
        n_samples (int): 실제 평가 행 수; 군집은 잡음 제외 후 행 수.
        details (dict): 클래스/확률 열 순서·표·정의·집계·적용 설정.
        warnings (list): 정의 불가 지표 사유; 묶음은 NaN(JSON null)으로 보존.
    """

    metric_name: str
    value: Any
    n_samples: int
    details: dict
    warnings: list


# 4. Input Validation
def validate_inputs(profile):
    """전용 Profile 및 설정의 정의역을 검사합니다.

    Args:
        profile: ClassificationBalancedAccuracyProfile.
    Returns:
        None: 정상 설정.
    Raises:
        TypeError/ValueError: Profile 타입·필드 오류.
    """
    validate_profile(profile, ClassificationBalancedAccuracyProfile, "classification")


# 5. Individual Model Runners
def _run(y_true, y_pred, y_score, X, y_train, profile):
    """공통 계산에서 해당 메트릭만 실행하고 전용 DTO로 반환합니다.

    Args:
        y_true: 1D 라벨 또는 수치 정답; 내부 군집 지표는 None 가능.
        y_pred: 대응 예측 라벨/수치; 군집은 예측 군집 라벨.
        y_score: 양성 점수 1D 또는 labels 순서 확률 행렬.
        X: 내부 군집 평가에 사용한 동일 전처리 좌표.
        y_train: MASE/RMSSE 기준 학습 시계열 (시간,타깃).
        profile: 전용 평가 설정.
    Returns:
        ClassificationBalancedAccuracyOutputDTO: 수치 평가.
    Raises:
        ValueError/TypeError: 입력 정의역·정의 불가 단일 메트릭.
    """
    return ClassificationBalancedAccuracyOutputDTO(
        **evaluate(
            "classification",
            "balanced_accuracy",
            y_true,
            y_pred,
            y_score,
            X,
            y_train,
            profile,
        )
    )


# 6. Public Runner
def run_balanced_accuracy(
    y_true=None,
    y_pred=None,
    profile=None,
    *,
    y_score=None,
    X=None,
    y_train=None,
    recipe_path=None,
    overrides=None,
):
    """JSON → overrides → Profile로 독립 평가를 실행합니다.

    Args:
        y_true: 같은 행 순서의 정답; 내부 군집 평가만 None 가능.
        y_pred: 예측값 또는 군집 라벨; 확률/순위 전용 지표는 None 가능.
        profile: 명시적 ClassificationBalancedAccuracyProfile 또는 None.
        y_score: 양성 점수 또는 labels 열 순서의 확률 행렬.
        X: 내부 군집 지표의 전처리 좌표 (n,p).
        y_train: MASE/RMSSE 분모 계산용 학습 시계열; 평가 자료 사용 금지.
        recipe_path: 사용자 JSON; Profile과 동시 지정 금지.
        overrides: JSON 재정의; Profile과 동시 지정 금지.
    Returns:
        ClassificationBalancedAccuracyOutputDTO: value·표/순서·경고·JSON 변환 지원.
    Raises:
        ValueError/TypeError: 설정·입력 오류; 정의 불가 단일 지표.
        OSError: JSON 읽기 오류.
    """
    p = resolve_profile(
        "metric-classification-balanced-accuracy",
        profile,
        recipe_path,
        overrides=overrides,
    )
    validate_inputs(p)
    return _run(y_true, y_pred, y_score, X, y_train, p)


# 7. Example Execution
if __name__ == "__main__":
    print(run_balanced_accuracy(**example_inputs("classification")).to_json(indent=2))
