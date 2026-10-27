"""
================================================================================
classification.py - TensorFlow Dense 분류
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 유한한 수치 입력; 시계열은 시간 순서로 정렬. 전처리와 조기 종료에 보류 test를 사용하지 않음.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    입력 검증·전용 DTO·레시피·예제 추가
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.output import OutputDTO
from src.Recipes import resolve_profile
from .common import validate_profile, run_supervised, run_forecast


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class TFClassificationProfile:
    """TFClassificationProfile 필드 계약.

    Attributes:
        random_state (int): 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234.
        epochs (int): 최대 학습 epoch, 양수. 기본 30.
        batch_size (int): 학습 batch 크기, 양수. 기본 32.
        learning_rate (float): Adam 학습률, 양수. 기본 0.001.
        patience (int): validation loss 조기 종료 대기, 0 이상. 기본 5.
        test_size (float): 전체 표본/시점 중 test 비율. 기본 0.2.
        validation_size (float): 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2.
        scaling (str): none/standard; 학습 입력만 적합. 기본 standard.
        verbose (int): Keras 출력 0/1/2. 기본 0.
        hidden_layer_sizes (Tuple[int, ...]): 은닉 Dense 폭, 양수 정수 tuple. 빈 tuple은 선형 출력. 기본 (32,16).
    """

    random_state: int = 1234
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    scaling: str = "standard"
    verbose: int = 0
    hidden_layer_sizes: Tuple[int, ...] = (32, 16)


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class TFClassificationOutputDTO(OutputDTO):
    """TFClassificationOutputDTO 필드 계약.

    Attributes:
        task (str): classification/regression. 시계열은 regression.
        parameters (Dict[str, Any]): 실제 적용한 Profile 설정. 레시피 및 overrides 반영.
        model (Any): 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외.
        history (Dict[str, List[float]]): epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록.
        test_metrics (Dict[str, float]): 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위.
        predictions (np.ndarray): test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target).
        probabilities (Optional[np.ndarray]): 분류 (n,classes) 확률; 회귀 None.
        classes (Optional[np.ndarray]): 확률 열 순서의 학습 클래스; 회귀 None.
        train_indices (np.ndarray): 학습 표본 번호; 시계열은 원 시점 번호.
        validation_indices (np.ndarray): 조기 종료 전용 검증 표본/시점 번호.
        test_indices (np.ndarray): 보류 평가 표본/시점 번호.
        target_indices (Optional[np.ndarray]): 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None.
        y_test (np.ndarray): predictions와 같은 shape/순서의 실제 test 타깃.
        input_shape (Tuple[int, ...]): Keras 입력 shape(batch 제외).
        epochs_trained (int): 실제 실행한 epoch 수.
        best_epoch (int): 최저 val_loss epoch, 1부터 시작. 최적 weights 복원.
    """

    task: str
    parameters: Dict[str, Any]
    model: Any
    history: Dict[str, List[float]]
    test_metrics: Dict[str, float]
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    classes: Optional[np.ndarray]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    target_indices: Optional[np.ndarray]
    y_test: np.ndarray
    input_shape: Tuple[int, ...]
    epochs_trained: int
    best_epoch: int


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(profile: Optional[TFClassificationProfile]) -> None:
    """설정을 검증합니다.

    Args:
        profile: TFClassificationProfile.
    Returns:
        None: 유효하면 반환.
    Raises:
        TypeError/ValueError: 잘못된 타입·범위.
    """
    if not isinstance(profile, TFClassificationProfile):
        raise TypeError("TFClassificationProfile이 필요합니다.")
    validate_profile(profile, "classification")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_model(
    X: Any, y: Any, profile: Optional[TFClassificationProfile]
) -> TFClassificationOutputDTO:
    """검증된 모델을 적합합니다.

    Args:
        X: 수치 입력, Dense 2차원 또는 RNN 3차원.
        y: 표본 순서의 분류 라벨 또는 회귀 타깃.
        profile: 학습 조건.
    Returns:
        TFClassificationOutputDTO: 모델·학습 이력·보류 결과.
    Raises:
        ValueError: 자료 shape·분할·타깃 오류.
        ImportError: TensorFlow 미설치.
    """
    return TFClassificationOutputDTO(**run_supervised(X, y, profile, "classification"))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_tf_classification(
    X: Any,
    y: Any,
    profile: Optional[TFClassificationProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TFClassificationOutputDTO:
    """TensorFlow Dense 분류 실행.

    Args:
        X: Dense (표본,특성), RNN (표본,시간,특성).
        y: 1차원 분류 라벨 또는 (표본,)/(표본,타깃) 회귀값.
        profile: 명시적 TFClassificationProfile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피에 적용할 설정.
    Returns:
        TFClassificationOutputDTO: test 성능 및 원 타깃 순서 결과.
    Raises:
        ValueError/TypeError: 입력·설정 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: TensorFlow 미설치.
    """
    profile = resolve_profile(
        "tf-classification", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_inputs(profile)
    return _run_model(X, y, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] > 0).astype(int)
    result = run_tf_classification(X, y, overrides={"epochs": 2})
    print(result.test_metrics, result.epochs_trained)
