"""
================================================================================
cnn.py - TensorFlow CNN 1D·2D 전용 분류·회귀
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 독립 이미지/시퀀스 표본; 전처리와 조기 종료는 학습/검증 자료만 사용.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    입력 검증·전용 DTO·레시피·예제 추가
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
from src.MachineLearning.analysis_common import positive_int, finite_number
from .common import numeric_X
from ._deep_common import validate_training, target_array, fit_supervised


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class TFCNNProfile:
    """TFCNNProfile 필드 계약.

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
        task (str): classification/regression. 기본 classification.
        dimension (int): 1은 (표본,시간,채널), 2는 (표본,높이,너비,채널); 기본 2.
        conv_filters (Tuple[int, ...]): 합성곱 블록의 양수 필터 수; 기본 (16,32).
        kernel_size (int): same padding 합성곱 크기; 양수, 기본 3.
        pool_size (int): 블록별 valid 최대풀링 크기; 양수, 기본 2.
        hidden_layer_sizes (Tuple[int, ...]): Flatten 이후 Dense 폭; 기본 (32,).
        dropout (float): 각 Dense 이후 dropout 비율 [0,1); 기본 0.
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
    task: str = "classification"
    dimension: int = 2
    conv_filters: Tuple[int, ...] = (16, 32)
    kernel_size: int = 3
    pool_size: int = 2
    hidden_layer_sizes: Tuple[int, ...] = (32,)
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class TFCNNOutputDTO(OutputDTO):
    """TFCNNOutputDTO 필드 계약.

    Attributes:
        task (str): classification/regression. 시계열은 regression.
        split_mode (str): random_holdout/provided_test; 고정 test에서는 test_size를 사용하지 않음.
        parameters (Dict[str, Any]): 적용 Profile 및 effective_test_fraction/effective_validation_fraction.
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
    split_mode: str


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(X: Any, y: Any, profile: TFCNNProfile) -> tuple:
    """전용 설정과 모델의 rank·축·타깃을 검증합니다.

    Args:
        X: 모델별 CNN/순환 수치 입력.
        y: 원 분류 라벨 또는 회귀 타깃.
        profile: TFCNNProfile.
    Returns:
        tuple: 검증한 float32 입력과 타깃.
    Raises:
        ValueError/TypeError: Profile·설정·shape·타깃 오류.
    """
    if not isinstance(profile, TFCNNProfile):
        raise TypeError("TFCNNProfile이 필요합니다.")
    validate_training(profile)
    if np.iscomplexobj(X):
        raise ValueError("실수 입력이 필요합니다.")
    positive_int(profile.dimension, "dimension")
    if profile.dimension not in (1, 2):
        raise ValueError("dimension은 1/2입니다.")
    for field in ("conv_filters", "hidden_layer_sizes"):
        values = getattr(profile, field)
        if not isinstance(values, tuple) or (field == "conv_filters" and not values):
            raise ValueError(field + "는 tuple입니다.")
        for width in values:
            positive_int(width, field)
    positive_int(profile.kernel_size, "kernel_size")
    positive_int(profile.pool_size, "pool_size")
    arr = numeric_X(X, profile.dimension + 2)
    for width in arr.shape[1:-1]:
        if width // (profile.pool_size ** len(profile.conv_filters)) < 1:
            raise ValueError("풀링 블록 이후 공간/시간 축이 사라집니다.")
    return arr, target_array(y, len(arr), profile.task)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _build_model(tf, input_shape, output_shape, classification, profile):
    """해당 파일의 전용 TFCNN 아키텍처를 구성합니다.

    Args:
        tf: 로딩한 TensorFlow.
        input_shape: batch 이외 입력 shape.
        output_shape: 출력 클래스/회귀 shape.
        classification: 분류 여부.
        profile: 전용 Profile.
    Returns:
        object: 미적합 Keras Sequential.
    Raises:
        ValueError: 아키텍처 shape·층 조건 오류.
    """
    layers = [tf.keras.layers.Input(shape=input_shape)]
    Conv = tf.keras.layers.Conv1D if profile.dimension == 1 else tf.keras.layers.Conv2D
    Pool = (
        tf.keras.layers.MaxPooling1D
        if profile.dimension == 1
        else tf.keras.layers.MaxPooling2D
    )
    for filters in profile.conv_filters:
        layers.append(
            Conv(filters, profile.kernel_size, padding="same", activation="relu")
        )
        layers.append(Pool(pool_size=profile.pool_size))
    layers.append(tf.keras.layers.Flatten())
    for width in profile.hidden_layer_sizes:
        layers.append(tf.keras.layers.Dense(width, activation="relu"))
        if profile.dropout:
            layers.append(tf.keras.layers.Dropout(profile.dropout))
    layers.append(
        tf.keras.layers.Dense(
            int(np.prod(output_shape)), activation="softmax" if classification else None
        )
    )
    if len(output_shape) > 1:
        layers.append(tf.keras.layers.Reshape(output_shape))
    return tf.keras.Sequential(layers)


def _run_model(X, y, profile, test_data=None):
    """공통 학습 계약을 전용 아키텍처·DTO와 연결합니다.

    Args:
        X: 검증한 입력.
        y: 검증한 타깃.
        profile: 전용 설정.
        test_data: 명시적인 고정 보류 자료 또는 None.
    Returns:
        TFCNNOutputDTO: 원 라벨/단위 예측과 학습 진단.
    Raises:
        ValueError/ImportError: 학습·분할 오류 또는 TensorFlow 미설치.
    """
    builder = lambda tf, shape, output, classification: _build_model(
        tf, shape, output, classification, profile
    )
    return TFCNNOutputDTO(**fit_supervised(X, y, profile, builder, test_data))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_tf_cnn(
    X: Any,
    y: Any,
    profile: Optional[TFCNNProfile] = None,
    *,
    task: Optional[str] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    test_data: Any = None,
) -> TFCNNOutputDTO:
    """TFCNN 분류·회귀를 전용 JSON/Profile로 실행합니다.

    Args:
        X: CNN은 (n,time,channel)/(n,height,width,channel), 순환은 (n,time,feature).
        y: 1차원 분류 라벨 또는 1/2차원 회귀값.
        profile: 명시적 전용 Profile 또는 None.
        task: Profile 생략 시 classification/regression 목적.
        recipe_path: 사용자 JSON 경로.
        overrides: Profile 생략 시 명시적 재정의.
        test_data: 고정 (X_test,y_test); 제공 시 validation_size는 X 안의 비율, test_size 미사용.
    Returns:
        TFCNNOutputDTO: 원 클래스/단위 예측·분할·실제 비율·학습 이력.
    Raises:
        ValueError/TypeError/OSError/ImportError: 자료·설정·레시피·TensorFlow 오류.
    """
    profile = resolve_profile(
        "tf-cnn", profile, task=task, recipe_path=recipe_path, overrides=overrides
    )
    X, y = _validate_inputs(X, y, profile)
    return _run_model(X, y, profile, test_data)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(4)
    X = rng.normal(size=(80, 8, 8, 1))
    y = np.where(X[:, 0, 0, 0] > 0, "high", "low")
    result = run_tf_cnn(X, y, overrides={"epochs": 2, "patience": 1})
    print(result.test_metrics, result.input_shape)
