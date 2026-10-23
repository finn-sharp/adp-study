"""
================================================================================
time_series.py - TensorFlow LSTM 시계열 예측
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 유한한 수치 입력; 시계열은 시간 순서로 정렬. 전처리와 조기 종료에 보류 test를 사용하지 않음.
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
from .common import validate_profile, run_supervised, run_forecast


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class TFTimeSeriesProfile:
    """TFTimeSeriesProfile 필드 계약.

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
        cell (str): simple_rnn/lstm/gru 셀 선택. 기본 lstm.
        units (int): 순환 층 폭, 양수. 기본 16.
        input_width (int): 관측 입력 창 길이. 기본 12.
        horizon (int): 미래 예측 타임스텝 수. 기본 1.
        target_columns (Optional[Tuple[int, ...]]): 예측할 열 위치; None은 모든 입력 열. 기본 None.
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
    cell: str = "lstm"
    units: int = 16
    input_width: int = 12
    horizon: int = 1
    target_columns: Optional[Tuple[int, ...]] = None


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class TFTimeSeriesOutputDTO(OutputDTO):
    """TFTimeSeriesOutputDTO 필드 계약.

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


def _validate_inputs(profile: Optional[TFTimeSeriesProfile]) -> None:
    """설정을 검증합니다.

    Args:
        profile: TFTimeSeriesProfile.
    Returns:
        None: 유효하면 반환.
    Raises:
        TypeError/ValueError: 잘못된 타입·범위.
    """
    if not isinstance(profile, TFTimeSeriesProfile):
        raise TypeError("TFTimeSeriesProfile이 필요합니다.")
    validate_profile(profile, "time_series")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_model(
    series: Any, profile: Optional[TFTimeSeriesProfile]
) -> TFTimeSeriesOutputDTO:
    """검증된 모델을 적합합니다.

    Args:
        series: 수치 입력과 타깃(시계열은 원 시계열).
        profile: 학습 조건.
    Returns:
        TFTimeSeriesOutputDTO: 모델·학습 이력·보류 결과.
    Raises:
        ValueError: 자료 shape·분할·타깃 오류.
        ImportError: TensorFlow 미설치.
    """
    return TFTimeSeriesOutputDTO(**run_forecast(series, profile))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_tf_time_series(
    series: Any,
    profile: Optional[TFTimeSeriesProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TFTimeSeriesOutputDTO:
    """TensorFlow LSTM 시계열 예측 실행.

    Args:
        series: series는 (시점,특성), 시간 순 정렬.
        profile: 명시적 TFTimeSeriesProfile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피에 적용할 설정.
    Returns:
        TFTimeSeriesOutputDTO: test 성능 및 원 타깃 순서 결과.
    Raises:
        ValueError/TypeError: 입력·설정 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: TensorFlow 미설치.
    """
    profile = resolve_profile(
        "tf-time-series", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_inputs(profile)
    return _run_model(series, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    series = np.sin(np.arange(160) / 8.0)[:, None]
    result = run_tf_time_series(series, overrides={"epochs": 2, "input_width": 6})
    print(result.test_metrics, result.epochs_trained)
