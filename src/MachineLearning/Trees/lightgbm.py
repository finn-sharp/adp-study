"""
================================================================================
lightgbm.py - LightGBM 회귀·분류
================================================================================
설명:
    - LightGBM 회귀·분류의 학습·평가·학습 구간 교차검증을 제공합니다.
    - 분석 조건은 Profile, 분석 결과는 Output DTO로 정의합니다.
    - Profile 생략 시 모델별 JSON 레시피를 읽어 학습에 적용합니다.

분석 가정:
    - 독립·동일 분포 자료의 무작위 학습/테스트 분할을 사용합니다.
    - 입력은 결측값 없는 수치형 특성 및 단일 타깃입니다.
    - 테스트 자료를 모델 선택·스케일러 적합에 사용하지 않습니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.0.0    동일 명세의 Profile·Output DTO·실행 함수 작성
    2026-10-11  김재현(Finn)     v1.1.0    모델별 JSON 레시피 로딩과 공개 실행기 연결
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.1.0"
__status__ = "Development"


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.output import OutputDTO
from src.Recipes import load_profile, resolve_profile
from src.MachineLearning.common import (
    SupervisedOutputDTO,
    run_supervised,
    validate_supervised_inputs,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class LightGBMProfile:
    """LightGBM 회귀·분류의 입력 조건.

    Attributes:
        task (str): 분석 목적. 기본 classification. classification/regression 지원.
        test_size (float): 보류 테스트 비율. 기본 0.3, 0 < 값 < 1.
        random_state (int): 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수.
        scaling (str): 스케일링 방법. none/standard/minmax; 기본 none. 학습 폴드 안에서만 적합.
        stratify (bool): 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음.
        param_grid (Optional[Dict[str, Any]]): 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용.
        cv (int): CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요.
        scoring (Optional[str]): sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error.
        n_jobs (int): CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고.
        n_estimators (int): 부스팅 단계 수. 기본 100, 1 이상 정수.
        num_leaves (int): 트리별 최대 리프 수. 기본 31, 2 이상 정수.
        learning_rate (float): 학습률. 기본 0.1, 유한한 양수.
        max_depth (int): 최대 깊이. 기본 -1은 제한 없음, 이외에는 양수.
        min_child_samples (int): 리프 최소 관측 수. 기본 20, 1 이상 정수.
        subsample (float): 행 표본 비율. 기본 1.0, (0,1].
        subsample_freq (int): 행 샘플링 주기. 기본 0은 비활성; subsample<1이면 1 이상 필요.
        colsample_bytree (float): 특성 표본 비율. 기본 1.0, (0,1].
    """

    task: str = "classification"
    test_size: float = 0.3
    random_state: int = 1234
    scaling: str = "none"
    stratify: bool = True
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 3
    scoring: Optional[str] = None
    n_jobs: int = 1
    n_estimators: int = 100
    num_leaves: int = 31
    learning_rate: float = 0.1
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 1.0
    subsample_freq: int = 0
    colsample_bytree: float = 1.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class LightGBMOutputDTO(OutputDTO):
    """LightGBM 회귀·분류의 학습·평가 결과.

    Attributes:
        model_name (str): 적합된 추정기의 클래스 이름.
        task (str): 실행한 분석 목적. classification 또는 regression.
        model (Any): 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외.
        train_metrics (Dict[str, float]): 학습 지표. 일반화 성능을 뜻하지 않음. 분류는 accuracy·weighted precision/recall/F1, 회귀는 MAE/MSE/RMSE/R².
        test_metrics (Dict[str, float]): 보류 지표. 비율 지표는 0~1, MAE/RMSE는 y의 단위. 이진 확률 모델에 한해 ROC-AUC 추가.
        y_test (np.ndarray): 테스트 정답 (n_test,). test_indices와 동일한 행 순서.
        predictions (np.ndarray): 테스트 예측 (n_test,). y_test와 동일한 행 순서.
        train_indices (np.ndarray): 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치.
        test_indices (np.ndarray): 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음.
        feature_names (List[str]): 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름.
        classes (Optional[np.ndarray]): 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1].
        probabilities (Optional[np.ndarray]): 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None.
        confusion (Optional[np.ndarray]): classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None.
        best_params (Dict[str, Any]): CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전.
        best_cv_score (Optional[float]): 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음.
        cv_results (Optional[Dict[str, Any]]): GridSearchCV 전체 결과. 탐색하지 않았으면 None.
        feature_importances (Optional[np.ndarray]): 적합 모델의 입력 열 순서 특성 중요도 (n_features,). 원 라이브러리 값으로 모델 간 절대값 비교를 하지 않음.
    """

    model_name: str
    task: str
    model: Any
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    y_test: np.ndarray
    predictions: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: List[str]
    classes: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confusion: Optional[np.ndarray] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_cv_score: Optional[float] = None
    cv_results: Optional[Dict[str, Any]] = None
    feature_importances: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _require_positive_integer(value: Any, name: str) -> None:
    """양의 정수 설정을 검증합니다.

    Args:
        value: 검증할 값. bool은 허용하지 않음.
        name: 오류 메시지에 사용할 설정 이름.
    Returns:
        None: 유효하면 반환.
    Raises:
        ValueError: 1 미만이거나 정수가 아닌 값.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + "는 1 이상 정수여야 합니다.")


def _require_number(value: Any, name: str, positive: bool = False) -> None:
    """유한한 0 이상 또는 양수 설정을 검증합니다.

    Args:
        value: int/float 설정. bool은 허용하지 않음.
        name: 오류 메시지에 사용할 설정 이름.
        positive: True이면 0도 거부. 기본 False.
    Returns:
        None: 유효하면 반환.
    Raises:
        ValueError: 자료형·유한성·범위를 만족하지 않는 값.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(name + "는 유한한 수여야 합니다.")
    if value < 0 or (positive and value == 0):
        raise ValueError(name + "의 범위를 확인하세요.")


def _validate_inputs(X: Any, y: Any, profile: LightGBMProfile) -> None:
    """공통 입력과 모델별 설정을 검증합니다.

    Args:
        X: 최소 4행의 수치형 2차원 특성.
        y: X와 길이가 같은 단일 타깃.
        profile: LightGBMProfile 분석 조건.
    Returns:
        None: 검증 성공 시 반환.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·공통 조건·모델별 설정 오류.
    """
    if not isinstance(profile, LightGBMProfile):
        raise TypeError("LightGBMProfile이 필요합니다.")
    _, target = validate_supervised_inputs(X, y, profile)
    for name in ("n_estimators", "num_leaves", "min_child_samples"):
        _require_positive_integer(getattr(profile, name), name)
    if profile.num_leaves < 2:
        raise ValueError("num_leaves는 2 이상이어야 합니다.")
    if isinstance(profile.max_depth, bool) or not isinstance(profile.max_depth, int):
        raise ValueError("max_depth는 -1 또는 양수 정수여야 합니다.")
    if profile.max_depth != -1:
        _require_positive_integer(profile.max_depth, "max_depth")
    _require_number(profile.learning_rate, "learning_rate", positive=True)
    for name in ("subsample", "colsample_bytree"):
        value = getattr(profile, name)
        _require_number(value, name, positive=True)
        if value > 1:
            raise ValueError(name + "는 1 이하여야 합니다.")
    if (
        isinstance(profile.subsample_freq, bool)
        or not isinstance(profile.subsample_freq, int)
        or profile.subsample_freq < 0
    ):
        raise ValueError("subsample_freq는 0 이상 정수여야 합니다.")
    if profile.subsample < 1 and profile.subsample_freq == 0:
        raise ValueError("subsample<1일 때 subsample_freq를 1 이상 지정하세요.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _create_model(profile: LightGBMProfile) -> Any:
    """분류/회귀 목적에 맞는 미적합 모델을 생성합니다.

    Args:
        profile: 검증된 LightGBMProfile.
    Returns:
        Any: sklearn 호환 미적합 추정기.
    Raises:
        ImportError: 필요한 모델 라이브러리 미설치.
        ValueError: 라이브러리가 설정을 거부한 경우.
    """
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm 패키지를 설치하세요.") from exc
    estimator = LGBMClassifier if profile.task == "classification" else LGBMRegressor
    return estimator(
        n_estimators=profile.n_estimators,
        num_leaves=profile.num_leaves,
        learning_rate=profile.learning_rate,
        max_depth=profile.max_depth,
        min_child_samples=profile.min_child_samples,
        subsample=profile.subsample,
        subsample_freq=profile.subsample_freq,
        colsample_bytree=profile.colsample_bytree,
        random_state=profile.random_state,
        n_jobs=profile.n_jobs,
        verbosity=-1,
    )


def _fit_model(
    X: Any, y: Any, profile: LightGBMProfile, estimator: Any
) -> SupervisedOutputDTO:
    """학습/보류 분할, Pipeline 적합 및 평가를 수행합니다.

    Args:
        X: 검증된 원 입력. DataFrame 열 이름을 보존.
        y: 검증된 단일 타깃.
        profile: 분할·전처리·CV 및 모델 조건.
        estimator: _create_model이 생성한 미적합 추정기.
    Returns:
        SupervisedOutputDTO: 모델·지표·예측·분할 위치·CV 결과.
    Raises:
        ValueError: 표본/클래스 부족, 잘못된 CV 인자 또는 학습 실패.
    Notes:
        스케일러와 CV는 학습 자료에서만 적합하며 보류 자료는 평가에만 사용합니다.
    """
    return run_supervised(X, y, estimator, profile)


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_lightgbm(
    X: Any,
    y: Any,
    profile: Optional[LightGBMProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    task: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> LightGBMOutputDTO:
    """LightGBM 회귀·분류를 검증·학습하고 전용 DTO로 반환합니다.

    Args:
        X: 유한한 수치형 (n_samples, n_features) 배열 또는 DataFrame.
        y: 길이 n_samples의 1차원 타깃. 회귀는 수치, 분류는 이산 라벨.
        profile: LightGBMProfile. 기본 None이면 모델별 JSON 레시피에서 구성.
        recipe_path: 사용자 JSON 경로. None이면 src/Recipes의 모델별 기본 파일.
        task: Profile이 없을 때 분류/회귀 레시피 선택. None이면 모델 기본 목적.
        overrides: Profile이 없을 때 레시피 위에 적용할 명시적 설정 사전.
    Returns:
        LightGBMOutputDTO: 적합 Pipeline·평가·예측 및 모델별 결과.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·분할·CV·학습 오류.
        ImportError: 선택 모델 라이브러리 미설치.
        FileNotFoundError/OSError: 레시피 파일 누락 또는 읽기 실패.
    Notes:
        LightGBM 2.2.3과 현대 API의 공통 인자를 사용합니다. subsample만 줄이면 행 샘플링이 활성화되지 않으므로 subsample_freq도 지정합니다. n_jobs는 모델 스레드와 CV에 적용됩니다.
        입력 열 순서는 새 데이터 예측에서도 유지합니다. 자동 결측 대체·범주 인코딩은 하지 않습니다.
        기본 to_dict()/to_json()은 모델 실행 객체를 제외하고 분석 데이터를 반환합니다.
    """
    profile = resolve_profile(
        "lightgbm", profile, recipe_path=recipe_path, task=task, overrides=overrides
    )
    _validate_inputs(X, y, profile)
    estimator = _create_model(profile)
    result = _fit_model(X, y, profile, estimator)
    fitted = result.model.named_steps["model"]
    return LightGBMOutputDTO(
        **result.to_dict(include_runtime=True),
        feature_importances=np.asarray(fitted.feature_importances_, dtype=float).copy(),
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    profile = load_profile("lightgbm")
    result = run_lightgbm(X, y, profile)

    print("Test metrics:", result.test_metrics)
    print("Predictions:", result.predictions[:5])
    print("New predictions:", result.model.predict(X[:3]))
    print("Result keys:", result.to_dict().keys())
    print("JSON preview:", result.to_json(indent=2)[:300])
