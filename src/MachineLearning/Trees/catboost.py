"""
================================================================================
catboost.py - CatBoost 회귀·분류
================================================================================
설명:
    - CatBoost 회귀·분류의 학습·평가·학습 구간 교차검증을 제공합니다.
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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class CatBoostProfile:
    """CatBoost 회귀·분류의 입력 조건.

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
        iterations (int): 부스팅 반복 수. 기본 100, 1 이상 정수.
        depth (int): 대칭 트리 깊이. 기본 6, CPU 모듈에서 1~16.
        learning_rate (float): 학습률. 기본 0.1, 유한한 양수.
        l2_leaf_reg (float): 리프 L2 규제 강도. 기본 3.0, 유한한 0 이상 값.
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
    iterations: int = 100
    depth: int = 6
    learning_rate: float = 0.1
    l2_leaf_reg: float = 3.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class CatBoostOutputDTO(OutputDTO):
    """CatBoost 회귀·분류의 학습·평가 결과.

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


def _validate_inputs(X: Any, y: Any, profile: CatBoostProfile) -> None:
    """공통 입력과 모델별 설정을 검증합니다.

    Args:
        X: 최소 4행의 수치형 2차원 특성.
        y: X와 길이가 같은 단일 타깃.
        profile: CatBoostProfile 분석 조건.
    Returns:
        None: 검증 성공 시 반환.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·공통 조건·모델별 설정 오류.
    """
    if not isinstance(profile, CatBoostProfile):
        raise TypeError("CatBoostProfile이 필요합니다.")
    _, target = validate_supervised_inputs(X, y, profile)
    _require_positive_integer(profile.iterations, "iterations")
    _require_positive_integer(profile.depth, "depth")
    if profile.depth > 16:
        raise ValueError("depth는 16 이하여야 합니다.")
    _require_number(profile.learning_rate, "learning_rate", positive=True)
    _require_number(profile.l2_leaf_reg, "l2_leaf_reg")
    if profile.n_jobs < -1:
        raise ValueError("CatBoost n_jobs는 -1 또는 양수여야 합니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


class _CatBoostEstimator(BaseEstimator):
    """CatBoost를 sklearn의 clone·Pipeline·CV 계약에 연결합니다.

    Attributes:
        iterations, depth, learning_rate, l2_leaf_reg: CatBoostProfile과 동일.
        random_state, n_jobs, task: 학습 시드·스레드·분류/회귀 선택.
        native_model_: fit 후 원 CatBoost 모델. 호출 전에는 존재하지 않음.
        n_features_in_: fit 입력의 특성 수.
        classes_: 분류 모델 fit 후 원래 타깃 클래스 순서.
    """

    def __init__(
        self,
        iterations=100,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3.0,
        random_state=1234,
        n_jobs=1,
        task="classification",
    ):
        """clone이 재구성할 수 있도록 학습 설정을 그대로 저장합니다.

        Args:
            iterations: 부스팅 단계 수.
            depth: 트리 깊이.
            learning_rate: 학습률.
            l2_leaf_reg: L2 규제 강도.
            random_state: 난수 시드.
            n_jobs: CPU 스레드 수.
            task: classification 또는 regression.
        Returns:
            None: 생성자 반환값.
        Raises:
            None: 설정 검증은 공개 실행기와 fit에서 수행.
        """
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.task = task

    def fit(self, X: Any, y: Any) -> "_CatBoostEstimator":
        """주어진 학습 구간만으로 원 CatBoost 모델을 적합합니다.

        Args:
            X: 검증된 수치형 2차원 학습 특성.
            y: X와 같은 길이의 단일 타깃.
        Returns:
            _CatBoostEstimator: 원 모델을 보유한 self.
        Raises:
            ImportError: CatBoost 미설치.
            ValueError: 원 모델이 입력이나 학습 설정을 거부한 경우.
        """
        from catboost import CatBoostClassifier, CatBoostRegressor

        estimator = (
            CatBoostClassifier if self.task == "classification" else CatBoostRegressor
        )
        self.native_model_ = estimator(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            thread_count=self.n_jobs,
            verbose=False,
            allow_writing_files=False,
        ).fit(X, y)
        self.n_features_in_ = np.asarray(X).shape[1]
        if self.task == "classification":
            self.classes_ = np.asarray(self.native_model_.classes_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """학습과 같은 열 순서의 새 입력을 예측합니다.

        Args:
            X: 학습과 같은 열 수·열 순서의 특성.
        Returns:
            np.ndarray: 원 라벨 또는 회귀 예측 (n_samples,).
        Raises:
            NotFittedError: fit 전 호출.
            ValueError: 원 모델이 입력을 거부한 경우.
        """
        check_is_fitted(self, "native_model_")
        return np.asarray(self.native_model_.predict(X)).reshape(-1)

    @property
    def feature_importances_(self) -> np.ndarray:
        """적합 모델의 원래 입력 열 순서의 중요도를 반환합니다.

        Args:
            없음: 적합 모델 속성 접근.
        Returns:
            np.ndarray: 중요도 (n_features,). CatBoost 원 값이며 모듈에서 재정규화하지 않음.
        Raises:
            NotFittedError: fit 전 호출.
        """
        check_is_fitted(self, "native_model_")
        return np.asarray(self.native_model_.feature_importances_)


class _CatBoostClassifier(ClassifierMixin, _CatBoostEstimator):
    """분류 태그와 확률 예측을 제공하는 CatBoost 어댑터."""

    def predict_proba(self, X: Any) -> np.ndarray:
        """원 클래스 순서의 확률을 반환합니다.

        Args:
            X: 학습과 같은 열 순서의 특성.
        Returns:
            np.ndarray: classes_ 순서 확률 (n_samples, n_classes).
        Raises:
            NotFittedError: fit 전 호출.
            ValueError: 원 모델이 입력을 거부한 경우.
        """
        check_is_fitted(self, "native_model_")
        return np.asarray(self.native_model_.predict_proba(X))


class _CatBoostRegressor(RegressorMixin, _CatBoostEstimator):
    """회귀 태그를 제공하는 CatBoost 어댑터."""


def _create_model(profile: CatBoostProfile) -> Any:
    """분류/회귀 목적에 맞는 미적합 모델을 생성합니다.

    Args:
        profile: 검증된 CatBoostProfile.
    Returns:
        Any: sklearn 호환 미적합 추정기.
    Raises:
        ImportError: 필요한 모델 라이브러리 미설치.
        ValueError: 라이브러리가 설정을 거부한 경우.
    """
    try:
        import catboost  # 선택 의존성 확인. 학습은 어댑터에서 실행합니다.
    except ImportError as exc:
        raise ImportError("catboost 패키지를 설치하세요.") from exc
    estimator = (
        _CatBoostClassifier if profile.task == "classification" else _CatBoostRegressor
    )
    return estimator(
        iterations=profile.iterations,
        depth=profile.depth,
        learning_rate=profile.learning_rate,
        l2_leaf_reg=profile.l2_leaf_reg,
        random_state=profile.random_state,
        n_jobs=profile.n_jobs,
        task=profile.task,
    )


def _fit_model(
    X: Any, y: Any, profile: CatBoostProfile, estimator: Any
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


def run_catboost(
    X: Any,
    y: Any,
    profile: Optional[CatBoostProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    task: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CatBoostOutputDTO:
    """CatBoost 회귀·분류를 검증·학습하고 전용 DTO로 반환합니다.

    Args:
        X: 유한한 수치형 (n_samples, n_features) 배열 또는 DataFrame.
        y: 길이 n_samples의 1차원 타깃. 회귀는 수치, 분류는 이산 라벨.
        profile: CatBoostProfile. 기본 None이면 모델별 JSON 레시피에서 구성.
        recipe_path: 사용자 JSON 경로. None이면 src/Recipes의 모델별 기본 파일.
        task: Profile이 없을 때 분류/회귀 레시피 선택. None이면 모델 기본 목적.
        overrides: Profile이 없을 때 레시피 위에 적용할 명시적 설정 사전.
    Returns:
        CatBoostOutputDTO: 적합 Pipeline·평가·예측 및 모델별 결과.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·분할·CV·학습 오류.
        ImportError: 선택 모델 라이브러리 미설치.
        FileNotFoundError/OSError: 레시피 파일 누락 또는 읽기 실패.
    Notes:
        CPU와 수치형 입력을 지원합니다. 범주형 열은 호출 전에 인코딩합니다. sklearn 태그 어댑터를 사용하며 원 CatBoost 모델은 result.model.named_steps["model"].native_model_로 접근합니다. n_jobs는 thread_count와 CV에 적용됩니다. 학습 로그 파일을 생성하지 않습니다.
        입력 열 순서는 새 데이터 예측에서도 유지합니다. 자동 결측 대체·범주 인코딩은 하지 않습니다.
        기본 to_dict()/to_json()은 모델 실행 객체를 제외하고 분석 데이터를 반환합니다.
    """
    profile = resolve_profile(
        "catboost", profile, recipe_path=recipe_path, task=task, overrides=overrides
    )
    _validate_inputs(X, y, profile)
    estimator = _create_model(profile)
    result = _fit_model(X, y, profile, estimator)
    fitted = result.model.named_steps["model"]
    return CatBoostOutputDTO(
        **result.to_dict(include_runtime=True),
        feature_importances=np.asarray(fitted.feature_importances_, dtype=float).copy(),
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    profile = load_profile("catboost")
    result = run_catboost(X, y, profile)

    print("Test metrics:", result.test_metrics)
    print("Predictions:", result.predictions[:5])
    print("New predictions:", result.model.predict(X[:3]))
    print("Result keys:", result.to_dict().keys())
    print("JSON preview:", result.to_json(indent=2)[:300])
