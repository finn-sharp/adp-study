"""
================================================================================
xgboost.py - XGBoost 회귀·분류
================================================================================
설명:
    - XGBoost 회귀·분류의 학습·평가·학습 구간 교차검증을 제공합니다.
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
import inspect
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

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
class XGBoostProfile:
    """XGBoost 회귀·분류의 입력 조건.

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
        max_depth (int): 트리 최대 깊이. 기본 3, 1 이상 정수.
        learning_rate (float): 학습률. 기본 0.1, 유한한 양수.
        subsample (float): 트리별 행 표본 비율. 기본 1.0, (0,1].
        colsample_bytree (float): 트리별 특성 표본 비율. 기본 1.0, (0,1].
        min_child_weight (float): 리프 최소 Hessian 합. 기본 1.0, 0 이상.
        reg_alpha (float): L1 규제 강도. 기본 0.0, 0 이상.
        reg_lambda (float): L2 규제 강도. 기본 1.0, 0 이상.
        gamma (float): 트리 분할 최소 손실 감소. 기본 0.0, 유한한 0 이상 값.
        objective (Optional[str]): 목적함수. 기본 None이면 task와 클래스 수에 맞춰 선택. 분류는 binary:logistic 또는 multi:softprob, 회귀는 reg:squarederror.
        eval_metric (Optional[str]): 조기 종료 평가 지표. 기본 None이면 회귀 rmse, 이진 logloss, 다중 mlogloss.
        early_stopping_rounds (Optional[int]): 개선 없는 라운드 허용 수. 기본 None이면 조기 종료 없음. 지정 시 양의 정수.
        validation_fraction (float): 조기 종료 시 현재 학습 폴드 내부 검증 비율. 기본 0.2, (0,1). 보류 테스트는 사용하지 않음.
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
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    objective: Optional[str] = None
    eval_metric: Optional[str] = None
    early_stopping_rounds: Optional[int] = None
    validation_fraction: float = 0.2


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class XGBoostOutputDTO(OutputDTO):
    """XGBoost 회귀·분류의 학습·평가 결과.

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
        best_iteration (Optional[int]): 조기 종료의 최적 라운드. 0부터 시작. 조기 종료를 사용하지 않으면 None.
        early_stopping_train_indices (Optional[np.ndarray]): 최종 모델의 내부 적합 행 위치. 원 입력 기준. 조기 종료 미사용 시 None.
        early_stopping_validation_indices (Optional[np.ndarray]): 최종 모델의 내부 검증 행 위치. 원 입력 기준. 보류 테스트와 겹치지 않음. 조기 종료 미사용 시 None.
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
    best_iteration: Optional[int] = None
    early_stopping_train_indices: Optional[np.ndarray] = None
    early_stopping_validation_indices: Optional[np.ndarray] = None


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


def _validate_inputs(X: Any, y: Any, profile: XGBoostProfile) -> None:
    """공통 입력과 모델별 설정을 검증합니다.

    Args:
        X: 최소 4행의 수치형 2차원 특성.
        y: X와 길이가 같은 단일 타깃.
        profile: XGBoostProfile 분석 조건.
    Returns:
        None: 검증 성공 시 반환.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·공통 조건·모델별 설정 오류.
    """
    if not isinstance(profile, XGBoostProfile):
        raise TypeError("XGBoostProfile이 필요합니다.")
    _, target = validate_supervised_inputs(X, y, profile)
    for name in ("n_estimators", "max_depth"):
        _require_positive_integer(getattr(profile, name), name)
    _require_number(profile.learning_rate, "learning_rate", positive=True)
    for name in ("subsample", "colsample_bytree"):
        value = getattr(profile, name)
        _require_number(value, name, positive=True)
        if value > 1:
            raise ValueError(name + "는 1 이하여야 합니다.")
    for name in ("min_child_weight", "reg_alpha", "reg_lambda", "gamma"):
        _require_number(getattr(profile, name), name)
    if profile.early_stopping_rounds is not None:
        _require_positive_integer(
            profile.early_stopping_rounds, "early_stopping_rounds"
        )
    if (
        isinstance(profile.validation_fraction, bool)
        or not isinstance(profile.validation_fraction, (int, float))
        or not 0 < profile.validation_fraction < 1
    ):
        raise ValueError("validation_fraction은 0과 1 사이여야 합니다.")
    if profile.eval_metric is not None and (
        not isinstance(profile.eval_metric, str) or not profile.eval_metric.strip()
    ):
        raise ValueError("eval_metric은 비어 있지 않은 문자열 또는 None이어야 합니다.")
    expected_objective = (
        "reg:squarederror"
        if profile.task == "regression"
        else ("binary:logistic" if len(np.unique(target)) == 2 else "multi:softprob")
    )
    if profile.objective is not None and profile.objective != expected_objective:
        raise ValueError(
            "objective는 현재 task·클래스 수에 맞는 "
            + expected_objective
            + "여야 합니다."
        )
    if profile.task == "classification":
        labels = np.unique(target)
        if not np.array_equal(labels, np.arange(len(labels))):
            raise ValueError("XGBoost 분류 라벨은 0부터 시작하는 연속 정수여야 합니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


class _XGBoostEstimator(BaseEstimator):
    """각 학습 폴드 안에서 조기 종료 자료를 분리하는 sklearn 추정기."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        random_state=42,
        n_jobs=1,
        objective=None,
        eval_metric=None,
        early_stopping_rounds=None,
        validation_fraction=0.2,
        task="classification",
    ):
        """Profile의 모델 설정을 보존합니다.

        Args:
            n_estimators: 최대 트리 수.
            max_depth: 트리 최대 깊이.
            learning_rate: 학습률.
            subsample: 행 표본 비율.
            colsample_bytree: 열 표본 비율.
            min_child_weight: 자식 노드 최소 가중치.
            reg_alpha: L1 규제.
            reg_lambda: L2 규제.
            gamma: 분할 최소 손실 감소.
            random_state: 재현용 시드.
            n_jobs: 학습 스레드 수.
            objective: 목적 함수 또는 자동 선택 None.
            eval_metric: 평가 지표 또는 자동 선택 None.
            early_stopping_rounds: 조기 종료 대기 횟수 또는 비활성 None.
            validation_fraction: 현재 학습 자료 중 조기 종료 평가 비율.
            task: classification 또는 regression.
        Returns:
            None: 설정을 저장.
        Raises:
            None: 적합 시 설정 검증.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.objective = objective
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.task = task

    def fit(self, X, y):
        """현재 학습 자료만 사용해 원래 XGBoost 추정기를 적합합니다.

        Args:
            X: 현재 학습 폴드의 전처리된 수치 특성.
            y: 현재 학습 폴드의 타깃.
        Returns:
            _XGBoostEstimator: 적합된 self.
        Raises:
            ValueError: 잘못된 설정 또는 조기 종료 분할에 표본/클래스 부족.
            ImportError: XGBoost 미설치.
        """
        from xgboost import XGBClassifier, XGBRegressor

        _validate_inputs(X, y, XGBoostProfile(**self.get_params(deep=False)))
        X, y = np.asarray(X), np.asarray(y)
        self.training_indices_ = np.arange(len(y))
        self.validation_indices_ = np.array([], dtype=int)
        if self.early_stopping_rounds is not None:
            self.training_indices_, self.validation_indices_ = train_test_split(
                self.training_indices_,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y if self.task == "classification" else None,
            )
            if min(len(self.training_indices_), len(self.validation_indices_)) < 2:
                raise ValueError(
                    "조기 종료 학습·평가 자료는 각각 최소 2행이어야 합니다."
                )
            if self.task == "classification" and not np.array_equal(
                np.unique(y[self.training_indices_]), np.unique(y)
            ):
                raise ValueError("조기 종료 학습 자료에 모든 클래스가 필요합니다.")
        native_class = XGBClassifier if self.task == "classification" else XGBRegressor
        kwargs = self.get_params(deep=False).copy()
        for name in ("task", "validation_fraction", "early_stopping_rounds"):
            kwargs.pop(name)
        kwargs["objective"] = self.objective or (
            "reg:squarederror"
            if self.task == "regression"
            else "binary:logistic" if len(np.unique(y)) == 2 else "multi:softprob"
        )
        metric = self.eval_metric
        if metric is None and self.early_stopping_rounds is not None:
            metric = (
                "rmse"
                if self.task == "regression"
                else ("logloss" if len(np.unique(y)) == 2 else "mlogloss")
            )
        kwargs.pop("eval_metric")
        kwargs["verbosity"] = 0
        fit_parameters = inspect.signature(native_class.fit).parameters
        fit_kwargs = {}
        if metric is not None:
            if "eval_metric" in fit_parameters:
                fit_kwargs["eval_metric"] = metric
            else:
                kwargs["eval_metric"] = metric
        if self.early_stopping_rounds is not None:
            if "early_stopping_rounds" in fit_parameters:
                fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            else:
                kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            fit_kwargs.update(
                eval_set=[(X[self.validation_indices_], y[self.validation_indices_])],
                verbose=False,
            )
        self.native_model_ = native_class(**kwargs)
        self.native_model_.fit(
            X[self.training_indices_], y[self.training_indices_], **fit_kwargs
        )
        self.n_features_in_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.best_iteration_ = (
            int(self.native_model_.best_iteration)
            if self.early_stopping_rounds is not None
            else None
        )
        return self

    def _prediction_kwargs(self, method):
        """구버전과 현대 버전 모두 최적 조기 종료 트리 수를 선택합니다.

        Args:
            method: 원래 모델의 예측 메서드.
        Returns:
            dict: 버전에 맞는 예측 인자.
        Raises:
            NotFittedError: 아직 적합되지 않은 경우.
        """
        check_is_fitted(self, "native_model_")
        parameters = inspect.signature(method).parameters
        if self.early_stopping_rounds is not None:
            if "iteration_range" in parameters:
                return {"iteration_range": (0, self.best_iteration_ + 1)}
            if "ntree_limit" in parameters:
                return {"ntree_limit": self.native_model_.best_ntree_limit}
        return {}

    def predict(self, X):
        """최적 트리까지 사용해 새 자료를 예측합니다.

        Args:
            X: 학습 때와 동일한 열 순서의 수치 특성.
        Returns:
            np.ndarray: 예측 라벨 또는 회귀값.
        Raises:
            NotFittedError: 미적합 모델.
            ValueError: 잘못된 예측 특성.
        """
        check_is_fitted(self, "native_model_")
        method = self.native_model_.predict
        return method(X, **self._prediction_kwargs(method))

    @property
    def feature_importances_(self):
        """원래 모델의 특성 중요도를 반환합니다.

        Args:
            None: 추가 인자 없음.
        Returns:
            np.ndarray: 입력 열 순서의 중요도.
        Raises:
            NotFittedError: 미적합 모델.
        """
        check_is_fitted(self, "native_model_")
        return self.native_model_.feature_importances_


class _XGBoostClassifier(ClassifierMixin, _XGBoostEstimator):
    """sklearn 분류 태그와 확률 예측을 제공하는 어댑터."""

    def predict_proba(self, X):
        """최적 트리까지 사용해 클래스별 확률을 반환합니다.

        Args:
            X: 학습 때와 동일한 열 순서의 수치 특성.
        Returns:
            np.ndarray: 행별 클래스 확률.
        Raises:
            NotFittedError: 미적합 모델.
            ValueError: 잘못된 예측 특성.
        """
        check_is_fitted(self, "native_model_")
        method = self.native_model_.predict_proba
        return method(X, **self._prediction_kwargs(method))


class _XGBoostRegressor(RegressorMixin, _XGBoostEstimator):
    """sklearn 회귀 태그를 제공하는 어댑터."""


def _create_model(profile: XGBoostProfile) -> Any:
    """분류/회귀 목적에 맞는 미적합 모델을 생성합니다.

    Args:
        profile: 검증된 XGBoostProfile.
    Returns:
        Any: sklearn 호환 미적합 추정기.
    Raises:
        ImportError: 필요한 모델 라이브러리 미설치.
        ValueError: 라이브러리가 설정을 거부한 경우.
    """
    adapter = (
        _XGBoostClassifier if profile.task == "classification" else _XGBoostRegressor
    )
    return adapter(
        n_estimators=profile.n_estimators,
        max_depth=profile.max_depth,
        learning_rate=profile.learning_rate,
        subsample=profile.subsample,
        colsample_bytree=profile.colsample_bytree,
        min_child_weight=profile.min_child_weight,
        reg_alpha=profile.reg_alpha,
        reg_lambda=profile.reg_lambda,
        gamma=profile.gamma,
        random_state=profile.random_state,
        n_jobs=profile.n_jobs,
        objective=profile.objective,
        eval_metric=profile.eval_metric,
        early_stopping_rounds=profile.early_stopping_rounds,
        validation_fraction=profile.validation_fraction,
        task=profile.task,
    )


def _fit_model(
    X: Any, y: Any, profile: XGBoostProfile, estimator: Any
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


def run_xgboost(
    X: Any,
    y: Any,
    profile: Optional[XGBoostProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    task: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> XGBoostOutputDTO:
    """XGBoost 회귀·분류를 검증·학습하고 전용 DTO로 반환합니다.

    Args:
        X: 유한한 수치형 (n_samples, n_features) 배열 또는 DataFrame.
        y: 길이 n_samples의 1차원 타깃. 회귀는 수치, 분류는 이산 라벨.
        profile: XGBoostProfile. 기본 None이면 모델별 JSON 레시피에서 구성.
        recipe_path: 사용자 JSON 경로. None이면 src/Recipes의 모델별 기본 파일.
        task: Profile이 없을 때 분류/회귀 레시피 선택. None이면 모델 기본 목적.
        overrides: Profile이 없을 때 레시피 위에 적용할 명시적 설정 사전.
    Returns:
        XGBoostOutputDTO: 적합 Pipeline·평가·예측 및 모델별 결과.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·분할·CV·학습 오류.
        ImportError: 선택 모델 라이브러리 미설치.
        FileNotFoundError/OSError: 레시피 파일 누락 또는 읽기 실패.
    Notes:
        XGBoost 0.90과 현대 API의 공통 인자를 사용합니다. 분류 라벨은 0부터 연속 정수로 입력합니다. n_jobs는 모델 스레드와 CV에 모두 적용됩니다.
        입력 열 순서는 새 데이터 예측에서도 유지합니다. 자동 결측 대체·범주 인코딩은 하지 않습니다.
        기본 to_dict()/to_json()은 모델 실행 객체를 제외하고 분석 데이터를 반환합니다.
    """
    profile = resolve_profile(
        "xgboost", profile, recipe_path=recipe_path, task=task, overrides=overrides
    )
    _validate_inputs(X, y, profile)
    estimator = _create_model(profile)
    result = _fit_model(X, y, profile, estimator)
    fitted = result.model.named_steps["model"]
    return XGBoostOutputDTO(
        **result.to_dict(include_runtime=True),
        feature_importances=np.asarray(fitted.feature_importances_, dtype=float).copy(),
        best_iteration=(
            fitted.best_iteration_ if fitted.early_stopping_rounds is not None else None
        ),
        early_stopping_train_indices=(
            result.train_indices[fitted.training_indices_]
            if fitted.early_stopping_rounds is not None
            else None
        ),
        early_stopping_validation_indices=(
            result.train_indices[fitted.validation_indices_]
            if fitted.early_stopping_rounds is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    profile = load_profile("xgboost")
    result = run_xgboost(X, y, profile)

    print("Test metrics:", result.test_metrics)
    print("Predictions:", result.predictions[:5])
    print("New predictions:", result.model.predict(X[:3]))
    print("Result keys:", result.to_dict().keys())
    print("JSON preview:", result.to_json(indent=2)[:300])
