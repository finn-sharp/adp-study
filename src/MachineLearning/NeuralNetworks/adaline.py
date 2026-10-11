"""
================================================================================
adaline.py - 아달린(ADALINE) 이진 분류
================================================================================
설명:
    - 아달린(ADALINE) 이진 분류의 학습·평가·학습 구간 교차검증을 제공합니다.
    - 분석 조건은 Profile, 분석 결과는 Output DTO로 정의합니다.

분석 가정:
    - 독립·동일 분포 자료의 무작위 학습/테스트 분할을 사용합니다.
    - 입력은 결측값 없는 수치형 특성 및 단일 타깃입니다.
    - 테스트 자료를 모델 선택·스케일러 적합에 사용하지 않습니다.
    - 이진 분류만 지원하며 선형 출력으로 제곱 손실을 학습한 뒤 0에서 분류합니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.0.0    동일 명세의 Profile·Output DTO·실행 함수 작성
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.output import OutputDTO
from src.MachineLearning.common import (
    SupervisedOutputDTO, run_supervised, validate_supervised_inputs,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class AdalineProfile:
    """아달린(ADALINE) 이진 분류의 입력 조건.

    Attributes:
        task (str): 분석 목적. 기본 classification. 이진 classification만 지원.
        test_size (float): 보류 테스트 비율. 기본 0.3, 0 < 값 < 1.
        random_state (int): 분할·CV 난수 시드. 모델은 영벡터로 초기화. 기본 1234, bool 이외 정수.
        scaling (str): 스케일링 방법. none/standard/minmax; 기본 standard. 학습 폴드 안에서만 적합.
        stratify (bool): 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음.
        param_grid (Optional[Dict[str, Any]]): 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용.
        cv (int): CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요.
        scoring (Optional[str]): sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error.
        n_jobs (int): CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고.
        max_iter (int): 최대 배치 학습 epoch 수. 기본 1000, 1 이상 정수.
        eta0 (float): 평균 제곱 손실의 상수 학습률. 기본 0.01, 유한한 양수.
        tol (Optional[float]): 연속 epoch의 손실 감소가 이 값 이하이면 종료. 기본 0.000001, 양수 또는 None.
        fit_intercept (bool): 절편 학습 여부. 기본 True.
    """

    task: str = "classification"
    test_size: float = 0.3
    random_state: int = 1234
    scaling: str = "standard"
    stratify: bool = True
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 3
    scoring: Optional[str] = None
    n_jobs: int = 1
    max_iter: int = 1000
    eta0: float = 0.01
    tol: Optional[float] = 0.000001
    fit_intercept: bool = True


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class AdalineOutputDTO(OutputDTO):
    """아달린(ADALINE) 이진 분류의 학습·평가 결과.

    Attributes:
        model_name (str): 적합된 추정기의 클래스 이름.
        task (str): 실행한 분석 목적. classification.
        model (Any): 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외.
        train_metrics (Dict[str, float]): 학습 accuracy·weighted precision/recall/F1. 일반화 성능을 뜻하지 않음.
        test_metrics (Dict[str, float]): 보류 accuracy·weighted precision/recall/F1 (0~1). 확률 모델이 아니므로 공통 실행기는 ROC-AUC를 추가하지 않음.
        y_test (np.ndarray): 테스트 정답 (n_test,). test_indices와 동일한 행 순서.
        predictions (np.ndarray): 테스트 예측 (n_test,). y_test와 동일한 행 순서.
        train_indices (np.ndarray): 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치.
        test_indices (np.ndarray): 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음.
        feature_names (List[str]): 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름.
        classes (Optional[np.ndarray]): 원래 클래스 정렬 순서 (2,). 학습 중 classes[0]→-1, classes[1]→+1.
        probabilities (Optional[np.ndarray]): Adaline은 확률을 지원하지 않으므로 None. 선형 출력은 activation_scores로 반환.
        confusion (Optional[np.ndarray]): classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None.
        best_params (Dict[str, Any]): CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전.
        best_cv_score (Optional[float]): 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음.
        cv_results (Optional[Dict[str, Any]]): GridSearchCV 전체 결과. 탐색하지 않았으면 None.
        coefficients (Optional[np.ndarray]): 선형 가중치 (1,n_features). 스케일러 변환 좌표계.
        intercepts (Optional[np.ndarray]): 절편 (1,). fit_intercept=False이면 0.
        activation_scores (Optional[np.ndarray]): 보류 자료의 연속 선형 출력 (n_test,). 확률이 아니며 0 이상이면 classes[1].
        loss_curve (Optional[np.ndarray]): 각 epoch 갱신 후 학습 손실 0.5*mean((z-target)²). target은 -1/+1.
        n_iter (Optional[int]): 실제 학습 epoch 수. loss_curve 길이와 동일.
        converged (Optional[bool]): tol 기준 종료 여부. None 설정 또는 기준 미충족 시 False.
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
    coefficients: Optional[np.ndarray] = None
    intercepts: Optional[np.ndarray] = None
    activation_scores: Optional[np.ndarray] = None
    loss_curve: Optional[np.ndarray] = None
    n_iter: Optional[int] = None
    converged: Optional[bool] = None


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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(name + "는 유한한 수여야 합니다.")
    if value < 0 or (positive and value == 0):
        raise ValueError(name + "의 범위를 확인하세요.")


def _validate_inputs(X: Any, y: Any, profile: AdalineProfile) -> None:
    """공통 입력과 모델별 설정을 검증합니다.

    Args:
        X: 최소 4행의 수치형 2차원 특성.
        y: X와 길이가 같은 단일 타깃.
        profile: AdalineProfile 분석 조건.
    Returns:
        None: 검증 성공 시 반환.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·공통 조건·모델별 설정 오류.
    """
    if not isinstance(profile, AdalineProfile):
        raise TypeError("AdalineProfile이 필요합니다.")
    _, target = validate_supervised_inputs(X, y, profile)
    if profile.task != "classification" or len(np.unique(target)) != 2:
        raise ValueError("Adaline은 정확히 두 클래스의 classification만 지원합니다.")
    _validate_adaline_parameters(profile)


def _validate_adaline_parameters(settings: Any) -> None:
    """Profile 또는 추정기의 배치 학습 설정을 검증합니다.

    Args:
        settings: max_iter, eta0, tol, fit_intercept 속성을 가진 객체.
    Returns:
        None: 검증 성공 시 반환.
    Raises:
        ValueError: 잘못된 반복 수·학습률·수렴 허용오차·절편 설정.
    """
    _require_positive_integer(settings.max_iter, "max_iter")
    _require_number(settings.eta0, "eta0", positive=True)
    if settings.tol is not None:
        _require_number(settings.tol, "tol", positive=True)
    if not isinstance(settings.fit_intercept, bool):
        raise ValueError("fit_intercept는 bool이어야 합니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

class AdalineClassifier(ClassifierMixin, BaseEstimator):
    """영벡터 초기화 및 배치 평균 경사하강법을 사용하는 ADALINE 추정기.

    Attributes:
        max_iter (int): 최대 epoch 수.
        eta0 (float): 상수 학습률.
        tol (Optional[float]): 손실 감소의 종료 허용오차. None이면 조기 종료하지 않음.
        fit_intercept (bool): 절편 학습 여부.
        classes_ (np.ndarray): fit 후 두 원래 클래스의 정렬 순서.
        n_features_in_ (int): fit 입력의 특성 수.
        coef_ (np.ndarray): 적합 가중치 (1,n_features).
        intercept_ (np.ndarray): 적합 절편 (1,).
        loss_curve_ (np.ndarray): 갱신 후 epoch별 0.5*mean squared error.
        n_iter_ (int): 실제 epoch 수.
        converged_ (bool): tol 기준 종료 여부.
    """

    def __init__(self, max_iter=1000, eta0=0.01, tol=0.000001, fit_intercept=True):
        """sklearn clone 및 GridSearchCV가 복제할 설정을 저장합니다.

        Args:
            max_iter: 최대 배치 학습 반복 수.
            eta0: 상수 학습률.
            tol: 손실 감소의 종료 허용오차 또는 None.
            fit_intercept: 절편 학습 여부.
        Returns:
            None: 생성자 반환값.
        Raises:
            None: 검증은 fit에서 수행.
        """
        self.max_iter = max_iter
        self.eta0 = eta0
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X: Any, y: Any) -> "AdalineClassifier":
        """두 라벨을 -1/+1로 변환하여 연속 출력의 제곱 손실을 학습합니다.

        Args:
            X: 유한한 수치형 2차원 학습 특성. 외부 Pipeline에서 스케일링 권장.
            y: X와 길이가 같은 두 클래스의 이산 라벨.
        Returns:
            AdalineClassifier: 적합된 self.
        Raises:
            ValueError: 설정·입력·클래스 수 오류 또는 학습 발산.
        Notes:
            J=0.5*mean((Xw+b-t)²). w += eta0*X.T@(t-z)/n,
            b += eta0*mean(t-z). 클래스 임계값은 손실 계산에 사용하지 않습니다.
            매번 영벡터로 초기화하며 warm start는 지원하지 않습니다.
            tol 설정 시 미수렴 반복 한도 도달은 ConvergenceWarning으로 알립니다.
        """
        _validate_adaline_parameters(self)
        X, y = check_X_y(X, y, dtype=float)
        if type_of_target(y) != "binary" or len(np.unique(y)) != 2:
            raise ValueError("Adaline에는 정확히 두 클래스의 이산 라벨이 필요합니다.")
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        target = np.where(y == self.classes_[1], 1.0, -1.0)
        self.coef_ = np.zeros((1, self.n_features_in_), dtype=float)
        self.intercept_ = np.zeros(1, dtype=float)
        history = []
        previous_loss = 0.5
        self.converged_ = False
        try:
            with np.errstate(over="raise", invalid="raise"):
                for _ in range(self.max_iter):
                    error = target - (X.dot(self.coef_[0]) + self.intercept_[0])
                    self.coef_[0] += self.eta0 * X.T.dot(error) / len(X)
                    if self.fit_intercept:
                        self.intercept_[0] += self.eta0 * error.mean()
                    residual = target - (X.dot(self.coef_[0]) + self.intercept_[0])
                    loss = float(0.5 * np.mean(residual ** 2))
                    if not np.isfinite(loss):
                        raise FloatingPointError("non-finite loss")
                    history.append(loss)
                    improvement = previous_loss - loss
                    if self.tol is not None and 0 <= improvement <= self.tol:
                        self.converged_ = True
                        break
                    previous_loss = loss
        except FloatingPointError as exc:
            raise ValueError("Adaline 학습이 발산했습니다. 특성을 스케일링하고 eta0를 낮추세요.") from exc
        self.loss_curve_ = np.asarray(history)
        self.n_iter_ = len(history)
        if self.tol is not None and not self.converged_:
            warnings.warn("Adaline이 max_iter 안에 수렴하지 않았습니다. loss_curve를 확인하세요.",
                          ConvergenceWarning, stacklevel=2)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """분류 임계값 적용 전의 연속 선형 출력을 반환합니다.

        Args:
            X: 학습과 같은 열 수·열 순서의 유한한 수치형 입력.
        Returns:
            np.ndarray: 선형 출력 (n_samples,). 확률이 아니며 양성 임계값은 0.
        Raises:
            NotFittedError: fit 전 호출.
            ValueError: 비수치·비유한 입력 또는 다른 특성 수.
        """
        check_is_fitted(self, ["coef_", "loss_curve_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("입력 특성 수가 학습 자료와 다릅니다.")
        return X.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, X: Any) -> np.ndarray:
        """연속 출력 0을 경계로 원래 클래스 라벨을 반환합니다.

        Args:
            X: 학습과 같은 열 순서의 수치형 입력.
        Returns:
            np.ndarray: 원래 라벨 (n_samples,). 출력>=0이면 classes_[1].
        Raises:
            NotFittedError: fit 전 호출.
            ValueError: decision_function의 입력 검증 실패.
        """
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]


def _create_model(profile: AdalineProfile) -> Any:
    """이진 분류 목적에 맞는 미적합 모델을 생성합니다.

    Args:
        profile: 검증된 AdalineProfile.
    Returns:
        Any: sklearn 호환 미적합 추정기.
    Raises:
        ImportError: 필요한 모델 라이브러리 미설치.
        ValueError: 라이브러리가 설정을 거부한 경우.
    """
    return AdalineClassifier(max_iter=profile.max_iter, eta0=profile.eta0,
                             tol=profile.tol, fit_intercept=profile.fit_intercept)


def _fit_model(X: Any, y: Any, profile: AdalineProfile, estimator: Any) -> SupervisedOutputDTO:
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

def run_adaline(
    X: Any,
    y: Any,
    profile: Optional[AdalineProfile] = None,
) -> AdalineOutputDTO:
    """아달린(ADALINE) 이진 분류를 검증·학습하고 전용 DTO로 반환합니다.

    Args:
        X: 유한한 수치형 (n_samples, n_features) 배열 또는 DataFrame.
        y: 길이 n_samples의 1차원 이산 라벨. 정확히 두 클래스만 지원.
        profile: AdalineProfile. 기본 None이면 기본 조건을 사용.
    Returns:
        AdalineOutputDTO: 적합 Pipeline·평가·예측 및 모델별 결과.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·분할·CV·학습 오류.
        ImportError: 선택 모델 라이브러리 미설치.
    Notes:
        두 클래스를 classes[0]→-1, classes[1]→+1로 매핑하고 선형 출력의 평균 제곱 손실을 배치 경사하강법으로 최소화합니다. 임계값은 0이고 확률은 반환하지 않습니다. 가중치와 절편은 0으로 초기화합니다. 큰 학습률은 발산할 수 있으며 기본 standard 스케일링을 권장합니다. n_jobs는 CV에만 적용됩니다.
        입력 열 순서는 새 데이터 예측에서도 유지합니다. 자동 결측 대체·범주 인코딩은 하지 않습니다.
        기본 to_dict()/to_json()은 모델 실행 객체를 제외하고 분석 데이터를 반환합니다.
    """
    profile = profile if profile is not None else AdalineProfile()
    _validate_inputs(X, y, profile)
    estimator = _create_model(profile)
    result = _fit_model(X, y, profile, estimator)
    fitted = result.model.named_steps["model"]
    return AdalineOutputDTO(
        **result.to_dict(include_runtime=True),
        coefficients=fitted.coef_.copy(),
        intercepts=fitted.intercept_.copy(),
        activation_scores=result.model.decision_function(np.asarray(X)[result.test_indices]),
        loss_curve=fitted.loss_curve_.copy(),
        n_iter=int(fitted.n_iter_),
        converged=bool(fitted.converged_),
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    X, y = X[y < 2], y[y < 2]
    profile = AdalineProfile(max_iter=2000)
    result = run_adaline(X, y, profile)

    print("Test metrics:", result.test_metrics)
    print("Predictions:", result.predictions[:5])
    print("New predictions:", result.model.predict(X[:3]))
    print("Result keys:", result.to_dict().keys())
    print("JSON preview:", result.to_json(indent=2)[:300])
