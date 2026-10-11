"""
================================================================================
regularized.py - L1·L2 규제 기반 변수 선택
================================================================================
설명:
    - L1·L2 규제 기반 변수 선택을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 학습 자료에만 계수와 임계값을 적합합니다. L2 규제는 계수를 자동으로 0으로 만들지 않습니다.
    - 결측값과 범주형 인코딩은 실행 전에 처리합니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.0.0    모델별 레시피·전체 명세·실행 예제 작성
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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.output import OutputDTO
from src.Recipes import load_profile, resolve_profile
from src.MachineLearning.common import validate_features, make_scaler
from src.MachineLearning.analysis_common import positive_int, finite_number, validate_scaling, feature_names
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.MachineLearning.FeatureSelection.wrapper import _base_estimator


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class RegularizedSelectionProfile:
    """L1·L2 규제 기반 변수 선택 입력 조건.

    Attributes:
        task (str): 분석 목적. classification/regression 지원. 기본 'classification'.
        test_size (float): 보류 테스트 비율. 기본 0.3, 0 < 값 < 1.
        random_state (int): 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수.
        scaling (str): 스케일링 방법. none/standard/minmax; 기본 standard. 학습 폴드 안에서만 적합.
        stratify (bool): 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음.
        param_grid (Optional[Dict[str, Any]]): 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용.
        cv (int): CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요.
        scoring (Optional[str]): sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error.
        n_jobs (int): CV 병렬 작업 수. 기본 1, 0 이외 정수. 후보 평가/CV의 작업 수. 선택용 선형 모델은 별도 스레드 설정을 지정하지 않음.
        penalty (str): l1/l2. L2는 자동 희소화가 아니라 중요도 임계값 선택. 기본 "l1".
        alpha (float): 회귀 Lasso/Ridge의 규제 강도. 유한한 양수. 기본 0.01.
        C (float): 분류 LogisticRegression 역규제 강도. 유한한 양수. 기본 1.0.
        threshold (Optional[float]): 계수 절대값 평균의 포함 하한. None이면 L1 1e-8, L2 평균 중요도. 양수. 기본 None.
        max_iter (int): 최대 반복 횟수. 양의 정수. 기본 5000.
    """

    task: str = 'classification'
    test_size: float = 0.3
    random_state: int = 1234
    scaling: str = 'standard'
    stratify: bool = True
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 3
    scoring: Optional[str] = None
    n_jobs: int = 1
    penalty: str = "l1"
    alpha: float = 0.01
    C: float = 1.0
    threshold: Optional[float] = None
    max_iter: int = 5000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class RegularizedSelectionOutputDTO(OutputDTO):
    """L1·L2 규제 기반 변수 선택 결과. 지도 예측은 test_indices, 선택 자료는 해당 분할 행 순서.

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
        classes (Optional[np.ndarray]): 분류 클래스 순서 (n_classes,). 회귀는 None; 이진 AUC 양성 클래스는 classes[1]. 기본 None.
        probabilities (Optional[np.ndarray]): 분류 확률 (n_test, n_classes). classes 열 순서. 회귀와 Perceptron은 None. 기본 None.
        confusion (Optional[np.ndarray]): classes 순서의 혼동행렬 (n_classes, n_classes). 행은 정답, 열은 예측; 회귀는 None. 기본 None.
        best_params (Dict[str, Any]): CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전. 기본 field(default_factory=dict).
        best_cv_score (Optional[float]): 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음.
        cv_results (Optional[Dict[str, Any]]): GridSearchCV 전체 결과. 탐색하지 않았으면 None. 기본 None.
        selector (Any): 적합된 선택 변환기. 원 입력에서 선택 열을 transform. 기본 JSON 제외.
        selected_indices (np.ndarray): 원 특성 순서의 선택 열 위치.
        selected_features (List[str]): 선택 열 이름. 원 입력 열 순서.
        support_mask (np.ndarray): (n_features,) 선택 여부.
        transformed_train (np.ndarray): 선택 열의 원 좌표 학습 자료. train_indices 행 순서.
        transformed_test (np.ndarray): 선택 열의 원 좌표 보류 자료. test_indices 행 순서.
        importances (np.ndarray): (n_features,) 전처리 좌표계의 계수 절대값. 다중 클래스는 클래스 평균.
        coefficients (np.ndarray): 선택용 모델의 계수. 회귀 (n_features,), 분류 (n_estimators,n_features).
        applied_threshold (float): 실제 적용한 중요도 임계값.
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
    selector: Any
    selected_indices: np.ndarray
    selected_features: List[str]
    support_mask: np.ndarray
    transformed_train: np.ndarray
    transformed_test: np.ndarray
    importances: np.ndarray
    coefficients: np.ndarray
    applied_threshold: float
    classes: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confusion: Optional[np.ndarray] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_cv_score: Optional[float] = None
    cv_results: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate_inputs(X: Any, y: Any, profile: RegularizedSelectionProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        y: X와 길이가 같은 이산 클래스 라벨.
        profile: RegularizedSelectionProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, RegularizedSelectionProfile):
        raise TypeError("RegularizedSelectionProfile이 필요합니다.")
    arr = validate_features(X)
    from src.MachineLearning.common import validate_supervised_inputs
    arr, target = validate_supervised_inputs(X, y, profile)
    validate_scaling(profile.scaling, profile.random_state)
    feature_names(X)
    if profile.penalty not in ("l1", "l2"): raise ValueError("penalty는 l1/l2입니다.")
    finite_number(profile.alpha, "alpha", positive=True)
    finite_number(profile.C, "C", positive=True)
    positive_int(profile.max_iter, "max_iter")
    if profile.threshold is not None: finite_number(profile.threshold, "threshold", positive=True)
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

class RegularizedSelector(TransformerMixin, BaseEstimator):
    """학습 자료의 규제 모델 계수로 선택하는 sklearn 변환기."""

    def __init__(self, task="classification", penalty="l1", alpha=0.01, C=1.0,
                 threshold=None, max_iter=5000, scaling="standard", random_state=1234):
        """규제 및 임계값 조건을 저장합니다.

        Args:
            task: classification/regression.
            penalty: l1/l2.
            alpha: 회귀 규제 강도.
            C: 분류 역규제 강도.
            threshold: 중요도 양수 하한 또는 자동 None.
            max_iter: 최대 반복.
            scaling: 계수 적합 좌표계의 스케일링.
            random_state: 분류 난수 시드.
        Returns:
            None: 설정 저장.
        Raises:
            None: fit에서 검증.
        """
        self.task = task
        self.penalty = penalty
        self.alpha = alpha
        self.C = C
        self.threshold = threshold
        self.max_iter = max_iter
        self.scaling = scaling
        self.random_state = random_state

    def fit(self, X, y):
        """현재 학습 자료에서 스케일러와 규제 모델을 적합합니다.

        Args:
            X: 현재 학습 폴드의 원 좌표 특성.
            y: 행 순서가 같은 타깃.
        Returns:
            RegularizedSelector: 적합 self.
        Raises:
            ValueError: 잘못된 자료·설정 또는 선택 변수가 0개인 경우.
        """
        from sklearn.linear_model import Lasso, Ridge, LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        arr = _validate_inputs(X, y, RegularizedSelectionProfile(**self.get_params()))
        self.n_features_in_ = arr.shape[1]
        if self.task == "regression":
            estimator = Lasso(alpha=self.alpha, max_iter=self.max_iter) if self.penalty == "l1" else Ridge(alpha=self.alpha, solver="lsqr", max_iter=self.max_iter)
        else:
            estimator = OneVsRestClassifier(LogisticRegression(penalty=self.penalty, C=self.C,
                solver="liblinear", max_iter=self.max_iter, random_state=self.random_state))
        self.selection_model_ = Pipeline([("scaler", make_scaler(self.scaling)), ("model", estimator)])
        self.selection_model_.fit(arr, y)
        self.coefficients_ = np.vstack([m.coef_.reshape(-1) for m in estimator.estimators_]) if self.task == "classification" else estimator.coef_.copy()
        self.importances_ = np.abs(self.coefficients_) if self.coefficients_.ndim == 1 else np.mean(np.abs(self.coefficients_), axis=0)
        self.threshold_ = self.threshold if self.threshold is not None else (1e-8 if self.penalty == "l1" else float(self.importances_.mean()))
        self.support_ = (self.importances_ >= self.threshold_) & (self.importances_ > 0)
        if not self.support_.any(): raise ValueError("선택 변수가 없습니다. 규제 강도와 threshold를 조정하세요.")
        return self

    def get_support(self):
        """원 특성 순서의 선택 마스크를 반환합니다.

        Args:
            None: 추가 인자 없음.
        Returns:
            np.ndarray: 선택 마스크 복사.
        Raises:
            NotFittedError: 미적합 선택기.
        """
        check_is_fitted(self, "support_")
        return self.support_.copy()

    def transform(self, X):
        """선택 열을 원 입력 좌표로 반환합니다.

        Args:
            X: 학습 열 순서의 수치 특성. 단일 행도 허용.
        Returns:
            np.ndarray: 선택된 열.
        Raises:
            NotFittedError/ValueError: 미적합 또는 입력/열 수 오류.
        """
        mask = self.get_support()
        arr = validate_features(X, min_samples=1)
        if arr.shape[1] != len(mask): raise ValueError("학습 특성 수와 다릅니다.")
        return arr[:, mask]


def _run_model(X: Any, y: Any, profile: RegularizedSelectionProfile) -> RegularizedSelectionOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        y: 검증된 클래스 라벨.
        profile: 검증된 RegularizedSelectionProfile.
    Returns:
        RegularizedSelectionOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    selector = RegularizedSelector(task=profile.task, penalty=profile.penalty, alpha=profile.alpha,
                                   C=profile.C, threshold=profile.threshold, max_iter=profile.max_iter,
                                   scaling=profile.scaling, random_state=profile.random_state)
    from dataclasses import replace
    from src.MachineLearning.common import run_supervised
    # 선택기의 내부 CV가 원 입력에서 스케일러를 폴드별 적합하도록 외부 선행 스케일러를 비활성화합니다.
    outer = replace(profile, scaling="none")
    result = run_supervised(X, y, _base_estimator(profile.task, profile.C, profile.max_iter), outer,
                            extra_steps=[("selector", selector), ("selected_scaler", make_scaler(profile.scaling))])
    fitted = result.model.named_steps["selector"]
    indices = np.flatnonzero(fitted.get_support())
    arr = validate_features(X)
    names = feature_names(X)
    return RegularizedSelectionOutputDTO(**result.to_dict(include_runtime=True), selector=fitted,
        selected_indices=indices, selected_features=[names[i] for i in indices], support_mask=fitted.get_support(),
        transformed_train=fitted.transform(arr[result.train_indices]), transformed_test=fitted.transform(arr[result.test_indices]),
        importances=fitted.importances_.copy(), coefficients=fitted.coefficients_.copy(), applied_threshold=fitted.threshold_)


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_regularized_selection(X: Any, y: Any, profile: Optional[RegularizedSelectionProfile] = None, *,
              task: Optional[str] = None,
              recipe_path: Optional[str] = None,
              overrides: Optional[Dict[str, Any]] = None) -> RegularizedSelectionOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        y: 길이 n_samples의 타깃. 분류는 이산 라벨, 회귀는 유한한 수치.
        profile: 명시적 RegularizedSelectionProfile. None이면 regularized-selection.json을 읽음.
        task: Profile 생략 시 선택할 classification/regression 목적.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        RegularizedSelectionOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile("regularized-selection", profile, recipe_path=recipe_path, task=task, overrides=overrides)
    _validate_inputs(X, y, profile)
    return _run_model(X, y, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = run_regularized_selection(X, y, load_profile("regularized-selection"))
    print(result.to_json(indent=2)[:500])
