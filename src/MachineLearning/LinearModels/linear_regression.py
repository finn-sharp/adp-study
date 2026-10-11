"""
================================================================================
linear_regression.py - 선형회귀(단순·다중)
================================================================================
설명:
    - 선형회귀(단순·다중)의 학습·평가·학습 구간 교차검증을 제공합니다.
    - 분석 조건은 Profile, 분석 결과는 Output DTO로 정의합니다.

분석 가정:
    - 독립·동일 분포 자료의 무작위 학습/테스트 분할을 사용합니다.
    - 입력은 결측값 없는 수치형 특성 및 단일 타깃입니다.
    - 테스트 자료를 모델 선택·스케일러 적합에 사용하지 않습니다.
    - OLS는 선형 조건부 평균을 적합합니다. 계수 추론을 위한 가정 검정은 수행하지 않습니다.

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


from src.output import OutputDTO
from src.MachineLearning.common import (
    SupervisedOutputDTO, run_supervised, validate_supervised_inputs,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class LinearRegressionProfile:
    """선형회귀(단순·다중)의 입력 조건.

    Attributes:
        task (str): 분석 목적. 기본 regression. regression만 지원.
        test_size (float): 보류 테스트 비율. 기본 0.3, 0 < 값 < 1.
        random_state (int): 분할·CV 난수 시드. OLS 적합 자체는 난수를 사용하지 않음. 기본 1234, bool 이외 정수.
        scaling (str): 스케일링 방법. none/standard/minmax; 기본 none. 학습 폴드 안에서만 적합.
        stratify (bool): 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음.
        param_grid (Optional[Dict[str, Any]]): 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용.
        cv (int): CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요.
        scoring (Optional[str]): sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error.
        n_jobs (int): CV 병렬 작업 수. 기본 1, 0 이외 정수. 모델 스레드 적용 여부는 Notes 참고.
        fit_intercept (bool): 절편 적합 여부. 기본 True. False이면 절편은 0.
        copy_X (bool): 모델 적합 시 입력 복사 여부. 기본 True.
        positive (bool): 계수를 0 이상으로 제한할지 여부. 기본 False. True이면 NNLS 사용.
    """

    task: str = "regression"
    test_size: float = 0.3
    random_state: int = 1234
    scaling: str = "none"
    stratify: bool = True
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 3
    scoring: Optional[str] = None
    n_jobs: int = 1
    fit_intercept: bool = True
    copy_X: bool = True
    positive: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class LinearRegressionOutputDTO(OutputDTO):
    """선형회귀(단순·다중)의 학습·평가 결과.

    Attributes:
        model_name (str): 적합된 추정기의 클래스 이름.
        task (str): 실행한 분석 목적. regression.
        model (Any): 적합된 sklearn Pipeline. 새 입력에는 model.predict(X_new)를 사용. 기본 사전/JSON에서 제외.
        train_metrics (Dict[str, float]): 학습 MAE/MSE/RMSE/R². 일반화 성능을 뜻하지 않음.
        test_metrics (Dict[str, float]): 보류 MAE/MSE/RMSE/R². MAE/RMSE는 y의 단위, MSE는 제곱 단위. R²는 음수일 수 있음.
        y_test (np.ndarray): 테스트 정답 (n_test,). test_indices와 동일한 행 순서.
        predictions (np.ndarray): 테스트 예측 (n_test,). y_test와 동일한 행 순서.
        train_indices (np.ndarray): 원본 입력의 학습 행 위치. DataFrame 인덱스 라벨이 아닌 정수 위치.
        test_indices (np.ndarray): 원본 입력의 보류 행 위치. train_indices와 서로 겹치지 않음.
        feature_names (List[str]): 입력 열 순서의 이름. 배열 입력은 feature_0 등의 이름.
        classes (Optional[np.ndarray]): 선형회귀에서는 None.
        probabilities (Optional[np.ndarray]): 회귀에서는 None. 선형회귀는 확률 분류를 지원하지 않음.
        confusion (Optional[np.ndarray]): 선형회귀에서는 None.
        best_params (Dict[str, Any]): CV 최적 Pipeline 파라미터. 탐색하지 않았으면 빈 사전.
        best_cv_score (Optional[float]): 학습 구간 CV 최적 점수. 기본 None; neg_mean_squared_error는 클수록 좋음.
        cv_results (Optional[Dict[str, Any]]): GridSearchCV 전체 결과. 탐색하지 않았으면 None.
        coefficients (Optional[np.ndarray]): 특성별 회귀 계수 (n_features,). 스케일러 변환 좌표계.
        intercept (Optional[float]): 회귀 절편. fit_intercept=False이면 0.0.
        residuals (Optional[np.ndarray]): 보류 잔차 y_test-predictions (n_test,). y와 같은 단위.
        rank (Optional[int]): 적합 설계 행렬의 rank. 절편 적합 시 중심화한 행렬 기준. positive=True이면 None.
        singular_values (Optional[np.ndarray]): 적합 설계 행렬의 특이값. positive=True이면 None.
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
    intercept: Optional[float] = None
    residuals: Optional[np.ndarray] = None
    rank: Optional[int] = None
    singular_values: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(X: Any, y: Any, profile: LinearRegressionProfile) -> None:
    """공통 입력과 모델별 설정을 검증합니다.

    Args:
        X: 최소 4행의 수치형 2차원 특성.
        y: X와 길이가 같은 단일 타깃.
        profile: LinearRegressionProfile 분석 조건.
    Returns:
        None: 검증 성공 시 반환.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·공통 조건·모델별 설정 오류.
    """
    if not isinstance(profile, LinearRegressionProfile):
        raise TypeError("LinearRegressionProfile이 필요합니다.")
    validate_supervised_inputs(X, y, profile)
    if profile.task != "regression":
        raise ValueError("LinearRegression은 regression만 지원합니다.")
    for name in ("fit_intercept", "copy_X", "positive"):
        if not isinstance(getattr(profile, name), bool):
            raise ValueError(name + "는 bool이어야 합니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _create_model(profile: LinearRegressionProfile) -> Any:
    """회귀 목적에 맞는 미적합 모델을 생성합니다.

    Args:
        profile: 검증된 LinearRegressionProfile.
    Returns:
        Any: sklearn 호환 미적합 추정기.
    Raises:
        ImportError: 필요한 모델 라이브러리 미설치.
        ValueError: 라이브러리가 설정을 거부한 경우.
    """
    from sklearn.linear_model import LinearRegression
    return LinearRegression(fit_intercept=profile.fit_intercept, copy_X=profile.copy_X,
                            positive=profile.positive, n_jobs=profile.n_jobs)


def _fit_model(X: Any, y: Any, profile: LinearRegressionProfile, estimator: Any) -> SupervisedOutputDTO:
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

def run_linear_regression(
    X: Any,
    y: Any,
    profile: Optional[LinearRegressionProfile] = None,
) -> LinearRegressionOutputDTO:
    """선형회귀(단순·다중)를 검증·학습하고 전용 DTO로 반환합니다.

    Args:
        X: 유한한 수치형 (n_samples, n_features) 배열 또는 DataFrame.
        y: 길이 n_samples의 유한한 1차원 수치형 회귀 타깃.
        profile: LinearRegressionProfile. 기본 None이면 기본 조건을 사용.
    Returns:
        LinearRegressionOutputDTO: 적합 Pipeline·평가·예측 및 모델별 결과.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·분할·CV·학습 오류.
        ImportError: 선택 모델 라이브러리 미설치.
    Notes:
        한 열이면 단순회귀, 여러 열이면 다중회귀입니다. 단일 타깃만 지원합니다. 기본 OLS이며 positive=True이면 NNLS입니다. 계수·절편은 스케일러 변환 좌표계이며 p-value·신뢰구간은 제공하지 않습니다. n_jobs는 모델과 CV에 전달합니다.
        입력 열 순서는 새 데이터 예측에서도 유지합니다. 자동 결측 대체·범주 인코딩은 하지 않습니다.
        기본 to_dict()/to_json()은 모델 실행 객체를 제외하고 분석 데이터를 반환합니다.
    """
    profile = profile if profile is not None else LinearRegressionProfile()
    _validate_inputs(X, y, profile)
    estimator = _create_model(profile)
    result = _fit_model(X, y, profile, estimator)
    fitted = result.model.named_steps["model"]
    return LinearRegressionOutputDTO(
        **result.to_dict(include_runtime=True),
        coefficients=np.asarray(fitted.coef_).copy(),
        intercept=float(fitted.intercept_),
        residuals=result.y_test - result.predictions,
        rank=int(fitted.rank_) if hasattr(fitted, "rank_") else None,
        singular_values=np.asarray(fitted.singular_).copy() if hasattr(fitted, "singular_") else None,
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=150, n_features=3, noise=1.0, random_state=1234)
    profile = LinearRegressionProfile()
    result = run_linear_regression(X, y, profile)

    print("Test metrics:", result.test_metrics)
    print("Predictions:", result.predictions[:5])
    print("New predictions:", result.model.predict(X[:3]))
    print("Result keys:", result.to_dict().keys())
    print("JSON preview:", result.to_json(indent=2)[:300])
