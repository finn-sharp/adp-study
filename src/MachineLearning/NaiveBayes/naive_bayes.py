"""
================================================================================
naive_bayes.py - 나이브 베이즈 분류
================================================================================
설명:
    - 나이브 베이즈 분류을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 클래스 조건부 특성 독립을 가정합니다. 전처리·CV·모델은 학습 구간에서만 적합합니다.
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



# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class NaiveBayesProfile:
    """나이브 베이즈 분류 입력 조건.

    Attributes:
        task (str): 분석 목적. 기본 classification. classification만 지원.
        test_size (float): 보류 테스트 비율. 기본 0.3, 0 < 값 < 1.
        random_state (int): 분할·CV·모델 난수 시드. 기본 1234, bool 이외 정수.
        scaling (str): 스케일링 방법. none/standard/minmax; 기본 none. 빈도·Bernoulli 모델은 none만 지원. 학습 폴드 안에서만 적합. 기본 "none".
        stratify (bool): 분류의 계층 분할 여부. 기본 True; 회귀에서는 적용하지 않음.
        param_grid (Optional[Dict[str, Any]]): 학습 구간 CV 탐색 사전. 기본 None. 모델 인자명 또는 model__인자명을 사용.
        cv (int): CV 폴드 수. 기본 3, 2 이상 정수. param_grid 지정 시 각 학습 클래스에 cv개 이상 필요.
        scoring (Optional[str]): sklearn CV 지표명. 기본 None이면 분류 accuracy, 회귀 neg_mean_squared_error.
        n_jobs (int): CV 병렬 작업 수. 기본 1, 0 이외 정수. 후보 평가/CV의 작업 수. 선택용 선형 모델은 별도 스레드 설정을 지정하지 않음.
        variant (str): gaussian/multinomial/bernoulli/complement. 기본 "gaussian".
        alpha (float): 가산 평활화 강도. 유한한 양수. Gaussian에는 미사용. 기본 1.0.
        var_smoothing (float): Gaussian 분산 안정화 강도. 양수. 기본 1e-9.
        binarize (Optional[float]): Bernoulli 이진화 임계값. 유한한 수 또는 None. None이면 입력은 0/1이어야 함. 기본 0.0.
    """

    task: str = 'classification'
    test_size: float = 0.3
    random_state: int = 1234
    scaling: str = "none"
    stratify: bool = True
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 3
    scoring: Optional[str] = None
    n_jobs: int = 1
    variant: str = "gaussian"
    alpha: float = 1.0
    var_smoothing: float = 1e-9
    binarize: Optional[float] = 0.0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class NaiveBayesOutputDTO(OutputDTO):
    """나이브 베이즈 분류 결과. 지도 예측은 test_indices, 선택 자료는 해당 분할 행 순서.

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
        class_counts (np.ndarray): 클래스별 학습 관측 수. classes 순서.
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
    class_counts: np.ndarray
    classes: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confusion: Optional[np.ndarray] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_cv_score: Optional[float] = None
    cv_results: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate_inputs(X: Any, y: Any, profile: NaiveBayesProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        y: X와 길이가 같은 이산 클래스 라벨.
        profile: NaiveBayesProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, NaiveBayesProfile):
        raise TypeError("NaiveBayesProfile이 필요합니다.")
    arr = validate_features(X)
    from src.MachineLearning.common import validate_supervised_inputs
    arr, target = validate_supervised_inputs(X, y, profile)
    if profile.task != "classification": raise ValueError("나이브 베이즈는 classification만 지원합니다.")
    if profile.variant not in ("gaussian", "multinomial", "bernoulli", "complement"): raise ValueError("variant를 확인하세요.")
    finite_number(profile.alpha, "alpha", positive=True)
    finite_number(profile.var_smoothing, "var_smoothing", positive=True)
    if profile.binarize is not None and (isinstance(profile.binarize, bool) or not isinstance(profile.binarize, (float,int)) or not np.isfinite(profile.binarize)):
        raise ValueError("binarize는 유한한 수 또는 None입니다.")
    if profile.variant in ("multinomial", "complement"):
        if profile.scaling != "none" or np.any(arr < 0): raise ValueError("빈도 기반 Bayes는 음이 아닌 입력과 scaling=none이 필요합니다.")
    if profile.variant == "bernoulli":
        if profile.scaling != "none": raise ValueError("Bernoulli Bayes는 scaling=none을 사용하세요.")
        if profile.binarize is None and not np.isin(arr, [0,1]).all(): raise ValueError("이진화 생략 시 입력은 0/1이어야 합니다.")
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_model(X: Any, y: Any, profile: NaiveBayesProfile) -> NaiveBayesOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        y: 검증된 클래스 라벨.
        profile: 검증된 NaiveBayesProfile.
    Returns:
        NaiveBayesOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
    from src.MachineLearning.common import run_supervised
    factories = {"gaussian": lambda: GaussianNB(var_smoothing=profile.var_smoothing),
                 "multinomial": lambda: MultinomialNB(alpha=profile.alpha),
                 "bernoulli": lambda: BernoulliNB(alpha=profile.alpha, binarize=profile.binarize),
                 "complement": lambda: ComplementNB(alpha=profile.alpha)}
    result = run_supervised(X, y, factories[profile.variant](), profile)
    return NaiveBayesOutputDTO(**result.to_dict(include_runtime=True),
                              class_counts=result.model.named_steps["model"].class_count_.copy())


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_naive_bayes(X: Any, y: Any, profile: Optional[NaiveBayesProfile] = None, *,
              task: Optional[str] = None,
              recipe_path: Optional[str] = None,
              overrides: Optional[Dict[str, Any]] = None) -> NaiveBayesOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        y: 길이 n_samples의 타깃. 분류는 이산 라벨, 회귀는 유한한 수치.
        profile: 명시적 NaiveBayesProfile. None이면 naive-bayes.json을 읽음.
        task: Profile 생략 시 선택할 classification/regression 목적.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        NaiveBayesOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile("naive-bayes", profile, recipe_path=recipe_path, task=task, overrides=overrides)
    _validate_inputs(X, y, profile)
    return _run_model(X, y, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = run_naive_bayes(X, y, load_profile("naive-bayes"))
    print(result.to_json(indent=2)[:500])
