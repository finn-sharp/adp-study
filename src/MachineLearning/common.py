"""
================================================================================
common.py - 지도 학습의 데이터 분할, Pipeline, 교차검증 및 평가
================================================================================
설명:
    - 01장 데이터 분할·스케일링과 03/06/11/14장 Pipeline·모델 선택을 공통화합니다.
    - 테스트 데이터는 튜닝에 사용하지 않고, 스케일러는 학습 데이터에서만 적합합니다.
작성 정보:
    - 작성일: 2026-10-10
    - 버전: 1.0.0
변경 이력:
    2026-10-10  v1.0.0  Profile/Result와 공통 실행기 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""

__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.multiclass import type_of_target


@dataclass
class SupervisedProfile:
    """공통 학습 조건.

    Attributes:
        task: 'regression' 또는 'classification'. 단일 타깃만 지원합니다.
        test_size: 전체 표본 중 테스트 비율 (0 < test_size < 1).
        random_state: 분할·교차검증에 사용할 정수 시드.
        scaling: 'none', 'standard', 'minmax' 중 하나.
        stratify: 분류에서 클래스 비율을 유지할지 여부. 회귀에서는 무시합니다.
        param_grid: 선택적 탐색 공간. 모델 인자는 'alpha' 또는 'model__alpha'로 지정합니다.
        cv: param_grid가 있을 때 학습 데이터의 교차검증 폴드 수 (최소 2).
        scoring: sklearn 평가 이름. None이면 회귀는 neg_mean_squared_error, 분류는 accuracy.
        n_jobs: GridSearchCV 병렬 작업 수. 기본 1로 시험 환경 자원 사용을 제한합니다.
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


@dataclass
class SupervisedOutputDTO(OutputDTO):
    """지도 학습 결과.

    Attributes:
        model_name: 실제 추정기 클래스 이름.
        task: 회귀 또는 분류.
        model: 적합된 Pipeline. 새 데이터의 predict에도 동일 전처리를 적용합니다.
        train_metrics: 학습 데이터 지표 (일반화 성능 지표가 아님).
        test_metrics: 보류한 테스트 데이터 지표.
        y_test: 테스트 정답, 원래 입력 순서가 아닌 test_indices 순서.
        predictions: 테스트 예측값, y_test와 같은 순서.
        train_indices: 입력 데이터의 학습 행 위치.
        test_indices: 입력 데이터의 테스트 행 위치.
        feature_names: DataFrame 열 이름 또는 자동 생성된 이름 (입력 순서).
        classes: 분류 클래스 순서. 회귀는 None.
        probabilities: predict_proba가 있는 분류 모델의 테스트 확률. 나머지는 None.
        confusion: classes 순서의 혼동행렬. 회귀는 None.
        best_params: 탐색을 수행한 경우 Pipeline 파라미터 이름의 최적 설정.
        best_cv_score: 학습 데이터 CV 최적 점수. 음의 MSE는 클수록 좋습니다.
        cv_results: 탐색 결과 사전. 탐색을 생략하면 None.
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


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
SupervisedResult = SupervisedOutputDTO


def validate_features(X: Any, min_samples: int = 2) -> np.ndarray:
    """수치형 2차원 특성 행렬을 검증합니다.

    Args:
        X: 배열 또는 DataFrame. 결측값은 호출 전에 처리해야 합니다.
        min_samples: 필요한 최소 행 수.
    Returns:
        np.ndarray: 유한한 float 행렬.
    Raises:
        ValueError: 비수치 데이터, NaN/무한대, 빈 열 또는 잘못된 차원·행 수.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < min_samples or arr.shape[1] < 1:
        raise ValueError("X는 최소 {}행, 1열의 2차원 수치 행렬이어야 합니다.".format(min_samples))
    if not np.isfinite(arr).all():
        raise ValueError("X의 결측값과 무한대를 먼저 처리해야 합니다.")
    return arr


def make_scaler(scaling: str) -> Any:
    """스케일러를 생성합니다.

    Args:
        scaling: 'none', 'standard', 'minmax'.
    Returns:
        Any: 미적합 스케일러 또는 'passthrough'.
    Raises:
        ValueError: 지원하지 않는 스케일링 이름.
    """
    if scaling not in ("none", "standard", "minmax"):
        raise ValueError("scaling은 none, standard, minmax 중 하나여야 합니다.")
    return {"none": "passthrough", "standard": StandardScaler(), "minmax": MinMaxScaler()}[scaling]


def _metrics(y: np.ndarray, prediction: np.ndarray, task: str) -> Dict[str, float]:
    if task == "regression":
        mse = float(mean_squared_error(y, prediction))
        return {"mae": float(mean_absolute_error(y, prediction)), "mse": mse,
                "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y, prediction))}
    return {"accuracy": float(accuracy_score(y, prediction)),
            "precision_weighted": float(precision_score(y, prediction, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y, prediction, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y, prediction, average="weighted", zero_division=0))}


def run_supervised(X: Any, y: Any, estimator: Any, profile: SupervisedProfile,
                   extra_steps: Optional[List[Tuple[str, Any]]] = None) -> SupervisedOutputDTO:
    """분할 → Pipeline 적합/학습 데이터 CV → 테스트 평가를 수행합니다.

    Args:
        X: 결측치 없는 수치형 2차원 특성. DataFrame 열 순서는 예측 시에도 유지합니다.
        y: 길이가 X와 같은 1차원 타깃. 분류는 일관된 문자열 또는 숫자 라벨.
        estimator: profile.task에 맞는 sklearn 추정기 (각 모델 실행기가 선택).
        profile: 공통 학습 조건 및 모델별 설정.
        extra_steps: 스케일러 뒤, 모델 앞에 배치할 미적합 변환기. 예: ('pca', PCA()).
    Returns:
        SupervisedOutputDTO: 학습 Pipeline, 평가 지표, 분할 위치 및 선택적 CV 결과.
    Raises:
        ValueError: 잘못된 입력·설정, 표본/클래스 수 부족 또는 유효하지 않은 모델 인자.

    Notes:
        무작위 IID 분할입니다. 시계열·그룹 데이터는 이 실행기 대신 별도 분할이 필요합니다.
        분류 ROC-AUC는 이진 타깃과 predict_proba가 있고 테스트에 두 클래스가 있을 때만 반환합니다.
        양성 클래스는 classes[1]입니다. SVC의 probability=False에서는 AUC를 반환하지 않습니다.
    """
    if profile is None or profile.task not in ("regression", "classification"):
        raise ValueError("profile.task는 regression 또는 classification이어야 합니다.")
    if not isinstance(profile.test_size, (float, int)) or not 0 < profile.test_size < 1:
        raise ValueError("test_size는 0과 1 사이의 비율이어야 합니다.")
    if not isinstance(profile.random_state, int) or isinstance(profile.random_state, bool):
        raise ValueError("random_state는 정수여야 합니다.")
    if not isinstance(profile.cv, int) or isinstance(profile.cv, bool) or profile.cv < 2:
        raise ValueError("cv는 2 이상의 정수여야 합니다.")
    if not isinstance(profile.n_jobs, int) or isinstance(profile.n_jobs, bool) or profile.n_jobs == 0:
        raise ValueError("n_jobs는 0이 아닌 정수여야 합니다.")
    arr = validate_features(X, min_samples=4)
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != len(arr):
        raise ValueError("y는 X와 길이가 같은 1차원 타깃이어야 합니다.")
    if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) for v in target):
        raise ValueError("y에 결측값이나 무한대를 포함할 수 없습니다.")
    if profile.task == "regression":
        target = target.astype(float)
        if not np.isfinite(target).all():
            raise ValueError("회귀 타깃은 유한한 수치여야 합니다.")
    elif type_of_target(target) not in ("binary", "multiclass") or len(np.unique(target)) < 2:
        raise ValueError("분류 타깃에는 최소 두 클래스의 이산 라벨이 필요합니다.")
    stratify = target if profile.task == "classification" and profile.stratify else None
    train_idx, test_idx = train_test_split(np.arange(len(arr)), test_size=profile.test_size,
                                          random_state=profile.random_state, stratify=stratify)
    if min(len(train_idx), len(test_idx)) < 2:
        raise ValueError("학습·테스트 데이터는 각각 최소 2행이어야 합니다.")
    X_train, X_test = arr[train_idx], arr[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    if profile.task == "classification" and len(np.unique(y_train)) < 2:
        raise ValueError("학습 데이터에 최소 두 클래스가 필요합니다.")
    steps = [("scaler", make_scaler(profile.scaling))] + (extra_steps or []) + [("model", estimator)]
    pipeline = Pipeline(steps)
    best_params, best_score, cv_results = {}, None, None
    if profile.param_grid is not None:
        if not isinstance(profile.param_grid, dict) or not profile.param_grid:
            raise ValueError("param_grid는 비어 있지 않은 사전이어야 합니다.")
        if profile.task == "classification":
            counts = np.unique(y_train, return_counts=True)[1]
            if counts.min() < profile.cv:
                raise ValueError("각 학습 클래스의 표본 수는 cv 이상이어야 합니다.")
            cv = StratifiedKFold(profile.cv, shuffle=True, random_state=profile.random_state)
        else:
            if len(y_train) < profile.cv:
                raise ValueError("학습 표본 수는 cv 이상이어야 합니다.")
            cv = KFold(profile.cv, shuffle=True, random_state=profile.random_state)
        grid = {k if "__" in k else "model__" + k: v for k, v in profile.param_grid.items()}
        scoring = profile.scoring or ("accuracy" if profile.task == "classification" else "neg_mean_squared_error")
        search = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring, n_jobs=profile.n_jobs,
                              refit=True, error_score="raise")
        search.fit(X_train, y_train)
        pipeline, best_params = search.best_estimator_, search.best_params_
        best_score, cv_results = float(search.best_score_), search.cv_results_
    else:
        pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    # sklearn 0.23의 일부 회귀 추정기도 classes_ 속성을 노출하므로 task로 먼저 구분합니다.
    classes = getattr(pipeline.named_steps["model"], "classes_", None) if profile.task == "classification" else None
    probabilities = None
    test_metrics = _metrics(y_test, prediction, profile.task)
    if profile.task == "classification" and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        if len(classes) == 2 and len(np.unique(y_test)) == 2:
            test_metrics["roc_auc"] = float(roc_auc_score(y_test == classes[1], probabilities[:, 1]))
    feature_names = [str(c) for c in X.columns] if hasattr(X, "columns") else ["feature_{}".format(i) for i in range(arr.shape[1])]
    return SupervisedOutputDTO(
        model_name=type(pipeline.named_steps["model"]).__name__, task=profile.task, model=pipeline,
        train_metrics=_metrics(y_train, pipeline.predict(X_train), profile.task), test_metrics=test_metrics,
        y_test=y_test, predictions=prediction, train_indices=train_idx, test_indices=test_idx,
        feature_names=feature_names, classes=classes, probabilities=probabilities,
        confusion=confusion_matrix(y_test, prediction, labels=classes) if classes is not None else None,
        best_params=best_params, best_cv_score=best_score, cv_results=cv_results)
