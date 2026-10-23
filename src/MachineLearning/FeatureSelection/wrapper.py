"""
================================================================================
wrapper.py - CV 기반 Wrapper 변수 선택
================================================================================
설명:
    - CV 기반 Wrapper 변수 선택을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - IID 분할입니다. 후보 평가는 학습 자료 내부 CV로 수행하고 보류 자료는 최종 평가에만 사용합니다.
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
from src.MachineLearning.analysis_common import (
    positive_int,
    finite_number,
    validate_scaling,
    feature_names,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class WrapperSelectionProfile:
    """CV 기반 Wrapper 변수 선택 입력 조건.

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
        method (str): forward/backward/stepwise. Stepwise는 CV 점수 기반 양방향 개선 탐색. 기본 "forward".
        n_features_to_select (Optional[int]): Forward/Backward의 목표 수. None이면 개선 기준 자동 종료. Stepwise에서는 최대 수. 기본 2.
        min_improvement (float): 자동/Stepwise 종료를 위한 최소 CV 점수 개선. 유한한 0 이상. 기본 0.0001.
        max_steps (int): 최대 이동 횟수. 양의 정수. 한도 도달은 history와 stopped_reason 확인. 기본 100.
        C (float): 분류 LogisticRegression의 역규제 강도. 양수. 회귀는 OLS 사용. 기본 1.0.
        max_iter (int): 분류 LogisticRegression 최대 반복. 양의 정수. 기본 2000.
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
    method: str = "forward"
    n_features_to_select: Optional[int] = 2
    min_improvement: float = 0.0001
    max_steps: int = 100
    C: float = 1.0
    max_iter: int = 2000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class WrapperSelectionOutputDTO(OutputDTO):
    """CV 기반 Wrapper 변수 선택 결과. 지도 예측은 test_indices, 선택 자료는 해당 분할 행 순서.

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
        history (List[Dict[str, Any]]): 선택 이동 기록. 열 위치·action·평균 CV 점수.
        selection_cv_score (float): 최종 선택 집합의 학습 자료 내부 CV 점수. 보류 성능 아님.
        stopped_reason (str): target/no_improvement/max_steps/no_candidates 중 종료 이유.
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
    history: List[Dict[str, Any]]
    selection_cv_score: float
    stopped_reason: str
    classes: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confusion: Optional[np.ndarray] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_cv_score: Optional[float] = None
    cv_results: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(X: Any, y: Any, profile: WrapperSelectionProfile) -> np.ndarray:
    """자료와 Profile 조건을 검증합니다.

    Args:
        X: 유한한 수치형 2차원 특성.
        y: X와 길이가 같은 이산 클래스 라벨.
        profile: WrapperSelectionProfile 조건.
    Returns:
        np.ndarray: 검증된 float 특성 행렬.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, WrapperSelectionProfile):
        raise TypeError("WrapperSelectionProfile이 필요합니다.")
    arr = validate_features(X)
    from src.MachineLearning.common import validate_supervised_inputs

    arr, target = validate_supervised_inputs(X, y, profile)
    validate_scaling(profile.scaling, profile.random_state)
    feature_names(X)
    if profile.n_features_to_select is not None:
        positive_int(profile.n_features_to_select, "n_features_to_select")
        if profile.n_features_to_select > arr.shape[1]:
            raise ValueError("선택 변수 수는 입력 열 수 이하여야 합니다.")
    if profile.method not in ("forward", "backward", "stepwise"):
        raise ValueError("method를 확인하세요.")
    finite_number(profile.min_improvement, "min_improvement")
    finite_number(profile.C, "C", positive=True)
    positive_int(profile.max_iter, "max_iter")
    positive_int(profile.max_steps, "max_steps")
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _base_estimator(task: str, C: float, max_iter: int) -> Any:
    """선택 평가와 최종 예측에 사용할 선형 모델을 생성합니다.

    Args:
        task: classification/regression.
        C: 분류 역규제 강도.
        max_iter: 분류 최대 반복.
    Returns:
        Any: 분류 LogisticRegression 또는 회귀 LinearRegression.
    Raises:
        ValueError: 잘못된 목적.
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if task == "classification":
        return LogisticRegression(C=C, max_iter=max_iter)
    if task == "regression":
        return LinearRegression()
    raise ValueError("지원하지 않는 task입니다.")


class WrapperSelector(TransformerMixin, BaseEstimator):
    """폴드 내부 전처리를 사용하는 greedy 변수 선택 변환기."""

    def __init__(
        self,
        task="classification",
        method="forward",
        n_features_to_select=2,
        min_improvement=0.0001,
        max_steps=100,
        scaling="standard",
        cv=3,
        scoring=None,
        random_state=1234,
        n_jobs=1,
        C=1.0,
        max_iter=2000,
    ):
        """탐색 설정을 저장합니다.

        Args:
            task: 분류/회귀.
            method: forward/backward/stepwise.
            n_features_to_select: 목표 수 또는 Stepwise 최대 수. None이면 자동.
            min_improvement: 자동 종료 최소 점수 개선.
            max_steps: 이동 한도.
            scaling: 폴드별 전처리.
            cv: 내부 CV 수.
            scoring: sklearn 점수 이름 또는 task 기본 None.
            random_state: 분할 시드.
            n_jobs: CV 작업 수.
            C: 분류 역규제 강도.
            max_iter: 분류 반복 한도.
        Returns:
            None: 설정 저장.
        Raises:
            None: fit에서 검증.
        """
        self.task = task
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.min_improvement = min_improvement
        self.max_steps = max_steps
        self.scaling = scaling
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        """CV 평균 점수로 후보 집합을 평가하고 선택 마스크를 적합합니다.

        Args:
            X: 현재 학습 폴드의 원 좌표 특성.
            y: 같은 행 순서의 타깃.
        Returns:
            WrapperSelector: 적합 self.
        Raises:
            ValueError: 조건 오류, CV 표본 부족 또는 모델 적합 실패.
        """
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        arr = _validate_inputs(X, y, WrapperSelectionProfile(**self.get_params()))
        target = np.asarray(y)
        if self.task == "classification":
            if np.unique(target, return_counts=True)[1].min() < self.cv:
                raise ValueError("각 학습 클래스에 cv개 이상의 관측이 필요합니다.")
            splitter = StratifiedKFold(
                self.cv, shuffle=True, random_state=self.random_state
            )
        else:
            if len(arr) // self.cv < 2:
                raise ValueError("회귀 CV 평가 폴드별 최소 2행이 필요합니다.")
            splitter = KFold(self.cv, shuffle=True, random_state=self.random_state)
        splits = list(splitter.split(arr, target))
        scoring = self.scoring or (
            "accuracy" if self.task == "classification" else "neg_mean_squared_error"
        )
        cache = {}

        def evaluate(columns):
            """선택 집합의 폴드별 평균 점수를 캐시합니다.

            Args:
                columns: 선택 열 위치 목록.
            Returns:
                float: 평균 CV 점수.
            Raises:
                ValueError: 적합 실패 또는 비유한 점수.
            """
            key = tuple(sorted(columns))
            if key not in cache:
                pipeline = Pipeline(
                    [
                        ("scaler", make_scaler(self.scaling)),
                        ("model", _base_estimator(self.task, self.C, self.max_iter)),
                    ]
                )
                score = cross_val_score(
                    pipeline,
                    arr[:, key],
                    target,
                    cv=splits,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                    error_score="raise",
                )
                if not np.isfinite(score).all():
                    raise ValueError("CV 점수는 유한해야 합니다.")
                cache[key] = float(score.mean())
            return cache[key]

        self.n_features_in_ = arr.shape[1]
        selected = list(range(self.n_features_in_)) if self.method == "backward" else []
        score = evaluate(selected) if selected else -np.inf
        visited = {tuple(selected)}
        self.history_ = []
        self.stopped_reason_ = "max_steps"
        cap = self.n_features_to_select or self.n_features_in_
        for step in range(self.max_steps):
            if (
                self.method != "stepwise"
                and self.n_features_to_select is not None
                and len(selected) == cap
            ):
                self.stopped_reason_ = "target"
                break
            candidates = []
            if self.method in ("forward", "stepwise") and len(selected) < cap:
                candidates.extend(
                    ("add", sorted(selected + [i]))
                    for i in range(self.n_features_in_)
                    if i not in selected
                )
            if self.method in ("backward", "stepwise") and len(selected) > 1:
                candidates.extend(
                    ("remove", [i for i in selected if i != j]) for j in selected
                )
            candidates = [
                (action, cols)
                for action, cols in candidates
                if tuple(cols) not in visited
            ]
            if not candidates:
                self.stopped_reason_ = "no_candidates"
                break
            ranked = sorted(
                [(evaluate(cols), tuple(cols), action) for action, cols in candidates],
                key=lambda v: (-v[0], v[1]),
            )
            best_score, columns, action = ranked[0]
            automatic = self.n_features_to_select is None or self.method == "stepwise"
            if automatic and selected and best_score <= score + self.min_improvement:
                self.stopped_reason_ = "no_improvement"
                break
            selected, score = list(columns), best_score
            visited.add(columns)
            self.history_.append(
                dict(
                    step=step + 1,
                    action=action,
                    indices=selected.copy(),
                    cv_score=score,
                )
            )
        if (
            self.method != "stepwise"
            and self.n_features_to_select is not None
            and len(selected) == cap
        ):
            self.stopped_reason_ = "target"
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[selected] = True
        self.best_score_ = float(score)
        return self

    def get_support(self):
        """원 열 순서의 선택 마스크를 반환합니다.

        Args:
            None: 추가 인자 없음.
        Returns:
            np.ndarray: bool 마스크 복사.
        Raises:
            NotFittedError: 미적합 선택기.
        """
        check_is_fitted(self, "support_")
        return self.support_.copy()

    def transform(self, X):
        """새 자료의 선택 열을 원 좌표로 반환합니다.

        Args:
            X: 학습 열 순서의 수치 특성. 단일 행도 허용.
        Returns:
            np.ndarray: 선택 열 행렬.
        Raises:
            NotFittedError/ValueError: 미적합 또는 열 수/입력 오류.
        """
        mask = self.get_support()
        arr = validate_features(X, min_samples=1)
        if arr.shape[1] != len(mask):
            raise ValueError("학습 특성 수와 다릅니다.")
        return arr[:, mask]


def _run_model(
    X: Any, y: Any, profile: WrapperSelectionProfile
) -> WrapperSelectionOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        X: 검증된 수치형 특성 또는 원 열 이름을 가진 DataFrame.
        y: 검증된 클래스 라벨.
        profile: 검증된 WrapperSelectionProfile.
    Returns:
        WrapperSelectionOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    selector = WrapperSelector(
        task=profile.task,
        method=profile.method,
        n_features_to_select=profile.n_features_to_select,
        min_improvement=profile.min_improvement,
        max_steps=profile.max_steps,
        scaling=profile.scaling,
        cv=profile.cv,
        scoring=profile.scoring,
        random_state=profile.random_state,
        n_jobs=profile.n_jobs,
        C=profile.C,
        max_iter=profile.max_iter,
    )
    from dataclasses import replace
    from src.MachineLearning.common import run_supervised

    # 선택기의 내부 CV가 원 입력에서 스케일러를 폴드별 적합하도록 외부 선행 스케일러를 비활성화합니다.
    outer = replace(profile, scaling="none")
    result = run_supervised(
        X,
        y,
        _base_estimator(profile.task, profile.C, profile.max_iter),
        outer,
        extra_steps=[
            ("selector", selector),
            ("selected_scaler", make_scaler(profile.scaling)),
        ],
    )
    fitted = result.model.named_steps["selector"]
    indices = np.flatnonzero(fitted.get_support())
    arr = validate_features(X)
    names = feature_names(X)
    return WrapperSelectionOutputDTO(
        **result.to_dict(include_runtime=True),
        selector=fitted,
        selected_indices=indices,
        selected_features=[names[i] for i in indices],
        support_mask=fitted.get_support(),
        transformed_train=fitted.transform(arr[result.train_indices]),
        transformed_test=fitted.transform(arr[result.test_indices]),
        history=fitted.history_,
        selection_cv_score=fitted.best_score_,
        stopped_reason=fitted.stopped_reason_,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_wrapper_selection(
    X: Any,
    y: Any,
    profile: Optional[WrapperSelectionProfile] = None,
    *,
    task: Optional[str] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> WrapperSelectionOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        X: 입력 자료. 배열은 (n_samples, n_features), DataFrame은 열 순서를 보존.
        y: 길이 n_samples의 타깃. 분류는 이산 라벨, 회귀는 유한한 수치.
        profile: 명시적 WrapperSelectionProfile. None이면 wrapper-selection.json을 읽음.
        task: Profile 생략 시 선택할 classification/regression 목적.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        WrapperSelectionOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile(
        "wrapper-selection",
        profile,
        recipe_path=recipe_path,
        task=task,
        overrides=overrides,
    )
    _validate_inputs(X, y, profile)
    return _run_model(X, y, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    result = run_wrapper_selection(X, y, load_profile("wrapper-selection"))
    print(result.to_json(indent=2)[:500])
