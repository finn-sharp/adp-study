"""분포별 NB의 split/CV·원 라벨·수치 안정성 공통 계약; likelihood는 각 파일 소유."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    TimeSeriesSplit,
    GridSearchCV,
)
from sklearn.utils.multiclass import check_classification_targets
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.Preprocessing._common import feature_array
from src.MachineLearning.common import make_scaler, _metrics
from src.stats.regression._common import validate_profile


class FeatureSchemaGuard(TransformerMixin, BaseEstimator):
    """구버전 sklearn에서도 학습 Feature 이름/순서를 보존합니다."""

    def fit(self, X, y=None):
        """학습 스키마만 보존합니다.

        Args:
            X: 학습 DataFrame/행렬.
            y: 미사용 타깃.
        Returns:
            FeatureSchemaGuard: 자신.
        Raises:
            ValueError: 2차원/고유 열 오류.
        """
        if np.ndim(X) != 2:
            raise ValueError("2차원 Feature가 필요합니다.")
        self.n_features_in_ = np.shape(X)[1]
        self.columns_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        if self.columns_ is not None and len(set(self.columns_)) != len(self.columns_):
            raise ValueError("중복 Feature 열입니다.")
        return self

    def transform(self, X):
        """새 스키마를 확인하고 원 자료를 전달합니다.

        Args:
            X: 같은 원 열/순서의 새 자료.
        Returns:
            object: 변경하지 않은 원 자료.
        Raises:
            ValueError/NotFittedError: 미적합/열 오류.
        """
        check_is_fitted(self, "n_features_in_")
        if (
            np.ndim(X) != 2
            or np.shape(X)[1] != self.n_features_in_
            or (
                self.columns_ is not None
                and isinstance(X, pd.DataFrame)
                and X.columns.tolist() != self.columns_
            )
        ):
            raise ValueError("학습 Feature 이름/순서와 다릅니다.")
        return X


def numeric_counts(X, max_count=None):
    """분포의 원 정수 계수 정의역을 확인합니다.

    Args:
        X: 2차원 유한 계수.
        max_count: 선택 시행 수 상한.
    Returns:
        ndarray: 비음수 정수의 float 배열.
    Raises:
        ValueError: 음수·소수·시행 수 초과.
    """
    a, _ = feature_array(X, True, False)
    if (
        (a < 0).any()
        or (a != np.floor(a)).any()
        or (max_count is not None and (a > max_count).any())
    ):
        raise ValueError("분포에 맞는 비음수 정수/시행 수가 필요합니다.")
    return a


def labels_for_fit(y, n):
    """원 클래스의 정렬·빈도와 대응 길이를 확인합니다.

    Args:
        y: 이산 타깃.
        n: 행 수.
    Returns:
        tuple: classes, 정수 클래스 인덱스, 빈도.
    Raises:
        ValueError/TypeError: 타깃·클래스 오류.
    """
    a = np.asarray(y)
    if a.ndim != 1 or len(a) != n or pd.isna(a).any():
        raise ValueError("대응 클래스 라벨 오류")
    check_classification_targets(a)
    classes, index = np.unique(a, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("학습에 두 클래스 이상 필요합니다.")
    counts = np.bincount(index, minlength=len(classes)).astype(float)
    return classes, index, counts


def log_priors(counts, fit_prior):
    """학습 빈도 또는 균등 클래스 prior를 반환합니다.

    Args:
        counts: 양수 클래스 빈도.
        fit_prior: 경험 prior 학습 여부.
    Returns:
        ndarray: 클래스 log prior.
    Raises:
        ValueError: prior 설정 오류.
    """
    if not isinstance(fit_prior, bool):
        raise ValueError("fit_prior는 bool입니다.")
    return (
        np.log(counts / counts.sum())
        if fit_prior
        else np.full(len(counts), -np.log(len(counts)))
    )


class DistributionClassifierMixin:
    """분포별 likelihood에서 안정적인 확률/원 라벨 예측을 구성합니다."""

    def predict_log_proba(self, X):
        """클래스 log likelihood를 정규화합니다.

        Args:
            X: 새 분포 정의역 Feature.
        Returns:
            ndarray: classes_ 순서 log 확률.
        Raises:
            ValueError/NotFittedError: 정의역·미적합·비유한 점수.
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "classes_")
        score = self._joint_log_likelihood(X)
        if not np.isfinite(score).all():
            raise ValueError("likelihood 수치 overflow")
        return score - logsumexp(score, axis=1, keepdims=True)

    def predict_proba(self, X):
        """원 클래스 순서의 확률을 반환합니다.

        Args:
            X: 새 Feature.
        Returns:
            ndarray: 정규화 확률.
        Raises:
            ValueError/NotFittedError: 정의역 오류.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """최대 posterior 원 클래스 라벨을 반환합니다.

        Args:
            X: 새 Feature.
        Returns:
            ndarray: 원 라벨.
        Raises:
            ValueError/NotFittedError: 정의역 오류.
        """
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


def validate_nb_profile(profile, cls, numeric=True):
    """분할·CV·분포별 스케일 계약을 확인합니다.

    Args:
        profile: 전용 NB Profile.
        cls: 전용 클래스.
        numeric: categorical 외 수치 모델 여부.
    Returns:
        None: 유효 설정.
    Raises:
        ValueError/TypeError: 범위·스케일·CV 오류.
    """
    validate_profile(profile, cls, check_alpha=False)
    if (
        not 0 < profile.test_size < 1
        or profile.cv < 2
        or profile.n_jobs == 0
        or profile.split_strategy not in ("random", "chronological")
        or not isinstance(profile.stratify, bool)
    ):
        raise ValueError("분할/CV 설정 오류")
    if profile.scaling not in ("none", "standard", "minmax") or (
        cls.__name__ != "GaussianProfile" and profile.scaling != "none"
    ):
        raise ValueError("빈도/범주/이산 분포에는 scaling=none을 사용하세요.")
    if profile.param_grid is not None and (
        not isinstance(profile.param_grid, dict) or not profile.param_grid
    ):
        raise ValueError("param_grid 오류")
    for key, values in (profile.param_grid or {}).items():
        field = key.split("__")[-1]
        if field in ("alpha", "beta", "rate_alpha", "rate_beta", "var_smoothing"):
            if not isinstance(values, (list, tuple, np.ndarray)) or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not np.isfinite(v)
                or v <= 0
                for v in values
            ):
                raise ValueError("CV의 " + field + "도 유한 양수여야 합니다.")


def run_distribution(
    X, y, estimator, profile, cls, categorical=False, test_indices=None
):
    """미적합 분포별 모델을 train/CV 내부에서만 적합합니다.

    Args:
        X: 원 Feature; categorical은 원 범주/결측 허용.
        y: 대응 분류 타깃.
        estimator: 해당 파일의 미적합 모델.
        profile: 전용 분할/학습 Profile.
        cls: 전용 Output DTO.
        categorical: 원 범주 여부.
        test_indices: 선택 명시 보류 원 행 위치; 공식 분할 유지에 사용.
    Returns:
        OutputDTO: 원 라벨·보류 확률·모델·분할·클래스 학습 수.
    Raises:
        ValueError/TypeError: 자료·고정 분할·클래스/CV 오류.
    """
    a, names = feature_array(X, not categorical, categorical)
    raw = pd.DataFrame(
        a, index=X.index if isinstance(X, pd.DataFrame) else None, columns=names
    )
    if (
        isinstance(X, pd.DataFrame)
        and isinstance(y, pd.Series)
        and not X.index.equals(y.index)
    ):
        raise ValueError("Feature·타깃 인덱스 불일치")
    target = np.asarray(y)
    classes, _, _ = labels_for_fit(target, len(raw))
    positions = np.arange(len(raw))
    if test_indices is not None:
        test = np.asarray(test_indices)
        if (
            test.ndim != 1
            or test.dtype.kind not in "iu"
            or len(test) < 1
            or len(np.unique(test)) != len(test)
            or (test < 0).any()
            or (test >= len(raw)).any()
            or len(test) >= len(raw)
        ):
            raise ValueError("고정 보류 위치 오류")
        train = np.setdiff1d(positions, test)
        test = np.sort(test)
    elif profile.split_strategy == "chronological":
        cut = int(len(raw) * (1 - profile.test_size))
        train, test = positions[:cut], positions[cut:]
    else:
        train, test = train_test_split(
            positions,
            test_size=profile.test_size,
            random_state=profile.random_state,
            stratify=target if profile.stratify else None,
        )
        train = np.sort(train)
        test = np.sort(test)
    labels_for_fit(target[train], len(train))
    if not np.isin(target[test], np.unique(target[train])).all():
        raise ValueError("보류에 미학습 클래스가 있습니다.")
    pipeline = Pipeline(
        [
            ("schema", FeatureSchemaGuard()),
            ("scale", make_scaler(profile.scaling)),
            ("model", estimator),
        ]
    )
    best_params = {}
    best_score = None
    cv_results = None
    if profile.param_grid is not None:
        if profile.split_strategy == "chronological":
            cv = TimeSeriesSplit(profile.cv)
        else:
            _, _, counts = labels_for_fit(target[train], len(train))
            if counts.min() < profile.cv:
                raise ValueError("CV 클래스 수 부족")
            cv = StratifiedKFold(
                profile.cv, shuffle=True, random_state=profile.random_state
            )
        grid = {
            key if "__" in key else "model__" + key: value
            for key, value in profile.param_grid.items()
        }
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv,
            scoring=profile.scoring or "accuracy",
            n_jobs=profile.n_jobs,
            error_score="raise",
        ).fit(raw.iloc[train], target[train])
        pipeline = search.best_estimator_
        best_params = search.best_params_
        best_score = float(search.best_score_)
        cv_results = search.cv_results_
    else:
        pipeline.fit(raw.iloc[train], target[train])
    fitted = pipeline.named_steps["model"]
    prediction = pipeline.predict(raw.iloc[test])
    probabilities = pipeline.predict_proba(raw.iloc[test])
    test_metrics = _metrics(target[test], prediction, "classification")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0
    ):
        raise ValueError("분포별 posterior 확률이 유효하지 않습니다.")
    from sklearn.metrics import roc_auc_score, confusion_matrix

    if len(fitted.classes_) == 2 and len(np.unique(target[test])) == 2:
        test_metrics["roc_auc"] = float(
            roc_auc_score(target[test] == fitted.classes_[1], probabilities[:, 1])
        )
    return cls(
        type(fitted).__name__,
        pipeline,
        _metrics(target[train], pipeline.predict(raw.iloc[train]), "classification"),
        test_metrics,
        target[test],
        prediction,
        probabilities,
        fitted.classes_.copy(),
        fitted.class_count_.copy(),
        confusion_matrix(target[test], prediction, labels=fitted.classes_),
        train,
        test,
        names,
        best_params,
        best_score,
        cv_results,
        [
            "조건부 Feature 독립과 해당 분포/노출 단위를 확인하세요. posterior 확률의 calibration은 자동 보장하지 않습니다."
        ],
    )
