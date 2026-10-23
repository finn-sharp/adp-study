"""CV 탐색 공통 구현. scorer는 항상 클수록 좋으며 보류 자료를 받지 않습니다."""

import numbers
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
)
from sklearn.utils import _safe_indexing


def _integer(value, name, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise ValueError(name + " 정수 범위 오류")


def _finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not np.isfinite(value)
    ):
        raise ValueError(name + " 유한한 수치가 필요합니다.")


def validate_inputs(X, y, estimator, profile, method, groups):
    """JSON 검색 공간·복제 가능 추정기·단일 타깃·CV 분할을 검증합니다."""
    shape = np.shape(X)
    target = np.asarray(y)
    if len(shape) != 2 or shape[0] < 3 or shape[1] < 1:
        raise ValueError("X는 3행 이상 1열 이상 2차원 자료여야 합니다.")
    if target.ndim != 1 or len(target) != shape[0] or pd.isna(target).any():
        raise ValueError("y는 X와 같은 행 수의 결측 없는 단일 타깃입니다.")
    if profile.task not in ("classification", "regression"):
        raise ValueError("task는 classification/regression입니다.")
    if profile.task == "regression":
        if not np.isfinite(target.astype(float)).all():
            raise ValueError("회귀 y는 유한해야 합니다.")
    elif pd.Series(target).nunique() < 2:
        raise ValueError("분류는 두 클래스 이상 필요합니다.")
    _integer(profile.cv, "cv", 2)
    _integer(profile.random_state, "random_state", 0)
    _integer(profile.n_iter, "n_iter")
    if (
        isinstance(profile.n_jobs, bool)
        or not isinstance(profile.n_jobs, numbers.Integral)
        or profile.n_jobs == 0
        or profile.n_jobs < -1
    ):
        raise ValueError("n_jobs는 -1 또는 양의 정수입니다.")
    for name in ("shuffle", "refit", "pruning"):
        if not isinstance(getattr(profile, name), bool):
            raise TypeError(name + "은 bool입니다.")
    if profile.split not in ("iid", "group", "time"):
        raise ValueError("split은 iid/group/time입니다.")
    if profile.split != "iid" and profile.shuffle:
        raise ValueError("time/group split에서는 shuffle=False가 필요합니다.")
    if profile.timeout is not None:
        _finite(profile.timeout, "timeout")
        if profile.timeout <= 0:
            raise ValueError("timeout은 양수입니다.")
    if method != "optuna_search" and (profile.timeout is not None or profile.pruning):
        raise ValueError("timeout/pruning은 Optuna에서만 지원합니다.")
    if method == "optuna_search" and profile.n_jobs != 1:
        raise ValueError("Optuna 재현성을 위해 n_jobs=1만 지원합니다.")
    clone(estimator)
    if not hasattr(estimator, "fit"):
        raise TypeError("fit 가능한 estimator가 필요합니다.")
    scorer = profile.scoring or (
        "accuracy" if profile.task == "classification" else "neg_mean_absolute_error"
    )
    if not isinstance(scorer, str):
        raise TypeError("scoring은 sklearn scorer 이름입니다.")
    get_scorer(scorer)
    space = profile.search_space
    if not isinstance(space, dict) or not space:
        raise ValueError("search_space에 하나 이상의 후보 파라미터가 필요합니다.")
    valid_params = estimator.get_params(deep=True)
    for key, spec in space.items():
        if key not in valid_params:
            raise ValueError("estimator에 없는 파라미터: " + str(key))
        if method != "optuna_search":
            if not isinstance(spec, (list, tuple)) or not spec:
                raise ValueError("grid/random 후보는 비어 있지 않은 목록입니다.")
            continue
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError("Optuna 후보는 type을 포함한 객체입니다.")
        kind = spec["type"]
        allowed = (
            {"type", "choices"}
            if kind == "categorical"
            else {"type", "low", "high", "step", "log"}
        )
        if set(spec) - allowed:
            raise ValueError("Optuna 후보에 알 수 없는 키가 있습니다.")
        if kind == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, (list, tuple)) or not choices:
                raise ValueError("categorical choices 목록이 필요합니다.")
            if not all(
                x is None or isinstance(x, (str, bool, int, float)) for x in choices
            ):
                raise ValueError("categorical choices는 JSON 스칼라입니다.")
            if any(isinstance(x, float) and not np.isfinite(x) for x in choices):
                raise ValueError("categorical choices는 유한해야 합니다.")
        elif kind in ("int", "float"):
            low, high = spec.get("low"), spec.get("high")
            _finite(low, "low")
            _finite(high, "high")
            if kind == "int" and (
                not isinstance(low, numbers.Integral)
                or not isinstance(high, numbers.Integral)
            ):
                raise ValueError("int low/high는 정수입니다.")
            if low > high:
                raise ValueError("low는 high 이하여야 합니다.")
            log = spec.get("log", False)
            if not isinstance(log, bool) or (log and low <= 0):
                raise ValueError("log는 bool이며 log 범위는 양수입니다.")
            step = spec.get("step", 1 if kind == "int" else None)
            if step is not None:
                _finite(step, "step")
                if step <= 0 or (
                    kind == "int" and not isinstance(step, numbers.Integral)
                ):
                    raise ValueError("step은 양수이며 int step은 정수입니다.")
                if log and (kind == "float" or step != 1):
                    raise ValueError(
                        "log+float step 또는 log+int step!=1은 불가합니다."
                    )
                count = (high - low) / step
                if not np.isclose(count, round(count), atol=1e-9, rtol=1e-9):
                    raise ValueError("high-low는 step의 배수여야 합니다.")
        else:
            raise ValueError("Optuna type은 categorical/int/float입니다.")
    if profile.split == "group":
        if groups is None or np.shape(groups) != (shape[0],) or pd.isna(groups).any():
            raise ValueError("group split에는 결측 없는 groups 벡터가 필요합니다.")
        if pd.Series(groups).nunique() < profile.cv:
            raise ValueError("그룹 수보다 cv가 큽니다.")
        cv = GroupKFold(n_splits=profile.cv)
    else:
        if groups is not None:
            raise ValueError("groups는 group split에서만 전달합니다.")
        if profile.split == "time":
            cv = TimeSeriesSplit(n_splits=profile.cv)
        else:
            if (
                profile.task == "classification"
                and pd.Series(target).value_counts().min() < profile.cv
            ):
                raise ValueError("각 클래스 표본 수가 cv 이상이어야 합니다.")
            cls = StratifiedKFold if profile.task == "classification" else KFold
            cv = cls(
                n_splits=profile.cv,
                shuffle=profile.shuffle,
                random_state=profile.random_state if profile.shuffle else None,
            )
    folds = list(cv.split(X, target, groups))
    for train, valid in folds:
        if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
            raise ValueError("비어 있거나 중첩된 CV 폴드입니다.")
    return folds, scorer


def _suggest(trial, space):
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        else:
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
                log=spec.get("log", False),
            )
    return params


def execute_search(X, y, estimator, profile, method, folds, scoring):
    """같은 폴드에서 점수를 최대화합니다. 원 estimator는 변경하지 않습니다."""
    start = time.perf_counter()
    study = None
    if method != "optuna_search":
        cls = GridSearchCV if method == "grid_search" else RandomizedSearchCV
        kwargs = (
            {"param_grid": profile.search_space}
            if method == "grid_search"
            else {
                "param_distributions": profile.search_space,
                "n_iter": profile.n_iter,
                "random_state": profile.random_state,
            }
        )
        search = cls(
            clone(estimator),
            scoring=scoring,
            cv=folds,
            n_jobs=profile.n_jobs,
            refit=False,
            error_score="raise",
            return_train_score=False,
            **kwargs,
        )
        search.fit(X, y)
        raw = search.cv_results_
        results = pd.DataFrame(
            {
                "params": raw["params"],
                "mean_score": raw["mean_test_score"],
                "std_score": raw["std_test_score"],
                "state": "COMPLETE",
            }
        )
        for i in range(len(folds)):
            results["split%d_score" % i] = raw["split%d_test_score" % i]
        if not np.isfinite(results.mean_score).all():
            raise ValueError("CV 점수가 비유한합니다.")
        best_index = int(np.argmax(results.mean_score.to_numpy()))
        best_params = dict(results.iloc[best_index].params)
        best_score = float(results.iloc[best_index].mean_score)
    else:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna는 별도 requirements-optimization 의존성을 설치하세요."
            ) from exc
        scorer = get_scorer(scoring)
        sampler = optuna.samplers.TPESampler(seed=profile.random_state)
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=5)
            if profile.pruning
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )

        def objective(trial):
            params = _suggest(trial, profile.search_space)
            scores = []
            for i, (train, valid) in enumerate(folds):
                model = clone(estimator).set_params(**params)
                model.fit(_safe_indexing(X, train), _safe_indexing(y, train))
                score = float(
                    scorer(model, _safe_indexing(X, valid), _safe_indexing(y, valid))
                )
                if not np.isfinite(score):
                    raise ValueError("Optuna CV 점수가 비유한합니다.")
                scores.append(score)
                trial.set_user_attr("fold_scores", scores.copy())
                trial.report(float(np.mean(scores)), step=i)
                if profile.pruning and trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study.optimize(
            objective, n_trials=profile.n_iter, timeout=profile.timeout, n_jobs=1
        )
        rows = []
        for trial in study.trials:
            scores = trial.user_attrs.get("fold_scores", [])
            complete = trial.state == optuna.trial.TrialState.COMPLETE
            row = {
                "trial": trial.number,
                "params": dict(trial.params),
                "state": trial.state.name,
                "mean_score": trial.value if complete else None,
                "std_score": float(np.std(scores)) if complete else None,
            }
            row.update({"split%d_score" % i: s for i, s in enumerate(scores)})
            rows.append(row)
        results = pd.DataFrame(rows)
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            raise ValueError("완료된 Optuna trial이 없습니다.")
        best_params, best_score = dict(study.best_params), float(study.best_value)
    model = clone(estimator).set_params(**best_params) if profile.refit else None
    if model is not None:
        model.fit(X, y)
    return {
        "method": method,
        "task": profile.task,
        "scoring": scoring,
        "best_params": best_params,
        "best_score": best_score,
        "cv_results": results,
        "folds": [
            {"train_indices": t.copy(), "validation_indices": v.copy()}
            for t, v in folds
        ],
        "n_candidates": len(results),
        "elapsed_seconds": time.perf_counter() - start,
        "model": model,
        "study": study,
    }
