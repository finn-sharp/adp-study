"""독립 평가의 입력 검증·계산. Python 3.7/sklearn 1.0.2 공통 API만 사용합니다."""

from dataclasses import asdict
import numpy as np
import pandas as pd
from sklearn import metrics as sk


class UndefinedMetric(ValueError):
    """유효 입력이지만 해당 지표의 수학적 정의역이 충족되지 않습니다."""


CLASSIFICATION = (
    "accuracy precision recall f1 fbeta specificity balanced_accuracy mcc cohen_kappa confusion_matrix roc_auc average_precision pr_auc log_loss brier_score gini ks lift gain lift_curve gain_curve roc_curve precision_recall_curve calibration"
).split()
REGRESSION = (
    "mae mse rmse r2 adjusted_r2 median_absolute_error mape smape wape msle rmsle explained_variance max_error pinball_loss"
).split()
CLUSTERING = (
    "silhouette davies_bouldin calinski_harabasz ari ami nmi homogeneity completeness v_measure"
).split()
TIME_SERIES = ["mase", "rmsse", "horizon_errors"]
LOWER_IS_BETTER = set(
    "mae mse rmse median_absolute_error mape smape wape msle rmsle max_error pinball_loss log_loss brier_score calibration davies_bouldin mase rmsse".split()
)


def validate_profile(p, cls, group):
    """설정의 타입과 정의역을 검증합니다.

    Args:
        p: 평가 Profile.
        cls: 전용 Profile 타입.
        group: 평가 분야.
    Returns:
        None: 유효한 설정.
    Raises:
        TypeError/ValueError: 설정 오류.
    """
    if not isinstance(p, cls):
        raise TypeError("전용 Profile이 필요합니다.")

    def integer(value, minimum):
        return (
            isinstance(value, int) and not isinstance(value, bool) and value >= minimum
        )

    if group == "classification":
        if p.average not in ("binary", "micro", "macro", "weighted", None):
            raise ValueError("average: binary/micro/macro/weighted/None")
        if not integer(p.zero_division, 0) or p.zero_division not in (0, 1):
            raise ValueError("zero_division: 0/1")
        if (
            isinstance(p.beta, bool)
            or not np.isscalar(p.beta)
            or not np.isfinite(p.beta)
            or p.beta <= 0
        ):
            raise ValueError("beta는 유한 양수입니다.")
        if not integer(p.n_bins, 1):
            raise ValueError("n_bins는 양의 정수입니다.")
        f = np.asarray(p.top_fractions, dtype=float)
        if (
            f.ndim != 1
            or not len(f)
            or not np.all(np.isfinite(f))
            or np.any((f <= 0) | (f > 1))
        ):
            raise ValueError("top_fractions는 (0,1] 비율 목록입니다.")
        if p.labels is not None:
            labels(p.labels, "labels")
            if len(set(np.asarray(p.labels).tolist())) != len(p.labels):
                raise ValueError("labels 중복")
    elif group == "regression":
        if p.multioutput not in (
            "uniform_average",
            "raw_values",
        ) or p.zero_policy not in ("raise", "omit"):
            raise ValueError("multioutput/zero_policy 오류")
        if not integer(p.n_features, 0):
            raise ValueError("n_features는 0 이상 정수입니다.")
        if (
            isinstance(p.quantile, bool)
            or not np.isscalar(p.quantile)
            or not np.isfinite(p.quantile)
            or not 0 < p.quantile < 1
        ):
            raise ValueError("quantile은 (0,1)입니다.")
    elif group == "clustering":
        if not isinstance(p.exclude_noise, bool) or p.distance not in (
            "euclidean",
            "manhattan",
            "cosine",
        ):
            raise ValueError("exclude_noise/distance 오류")
    elif group == "time_series" and not integer(p.seasonal_period, 1):
        raise ValueError("seasonal_period는 양의 정수입니다.")


def numeric(x, name):
    """유한하고 비어 있지 않은 수치 배열을 반환합니다."""
    if x is None:
        raise ValueError(name + "이 필요합니다.")
    a = np.asarray(x, dtype=float)
    if a.ndim == 0 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError(name + "은 비어 있지 않은 유한 수치 배열입니다.")
    return a


def labels(x, name):
    """결측 없는 1D 단일 종류 라벨을 반환합니다."""
    if x is None:
        raise ValueError(name + "이 필요합니다.")
    a = np.asarray(x)
    if a.ndim != 1 or not len(a) or np.any(pd.isna(a)):
        raise ValueError(name + "은 결측 없는 비어 있지 않은 1D 라벨입니다.")
    if a.dtype.kind in "fc" and (
        not np.all(np.isfinite(a)) or np.any(a != np.floor(a))
    ):
        raise ValueError(name + "은 이산 라벨입니다.")
    return a


def paired_numeric(y_true, y_pred):
    """정답/예측의 동일 shape를 검증합니다."""
    a, b = numeric(y_true, "y_true"), numeric(y_pred, "y_pred")
    if a.shape != b.shape:
        raise ValueError("정답과 예측 shape가 같아야 합니다.")
    return a, b


def class_inputs(y_true, y_pred, y_score, p):
    """확률 열 순서를 인코딩하여 sklearn의 정렬 의존성을 제거합니다."""
    y = labels(y_true, "y_true")
    pred = None if y_pred is None else labels(y_pred, "y_pred")
    if pred is not None and len(pred) != len(y):
        raise ValueError("정답/예측 행 수 불일치")
    if p.labels is None:
        order = np.unique(np.concatenate((y, pred)) if pred is not None else y)
    else:
        order = np.asarray(p.labels)
    if not np.all(np.isin(y, order)) or (
        pred is not None and not np.all(np.isin(pred, order))
    ):
        raise ValueError("labels에 모든 정답/예측 클래스가 있어야 합니다.")
    score = None if y_score is None else numeric(y_score, "y_score")
    if score is not None:
        if score.ndim not in (1, 2) or len(score) != len(y):
            raise ValueError("y_score은 같은 행 수의 1D 점수 또는 2D 확률입니다.")
        if score.ndim == 2:
            if p.labels is None or score.shape[1] != len(order):
                raise ValueError("확률 행렬에는 열 순서 labels를 명시해야 합니다.")
            # Validate in the precision of the supplied softmax output, before
            # numeric() promotes float32 to float64. Four ULPs permit rounding,
            # while the cap prevents low-precision inputs accepting bad totals.
            source_dtype = np.asarray(y_score).dtype
            source_epsilon = (
                np.finfo(source_dtype).eps if source_dtype.kind == "f" else 0.0
            )
            tolerance = max(1e-7, min(1e-6, 4 * source_epsilon))
            row_sums = score.sum(axis=1, keepdims=True)
            if np.any((score < 0) | (score > 1)) or not np.allclose(
                row_sums, 1, atol=tolerance, rtol=0
            ):
                raise ValueError("확률은 [0,1]이며 각 행의 합이 1이어야 합니다.")
            # Work on a new array; only accepted rounding is normalized.
            score = score / row_sums
        elif len(order) > 2:
            raise ValueError("다중 클래스에는 확률 행렬을 전달하세요.")
    encoded = np.array([np.flatnonzero(order == v)[0] for v in y], dtype=int)
    return y, pred, score, order, encoded


def binary_inputs(y, score, order, p):
    """지정 양성 클래스의 1D 점수를 반환합니다."""
    if len(order) != 2 or p.positive_label not in order:
        raise UndefinedMetric(
            "이 메트릭에는 두 클래스와 올바른 positive_label이 필요합니다."
        )
    if score is None:
        raise ValueError("y_score이 필요합니다.")
    s = (
        score
        if score.ndim == 1
        else score[:, np.flatnonzero(order == p.positive_label)[0]]
    )
    b = (y == p.positive_label).astype(int)
    if len(np.unique(b)) != 2:
        raise UndefinedMetric("실제 양성·음성 모두 있어야 합니다.")
    return b, s


def classification(name, data, p, details):
    """단일 분류/확률/순위 메트릭 계산."""
    y, pred, score, order, encoded = data
    if name in CLASSIFICATION[:10]:
        if pred is None:
            raise ValueError("y_pred가 필요합니다.")
        args = dict(labels=order, average=p.average, zero_division=p.zero_division)
        if p.average == "binary":
            if len(order) != 2 or p.positive_label not in order:
                raise UndefinedMetric(
                    "binary 집계에는 두 클래스와 positive_label이 필요합니다."
                )
            args["pos_label"] = p.positive_label
        if name == "accuracy":
            return float(sk.accuracy_score(y, pred))
        if name == "balanced_accuracy":
            return float(sk.balanced_accuracy_score(y, pred))
        if name == "mcc":
            return float(sk.matthews_corrcoef(y, pred))
        if name == "cohen_kappa":
            if len(np.unique(np.concatenate((y, pred)))) < 2:
                raise UndefinedMetric("Kappa: 한 클래스만 존재합니다.")
            return float(sk.cohen_kappa_score(y, pred, labels=order))
        if name == "confusion_matrix":
            matrix = sk.confusion_matrix(y, pred, labels=order)
            details["matrix"] = pd.DataFrame(matrix, index=order, columns=order)
            return matrix
        if name in ("precision", "recall", "f1", "fbeta"):
            fn = {
                "precision": sk.precision_score,
                "recall": sk.recall_score,
                "f1": sk.f1_score,
                "fbeta": sk.fbeta_score,
            }[name]
            if name == "fbeta":
                args["beta"] = p.beta
            return fn(y, pred, **args)
        matrix = sk.confusion_matrix(y, pred, labels=order)
        tp = matrix.diagonal()
        fn = matrix.sum(axis=1) - tp
        fp = matrix.sum(axis=0) - tp
        tn = matrix.sum() - tp - fn - fp
        denom = tn + fp
        values = np.divide(
            tn, denom, out=np.full(len(order), float(p.zero_division)), where=denom != 0
        )
        if p.average == "binary":
            return float(values[np.flatnonzero(order == p.positive_label)[0]])
        if p.average is None:
            return values
        if p.average == "micro":
            return (
                float(tn.sum() / denom.sum()) if denom.sum() else float(p.zero_division)
            )
        return float(
            np.average(
                values, weights=matrix.sum(axis=1) if p.average == "weighted" else None
            )
        )
    if score is None:
        raise ValueError("y_score이 필요합니다.")
    if name in ("log_loss", "brier_score"):
        if score.ndim == 1:
            if (
                len(order) != 2
                or p.positive_label not in order
                or np.any((score < 0) | (score > 1))
            ):
                raise ValueError("확률 점수와 두 클래스·positive_label이 필요합니다.")
            positive = np.flatnonzero(order == p.positive_label)[0]
            probabilities = np.empty((len(y), 2))
            probabilities[:, positive] = score
            probabilities[:, 1 - positive] = 1 - score
        else:
            probabilities = score
        if name == "log_loss":
            if len(order) < 2:
                raise UndefinedMetric("Log loss는 두 개 이상 클래스가 필요합니다.")
            # Explicit clipping fixed across sklearn versions.
            q = np.clip(probabilities, 1e-15, 1 - 1e-15)
            q /= q.sum(axis=1, keepdims=True)
            return float(-np.log(q[np.arange(len(y)), encoded]).mean())
        target = np.eye(len(order))[encoded]
        if len(order) == 2:
            positive = np.flatnonzero(order == p.positive_label)
            if not len(positive):
                raise ValueError("positive_label 오류")
            return float(
                np.mean((probabilities[:, positive[0]] - (y == p.positive_label)) ** 2)
            )
        details["definition"] = (
            "다중 클래스 Brier = mean(sum((p-onehot)^2)); 범위 [0,2]"
        )
        return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))
    if name in ("roc_auc", "average_precision") and len(order) > 2:
        if score.ndim != 2:
            raise ValueError("다중 클래스 확률 행렬 필요")
        if len(np.unique(encoded)) != len(order):
            raise UndefinedMetric(
                "다중 클래스 확률 지표는 모든 클래스가 평가 자료에 있어야 합니다."
            )
        if p.average not in ("macro", "weighted", None):
            raise ValueError("다중 클래스 확률 집계는 macro/weighted/None입니다.")
        if name == "roc_auc":
            return sk.roc_auc_score(
                encoded,
                score,
                labels=np.arange(len(order)),
                multi_class="ovr",
                average=p.average,
            )
        return sk.average_precision_score(
            np.eye(len(order))[encoded], score, average=p.average
        )
    b, s = binary_inputs(y, score, order, p)
    if name in ("roc_auc", "gini"):
        return float(sk.roc_auc_score(b, s)) * (2 if name == "gini" else 1) - (
            1 if name == "gini" else 0
        )
    if name == "average_precision":
        return float(sk.average_precision_score(b, s))
    if name in ("pr_auc", "precision_recall_curve"):
        precision, recall, thresholds = sk.precision_recall_curve(b, s)
        details["curve"] = pd.DataFrame(
            {
                "recall": recall,
                "precision": precision,
                "threshold": np.r_[thresholds, np.nan],
            }
        )
        details["definition"] = "사다리꼴 PR-AUC; Average Precision과 별도"
        return float(sk.auc(recall, precision)) if name == "pr_auc" else precision
    if name in ("ks", "roc_curve"):
        fpr, tpr, thresholds = sk.roc_curve(b, s, drop_intermediate=False)
        details["curve"] = pd.DataFrame(
            {"fpr": fpr, "tpr": tpr, "threshold": thresholds}
        )
        if name == "ks":
            differences = tpr - fpr
            i = int(np.argmax(np.abs(differences)))
            details["threshold"] = float(thresholds[i])
            details["signed_difference"] = float(differences[i])
            return float(abs(differences[i]))
        return tpr
    if name == "calibration":
        if np.any((s < 0) | (s > 1)):
            raise ValueError("calibration은 [0,1] 확률 필요")
        bins = np.minimum((s * p.n_bins).astype(int), p.n_bins - 1)
        rows = []
        ece = 0.0
        for i in range(p.n_bins):
            mask = bins == i
            if not mask.any():
                continue
            forecast, actual = float(s[mask].mean()), float(b[mask].mean())
            rows.append((i, int(mask.sum()), forecast, actual))
            ece += mask.mean() * abs(forecast - actual)
        details["bins"] = pd.DataFrame(
            rows, columns=["bin", "count", "mean_probability", "positive_rate"]
        )
        details["definition"] = "ECE 동일 너비 bin; 빈 bin 제외"
        return float(ece)
    if name in ("lift", "gain", "lift_curve", "gain_curve"):
        idx = np.argsort(-s, kind="mergesort")
        positives = np.cumsum(b[idx])
        counts = np.arange(1, len(b) + 1)
        gain = positives / b.sum()
        fractions = counts / len(b)
        lift = gain / fractions
        details["tie_policy"] = "동률은 원 입력 행 순서를 유지; top-k는 ceil(n*f)"
        details["baseline_rate"] = float(b.mean())
        details["curve"] = pd.DataFrame(
            {
                "count": np.r_[0, counts],
                "fraction": np.r_[0, fractions],
                "gain": np.r_[0, gain],
                "lift": np.r_[np.nan, lift],
            }
        )
        if name.endswith("_curve"):
            return lift if name == "lift_curve" else gain
        k = np.ceil(np.asarray(p.top_fractions) * len(b)).astype(int)
        details["top"] = pd.DataFrame(
            {
                "requested_fraction": p.top_fractions,
                "count": k,
                "actual_fraction": k / len(b),
                "positives": positives[k - 1],
                "gain": gain[k - 1],
                "lift": lift[k - 1],
            }
        )
        return lift[k - 1] if name == "lift" else gain[k - 1]
    raise ValueError("알 수 없는 분류 메트릭: " + name)


def regression(name, a, b, p, details):
    """다중 타깃 회귀 메트릭; 백분율은 0~100 단위입니다."""
    if a.ndim not in (1, 2):
        raise ValueError("회귀는 (n,) 또는 (n,targets)입니다.")
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    error = b - a
    aggregate = lambda v: (
        np.asarray(v) if p.multioutput == "raw_values" else float(np.mean(v))
    )
    if name == "mae":
        values = np.abs(error).mean(axis=0)
    elif name == "mse":
        values = (error**2).mean(axis=0)
    elif name == "rmse":
        values = np.sqrt((error**2).mean(axis=0))
    elif name == "median_absolute_error":
        values = np.median(np.abs(error), axis=0)
    elif name == "max_error":
        values = np.max(np.abs(error), axis=0)
    elif name in ("r2", "adjusted_r2", "explained_variance"):
        denom = ((a - a.mean(axis=0)) ** 2).sum(axis=0)
        if len(a) < 2 or np.any(denom == 0):
            raise UndefinedMetric("R²/설명분산: n>=2 및 비상수 타깃이 필요합니다.")
        values = (
            1 - ((error - error.mean(axis=0)) ** 2).sum(axis=0) / denom
            if name == "explained_variance"
            else 1 - (error**2).sum(axis=0) / denom
        )
        if name == "adjusted_r2":
            if len(a) <= p.n_features + 1:
                raise UndefinedMetric("Adjusted R²: n > p+1 필요")
            values = 1 - (1 - values) * (len(a) - 1) / (len(a) - p.n_features - 1)
    elif name == "mape":
        mask = a != 0
        if not mask.all() and p.zero_policy == "raise":
            raise UndefinedMetric("MAPE 실제값 0; zero_policy=omit 가능")
        if np.any(mask.sum(axis=0) == 0):
            raise UndefinedMetric("MAPE 유효 실제값이 없는 타깃")
        ratio = np.divide(np.abs(error), np.abs(a), out=np.zeros_like(a), where=mask)
        values = 100 * ratio.sum(axis=0) / mask.sum(axis=0)
        details["omitted_per_target"] = (~mask).sum(axis=0)
    elif name == "smape":
        denom = np.abs(a) + np.abs(b)
        values = 200 * np.divide(
            np.abs(error), denom, out=np.zeros_like(a), where=denom != 0
        ).mean(axis=0)
        details["definition"] = (
            "0/0=0; 200*mean(abs(error)/(abs(actual)+abs(predicted)))"
        )
    elif name == "wape":
        denom = np.abs(a).sum(axis=0)
        if np.any(denom == 0):
            raise UndefinedMetric("WAPE 실제값 절대합 0")
        values = 100 * np.abs(error).sum(axis=0) / denom
    elif name in ("msle", "rmsle"):
        if np.any(a < 0) or np.any(b < 0):
            raise UndefinedMetric("MSLE/RMSLE은 비음수 실제값·예측값 필요")
        values = ((np.log1p(a) - np.log1p(b)) ** 2).mean(axis=0)
        if name == "rmsle":
            values = np.sqrt(values)
    elif name == "pinball_loss":
        delta = a - b
        values = np.maximum(p.quantile * delta, (p.quantile - 1) * delta).mean(axis=0)
    else:
        raise ValueError("알 수 없는 회귀 메트릭: " + name)
    details["per_target"] = values
    return aggregate(values)


def cluster_inputs(y_true, y_pred, X, p):
    """군집 라벨·좌표를 같은 마스크로 처리합니다."""
    pred = labels(y_pred, "y_pred")
    truth = None if y_true is None else labels(y_true, "y_true")
    coords = None if X is None else numeric(X, "X")
    if truth is not None and len(truth) != len(pred):
        raise ValueError("군집 정답 행 수 불일치")
    if coords is not None and (coords.ndim != 2 or len(coords) != len(pred)):
        raise ValueError("X는 같은 행 수의 2D 좌표입니다.")
    keep = (
        np.array([v != p.noise_label for v in pred], dtype=bool)
        if p.exclude_noise
        else np.ones(len(pred), dtype=bool)
    )
    return (
        (None if truth is None else truth[keep]),
        pred[keep],
        (None if coords is None else coords[keep]),
        keep,
    )


def clustering(name, data, p, details):
    """내부 군집 지표는 적합 좌표, 외부 지표는 참조 라벨을 사용합니다."""
    truth, pred, X, keep = data
    details["excluded_rows"] = int((~keep).sum())
    details["evaluation_indices"] = np.flatnonzero(keep)
    if not len(pred):
        raise UndefinedMetric("잡음 제외 후 평가 행 없음")
    if name in CLUSTERING[:3]:
        if X is None:
            raise ValueError("내부 군집 지표에 X가 필요합니다.")
        if not 2 <= len(np.unique(pred)) < len(pred):
            raise UndefinedMetric("내부 군집 지표는 2 <= 군집 수 < 행 수 필요")
        if name == "silhouette":
            return float(sk.silhouette_score(X, pred, metric=p.distance))
        if name == "davies_bouldin":
            return float(sk.davies_bouldin_score(X, pred))
        return float(sk.calinski_harabasz_score(X, pred))
    if truth is None:
        raise ValueError("외부 군집 지표에 y_true가 필요합니다.")
    fn = {
        "ari": sk.adjusted_rand_score,
        "ami": sk.adjusted_mutual_info_score,
        "nmi": sk.normalized_mutual_info_score,
        "homogeneity": sk.homogeneity_score,
        "completeness": sk.completeness_score,
        "v_measure": sk.v_measure_score,
    }[name]
    return float(fn(truth, pred))


def time_series(name, a, b, train, p, details):
    """순서/스텝을 보존하는 예측 평가; 학습 naive 오차로만 정규화합니다."""
    if a.ndim not in (1, 2, 3):
        raise ValueError("시계열 shape: (n,), (창,horizon), (창,horizon,target)")
    if name == "horizon_errors":
        arr = (b - a).reshape(len(a), a.shape[1] if a.ndim > 1 else 1, -1)
        mae = np.mean(np.abs(arr), axis=0)
        rmse = np.sqrt(np.mean(arr**2, axis=0))
        details["mae_by_horizon_target"] = mae
        details["rmse_by_horizon_target"] = rmse
        details["table"] = pd.DataFrame(
            [
                (h + 1, t, float(mae[h, t]), float(rmse[h, t]))
                for h in range(mae.shape[0])
                for t in range(mae.shape[1])
            ],
            columns=["horizon", "target", "mae", "rmse"],
        )
        return {
            "mae": float(np.mean(np.abs(arr))),
            "rmse": float(np.sqrt(np.mean(arr**2))),
        }
    training = numeric(train, "y_train")
    if training.ndim not in (1, 2) or len(training) <= p.seasonal_period:
        raise ValueError("y_train은 lag보다 긴 (시간,) 또는 (시간,타깃)입니다.")
    training = training.reshape(len(training), -1)
    # 2D evaluation is (window,horizon), never a targets matrix.
    err = (b - a).reshape(-1, a.shape[-1] if a.ndim == 3 else 1)
    if training.shape[1] != err.shape[1]:
        raise ValueError("학습/평가 타깃 수 불일치")
    diff = training[p.seasonal_period :] - training[: -p.seasonal_period]
    scale = (
        np.mean(np.abs(diff), axis=0) if name == "mase" else np.mean(diff**2, axis=0)
    )
    if np.any(scale == 0):
        raise UndefinedMetric("학습 naive 오차 0; MASE/RMSSE 정의 불가")
    values = (
        np.mean(np.abs(err), axis=0) / scale
        if name == "mase"
        else np.sqrt(np.mean(err**2, axis=0) / scale)
    )
    details["training_scale"] = scale
    details["per_target"] = values
    return float(values.mean())


def evaluate(group, name, y_true, y_pred, y_score, X, y_train, p):
    """공통 입력을 먼저 검증하고 단일/묶음 결과를 생성합니다.

    Args:
        group: classification/regression/clustering/time_series.
        name: 개별 메트릭명 또는 metrics.
        y_true: 원 행 순서 정답.
        y_pred: 대응 예측 또는 군집 라벨.
        y_score: 양성 점수 또는 명시 labels 순서 확률.
        X: 동일 전처리 군집 좌표.
        y_train: 학습 시계열.
        p: 검증한 전용 Profile.
    Returns:
        dict: metric_name/value/n_samples/details/warnings.
    Raises:
        ValueError/UndefinedMetric: 입력 오류 또는 정의 불가 단일 지표.
    """
    details = {"parameters": asdict(p)}
    warnings = []
    if group == "classification":
        data = class_inputs(y_true, y_pred, y_score, p)
        n = len(data[0])
        details["class_order"] = data[3]
        names = CLASSIFICATION[:] if name == "metrics" else [name]
        if name == "metrics":
            # Keep undefined math as explicit null; omit only unavailable input families.
            names = [
                v
                for v in names
                if (v not in CLASSIFICATION[:10] or y_pred is not None)
                and (v in CLASSIFICATION[:10] or y_score is not None)
            ]
            if len(data[3]) > 2:
                names = [
                    v
                    for v in names
                    if v
                    in CLASSIFICATION[:10]
                    + ["roc_auc", "average_precision", "log_loss", "brier_score"]
                ]
        calculator = lambda v, d: classification(v, data, p, d)
    elif group == "regression":
        a, b = paired_numeric(y_true, y_pred)
        n = len(a)
        names = REGRESSION if name == "metrics" else [name]
        calculator = lambda v, d: regression(v, a, b, p, d)
    elif group == "clustering":
        data = cluster_inputs(y_true, y_pred, X, p)
        n = len(data[1])
        names = CLUSTERING if name == "metrics" else [name]
        if name == "metrics":
            names = [
                v
                for v in names
                if (v not in CLUSTERING[:3] or X is not None)
                and (v in CLUSTERING[:3] or y_true is not None)
            ]
        calculator = lambda v, d: clustering(v, data, p, d)
    elif group == "time_series":
        a, b = paired_numeric(y_true, y_pred)
        n = len(a)
        names = TIME_SERIES if name == "metrics" else [name]
        if name == "metrics" and y_train is None:
            names = ["horizon_errors"]
        calculator = lambda v, d: time_series(v, a, b, y_train, p, d)
    else:
        raise ValueError("알 수 없는 평가 분야")
    if not names:
        raise ValueError("평가에 필요한 예측/점수/좌표/정답이 없습니다.")
    values = {}
    per_metric = {}
    for metric in names:
        info = {}
        try:
            values[metric] = calculator(metric, info)
        except UndefinedMetric as exc:
            if name != "metrics":
                raise
            values[metric] = float("nan")
            warnings.append(metric + ": " + str(exc))
        per_metric[metric] = info
    if name == "metrics":
        details["metrics"] = per_metric
        value = values
    else:
        details.update(per_metric[name])
        value = values[name]
    return dict(
        metric_name=group + "-" + name,
        value=value,
        n_samples=n,
        details=details,
        warnings=warnings,
    )


def example_inputs(group):
    """다운로드 없이 모든 개별 메트릭을 실행하는 작은 예제를 반환합니다."""
    if group == "classification":
        return dict(
            y_true=[0, 0, 1, 1, 0, 1],
            y_pred=[0, 1, 1, 1, 0, 0],
            y_score=[0.1, 0.6, 0.8, 0.9, 0.2, 0.4],
        )
    if group == "regression":
        return dict(y_true=[1.0, 2.0, 4.0, 8.0], y_pred=[1.2, 1.8, 3.0, 7.0])
    if group == "clustering":
        return dict(
            y_true=[0, 0, 1, 1], y_pred=[0, 0, 1, 1], X=[[0, 0], [0, 1], [5, 5], [5, 6]]
        )
    return dict(
        y_true=[[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
        y_pred=[[3.1, 4.2], [4.2, 5.1], [5.2, 6.2]],
        y_train=[1.0, 2.0, 3.0],
    )
