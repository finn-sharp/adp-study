"""회귀 workflow 내부 단계. 입력/분할/후보 적합을 분리하고 최종 test는 선택에서 제외합니다."""

import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    TimeSeriesSplit,
    cross_val_score,
)
from ._regression_estimator import RegressionEstimator


def regression_metrics(truth, prediction):
    """같은 원 반응 단위의 정답·예측에서 RMSE·MAE·R²를 계산합니다.

    Args:
        truth, prediction: 같은 순서·길이·반응 단위의 수치 배열.
    Returns:
        dict: rmse·mae·r2 스칼라.
    Raises:
        ValueError: 비유한 값·행 수 등 평가 입력 오류.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)),
    }


def prepare_data(X, y, profile):
    """원 Feature/타깃 대응과 결측 정책을 검증하고 원행 위치를 보존합니다.

    Args:
        X, y: 원 Feature와 대응 반응. profile: 결측·CV 설정.
    Returns:
        tuple: raw DataFrame·target 배열·결측 마스크·유효 원행 위치.
    Raises:
        ValueError/TypeError: 차원·인덱스·반응 값·최소 표본 수 위반.
    """
    if not isinstance(X, pd.DataFrame) and np.ndim(X) != 2:
        raise ValueError("2차원 원 Feature가 필요합니다.")
        # 후보별 옵션이 공통 추론 설정을 재정의합니다.
        options = {
            "alpha": profile.alpha,
            "cov_type": profile.cov_type,
            "hac_maxlags": profile.hac_maxlags,
            **options,
        }
    if (
        isinstance(X, pd.DataFrame)
        and isinstance(y, pd.Series)
        and not X.index.equals(y.index)
    ):
        raise ValueError("Feature/타깃 인덱스 불일치")
    if np.iscomplexobj(y):
        raise ValueError("실수 타깃 오류")
    target = (
        y.to_numpy(dtype=float, na_value=np.nan)
        if isinstance(y, pd.Series)
        else np.asarray(y, dtype=float)
    )
    if (
        raw.empty
        or not raw.columns.is_unique
        or target.ndim != 1
        or len(target) != len(raw)
        or np.isinf(target).any()
    ):
        raise ValueError("원 자료 스키마 오류")
    missing = np.isnan(target)
    if missing.any() and profile.target_missing == "raise":
        raise ValueError("결측 타깃은 학습/평가할 수 없습니다. drop 정책을 명시하세요.")
    valid = np.flatnonzero(~missing)
    if len(valid) < max(10, profile.cv * 3):
        raise ValueError("분할/CV 표본 수 부족")
    return raw, target, missing, valid


def split_data(valid, profile):
    """유효 원행을 IID 또는 시간순으로 분리하고 학습용 CV를 만듭니다.

    Args:
        valid: 유효 원행 위치. profile: 분할 비율·방식·시드·fold 수.
    Returns:
        tuple: train 위치·test 위치·학습용 CV 객체.
    Raises:
        ValueError: 분할 불가능 또는 보류 R²에 필요한 표본 수 부족.
    """
    if profile.split_strategy == "random":
        train, test = train_test_split(
            valid, test_size=profile.test_size, random_state=profile.random_state
        )
        train = np.sort(train)
        test = np.sort(test)
        cv = KFold(profile.cv, shuffle=True, random_state=profile.random_state)
    else:
        cut = int(len(valid) * (1 - profile.test_size))
        train, test = valid[:cut], valid[cut:]
        cv = TimeSeriesSplit(profile.cv)
    if len(test) < 2:
        raise ValueError("R² 보류 평가는 2개 이상 참값이 필요합니다.")
    return train, test, cv


def candidate_settings(preprocessor, profile):
    """기본/사용자 후보 이름과 옵션 선언을 검증합니다.

    Args:
        preprocessor: 사용자 미적합 변환기 또는 None. profile: 후보 설정.
    Returns:
        list: 고유 name과 estimator 옵션을 갖는 후보 사전 목록.
    Raises:
        ValueError: 비어 있거나 중복/잘못된 이름의 후보 선언.
    """
    if profile.candidates is None:
        candidates = [{"name": "baseline"}, {"name": "vif", "vif_threshold": 10.0}]
        if preprocessor is None:
            candidates.extend(
                [
                    {"name": "feature_power", "power": "yeo-johnson"},
                    {"name": "polynomial", "degree": 2, "vif_threshold": 10.0},
                    {"name": "log_target", "target_method": "log", "smearing": True},
                ]
            )
    else:
        candidates = profile.candidates
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
            for item in candidates
        )
        or len({item["name"] for item in candidates}) != len(candidates)
    ):
        raise ValueError("고유 이름의 후보 목록이 필요합니다.")
    return candidates


def fit_candidates(
    X_train, y_train, cv, candidates, preprocessor, profile, estimator_factory
):
    """동일 학습 fold에서 후보를 비교하고 원 반응 CV 점수로 선택합니다. 최종 test를 받지 않습니다.

    Args:
        X_train, y_train: 이미 분리된 학습 Feature와 원 반응만 전달합니다.
        cv: 이 학습 자료 내부의 분할.
        candidates: 검증된 후보 목록. preprocessor: 변환기 또는 None.
        profile: 공통 추론·선택 설정. estimator_factory: 미적합 estimator 생성 함수.
    Returns:
        tuple: 후보 비교표·선택 이름·선택 모형·성공 모형 사전.
    Raises:
        ValueError/TypeError: 잘못된 옵션·시간 조건 또는 모든 후보 적합 실패.
    """
    allowed = set(inspect.signature(RegressionEstimator).parameters) - {"preprocessor"}
    rows = []
    models = {}
    for candidate in candidates:
        options = {key: value for key, value in candidate.items() if key != "name"}
        if set(options) - allowed:
            raise ValueError("후보 추정기 옵션 오류: " + str(set(options) - allowed))
        options = (
            dict(
                alpha=profile.alpha,
                cov_type=profile.cov_type,
                hac_maxlags=profile.hac_maxlags,
                **options,
            )
            if not {"alpha", "cov_type", "hac_maxlags"} & set(options)
            else dict(
                {
                    "alpha": profile.alpha,
                    "cov_type": profile.cov_type,
                    "hac_maxlags": profile.hac_maxlags,
                },
                **options,
            )
        )
        estimator = estimator_factory(preprocessor=preprocessor, **options)
        if estimator.cov_type == "HAC" and profile.split_strategy != "chronological":
            raise ValueError("후보의 HAC도 chronological 자료에서만 사용하세요.")
        try:
            scores = cross_val_score(
                estimator,
                X_train.copy(),
                y_train.copy(),
                cv=cv,
                scoring=profile.scoring,
                error_score="raise",
            )
            estimator.fit(X_train.copy(), y_train.copy())
            metrics = regression_metrics(
                y_train.copy(), estimator.predict(X_train.copy())
            )
            models[candidate["name"]] = estimator
            rows.append(
                dict(
                    name=candidate["name"],
                    cv_score=float(scores.mean()),
                    cv_std=float(scores.std()),
                    train_r2=metrics["r2"],
                    train_rmse=metrics["rmse"],
                    fit_r2=estimator.inference_.fit_metrics["r2"],
                    fit_adjusted_r2=estimator.inference_.fit_metrics["adjusted_r2"],
                    n_features=len(estimator.output_features_),
                    removed_train_rows=len(estimator.removed_positions_),
                    error=None,
                )
            )
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            rows.append(
                dict(
                    name=candidate["name"],
                    cv_score=np.nan,
                    cv_std=np.nan,
                    train_r2=np.nan,
                    train_rmse=np.nan,
                    fit_r2=np.nan,
                    fit_adjusted_r2=np.nan,
                    n_features=None,
                    removed_train_rows=None,
                    error=str(error),
                )
            )
    comparison = pd.DataFrame(rows)
    eligible = comparison.loc[comparison.cv_score.notna()]
    if eligible.empty:
        raise ValueError(
            "모든 후보가 실패했습니다: "
            + str(comparison[["name", "error"]].to_dict("records"))
        )
    selected_name = eligible.loc[eligible.cv_score.idxmax(), "name"]
    selected = models[selected_name]
    return comparison, selected_name, selected, models
