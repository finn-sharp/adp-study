"""기존 ARIMA/SARIMA runner를 시간순 rolling CV에 조합합니다."""

from dataclasses import asdict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from src.TimeSeries.arima import run_arima
from src.TimeSeries.sarima import run_sarima


def rolling_fit(y, profile, dto):
    """미래 test를 봉인한 뒤 expanding 학습/검증으로 차수를 선택합니다.

    Args:
        y: 유한 등간격 시간순 1차원 배열 또는 정규 DatetimeIndex Series.
        profile: RollingSelectionProfile.
        dto: 전용 결과 클래스.
    Returns:
        OutputDTO: 차수 선택, 원행 분할, 선택/naive 보류 및 조건부 잔차 진단.
    Raises:
        ValueError: 불규칙/중복/비유한 시계열·표본 부족·모든 후보 실패.
    """
    if isinstance(y, pd.Series):
        if not y.index.is_unique or not y.index.is_monotonic_increasing:
            raise ValueError("시간 순서/중복 오류")
        if (
            isinstance(y.index, pd.DatetimeIndex)
            and y.index.freq is None
            and pd.infer_freq(y.index) is None
        ):
            raise ValueError("등간격 시계열이 필요합니다.")
    if np.iscomplexobj(y):
        raise ValueError("실수 시계열이 필요합니다.")
    raw = np.asarray(y, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError("유한 1차원 시계열이 필요합니다.")
    ntrain = len(raw) - profile.horizon
    first = ntrain - profile.folds * profile.validation_size
    if first < max(20, profile.period * 3):
        raise ValueError("학습/rolling/최종 보류에 필요한 표본 부족")
    orders = (
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        if profile.candidate_orders is None
        else profile.candidate_orders
    )
    if (
        not isinstance(orders, (list, tuple))
        or not orders
        or any(
            not isinstance(order, (list, tuple))
            or len(order) != 3
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in order
            )
            for order in orders
        )
    ):
        raise ValueError("차수 후보는 비어 있지 않은 비음수 정수 3개 목록입니다.")

    def fit(values, order, steps):
        """학습/분석 단계의 계산.

        Args:
        values: 명세의 단계 입력.
        order: 명세의 단계 입력.
        steps: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        settings = dict(
            order=order, steps=steps, maxiter=profile.maxiter, alpha=profile.alpha
        )
        if profile.period > 1:
            return run_sarima(
                values,
                overrides=dict(settings, seasonal_order=[1, 0, 0, profile.period]),
            )
        return run_arima(values, overrides=settings)

    folds = [
        dict(
            train_indices=list(range(end)),
            validation_indices=list(range(end, end + profile.validation_size)),
        )
        for end in range(first, ntrain, profile.validation_size)
    ]
    rows = []
    for order in orders:
        row = dict(order=list(order), cv_rmse=None, cv_mae=None, error=None)
        rmses = []
        maes = []
        try:
            for fold in folds:
                train = fold["train_indices"]
                valid = fold["validation_indices"]
                m = fit(
                    y.iloc[train] if isinstance(y, pd.Series) else raw[train],
                    order,
                    len(valid),
                )
                rmses.append(
                    float(np.sqrt(mean_squared_error(raw[valid], m.predictions)))
                )
                maes.append(float(mean_absolute_error(raw[valid], m.predictions)))
            row.update(cv_rmse=float(np.mean(rmses)), cv_mae=float(np.mean(maes)))
        except (ValueError, np.linalg.LinAlgError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    eligible = [r for r in rows if r["error"] is None]
    if not eligible:
        raise ValueError("모든 rolling 후보가 실패했습니다: " + str(rows))
    selected = min(eligible, key=lambda r: r["cv_rmse"])
    train = np.arange(ntrain)
    test = np.arange(ntrain, len(raw))
    model = fit(
        y.iloc[train] if isinstance(y, pd.Series) else raw[train],
        selected["order"],
        profile.horizon,
    )
    baseline = np.resize(raw[ntrain - profile.period : ntrain], profile.horizon)
    truth = raw[test]

    def metrics(pred):
        """학습/분석 단계의 계산.

        Args:
        pred: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        return dict(
            rmse=float(np.sqrt(mean_squared_error(truth, pred))),
            mae=float(mean_absolute_error(truth, pred)),
            r2=float(r2_score(truth, pred)),
        )

    denom = float(
        np.mean(np.abs(raw[profile.period : ntrain] - raw[: ntrain - profile.period]))
    )
    scores = dict(
        selected=metrics(model.predictions),
        baseline=metrics(baseline),
        mase_denominator=denom,
        mase=(
            None
            if denom == 0
            else float(np.mean(np.abs(truth - model.predictions)) / denom)
        ),
    )
    burn = max(int(getattr(model.model, "loglikelihood_burn", 0)), profile.period)
    resid = np.asarray(model.residuals)[burn:]
    modeldf = (
        selected["order"][0] + selected["order"][2] + (1 if profile.period > 1 else 0)
    )
    lag = min(max(profile.period * 2, modeldf + 3), len(resid) // 4)
    diagnostic = dict(
        aic=model.aic,
        bic=model.bic,
        burn=burn,
        model_df=modeldf,
        lag=lag,
        ljung_box_statistic=None,
        ljung_box_p_value=None,
    )
    if lag > modeldf:
        lb = acorr_ljungbox(resid, lags=[lag], model_df=modeldf, return_df=True)
        diagnostic.update(
            ljung_box_statistic=float(lb.lb_stat.iloc[0]),
            ljung_box_p_value=float(lb.lb_pvalue.iloc[0]),
        )
    return dto(
        model,
        pd.DataFrame(rows),
        folds,
        selected["order"],
        train,
        test,
        model.predictions,
        baseline,
        scores,
        diagnostic,
        asdict(profile),
        [
            "차수는 과거 rolling RMSE로만 선택합니다. 최종 미래 참값은 선택/적합에 사용하지 않습니다.",
            "AIC/BIC는 선택 모형의 학습 기술량입니다. 서로 다른 차분/정규화 우도를 CV 대신 직접 혼용하지 않습니다.",
            "잔차 자기상관 비기각은 독립성 증명이 아니며 예측구간은 차수 선택 불확실성을 보정하지 않습니다.",
        ],
    )
