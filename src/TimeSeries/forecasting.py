"""
================================================================================
forecasting.py - ARIMA/SARIMAX/VARMAX 시간 순서 예측
================================================================================
설명:
    - 시계열분석 PDF의 모델을 statsmodels.tsa.arima.model 등 지원 API로 구현합니다.
    - 뒤쪽 holdout을 분리하고 AIC 탐색은 앞쪽 학습 데이터에서만 수행합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 시간 분할·후보 이력·예측구간 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastProfile:
    """단변량 예측 조건.

    Attributes:
        method: arima 또는 sarimax.
        order: (p, d, q), 비음수 정수.
        seasonal_order: SARIMAX (P, D, Q, s). 계절항이 있으면 s >= 2.
        holdout: 마지막 테스트 관측 수. 학습에는 최소 8개 관측이 남아야 함.
        alpha: 예측구간의 유의수준 (0, 1).
        order_candidates: 학습 데이터 AIC로 비교할 order 목록. None이면 order만 적합.
        seasonal_candidates: SARIMAX 계절 후보 목록. None이면 seasonal_order만 사용.
    """
    method: str = "arima"
    order: Tuple[int, int, int] = (1, 1, 0)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    holdout: int = 12
    alpha: float = 0.05
    order_candidates: Optional[List[Tuple[int, int, int]]] = None
    seasonal_candidates: Optional[List[Tuple[int, int, int, int]]] = None


@dataclass
class ForecastOutputDTO(OutputDTO):
    """시간 순서 보류 예측 결과.

    Attributes:
        model: 학습 구간에만 적합된 statsmodels 결과 객체.
        train: 학습 원 시계열.
        test: 마지막 holdout 원 시계열.
        predictions: test 인덱스로 정렬된 원 단위 예측.
        intervals: test 인덱스의 lower/upper 예측구간.
        metrics: 보류 구간 MAE/MSE/RMSE.
        best_order: 선택된 비계절 order.
        best_seasonal_order: 선택된 계절 order.
        candidates: 후보별 AIC, 수렴 여부 및 실패 원인. 비수렴 후보는 선택하지 않음.
    """
    model: Any
    train: pd.Series
    test: pd.Series
    predictions: pd.Series
    intervals: pd.DataFrame
    metrics: Dict[str, float]
    best_order: Any
    best_seasonal_order: Any
    candidates: List[Dict[str, Any]]


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
ForecastResult = ForecastOutputDTO


@dataclass
class SeriesDiagnosticsOutputDTO(OutputDTO):
    """시계열 진단 출력 DTO.

    Attributes:
        transformed: 변환·차분 후 시계열, 원 인덱스 보존.
        boxcox_lambda: Box-Cox 사용 시 적합 lambda, 나머지는 None.
        adf_statistic, adf_p_value, reject_unit_root: ADF 통계량/p/단위근 기각 여부.
        acf, pacf: 0시차를 포함한 상관 배열.
        decomposition: 선택적 statsmodels 분해 객체(runtime). 미지정 시 None.
        decomposition_components: trend/seasonal/resid/observed 시계열 사전. 미지정 시 빈 사전.
    """
    transformed: pd.Series
    boxcox_lambda: Optional[float]
    adf_statistic: float
    adf_p_value: float
    reject_unit_root: bool
    acf: np.ndarray
    pacf: np.ndarray
    decomposition: Any
    decomposition_components: Dict[str, pd.Series] = field(default_factory=dict)


@dataclass
class MultivariateForecastOutputDTO(OutputDTO):
    """VARMAX 출력 DTO.

    Attributes:
        model: 적합 statsmodels 결과 객체(runtime).
        train, test: 시간 순서 학습/보류 DataFrame.
        predictions: test 인덱스 및 입력 열 순서의 예측 표.
        metrics: 열별 MAE/RMSE 사전.
        converged: 적합 수렴 여부; False인 결과는 실행기가 거부함.
    """
    model: Any
    train: pd.DataFrame
    test: pd.DataFrame
    predictions: pd.DataFrame
    metrics: Dict[str, Dict[str, float]]
    converged: bool

def validate_series(data: Any, min_size: int = 8) -> pd.Series:
    """시간 순서가 명시된 단변량 입력을 검증합니다.

    Args:
        data: 1차원 배열 또는 정렬된 고유 인덱스의 Series.
        min_size: 최소 관측 수.
    Returns:
        pd.Series: float 시계열. 입력을 재정렬하거나 결측치를 자동 보간하지 않습니다.
    Raises:
        ValueError: 결측·무한대, 잘못된 차원·크기 또는 중복/미정렬 인덱스.
    """
    raw = np.asarray(data, dtype=float)
    if raw.ndim != 1 or len(raw) < min_size or not np.isfinite(raw).all():
        raise ValueError("시계열은 최소 {}개의 유한한 1차원 수치여야 합니다.".format(min_size))
    series = data.astype(float).copy() if isinstance(data, pd.Series) else pd.Series(raw)
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("시계열 인덱스는 고유하며 시간 오름차순이어야 합니다.")
    if isinstance(series.index, pd.DatetimeIndex):
        if len(series) >= 3 and pd.infer_freq(series.index) is None:
            raise ValueError("날짜 관측 간격은 일정해야 합니다. 먼저 재표본화하세요.")
    return series


def _order(value, size):
    if len(value) != size or any(not isinstance(v, (int, np.integer)) or v < 0 for v in value):
        raise ValueError("order는 지정한 길이의 비음수 정수 튜플이어야 합니다.")
    if size == 4 and any(value[:3]) and value[3] < 2:
        raise ValueError("계절항을 사용하면 주기는 2 이상이어야 합니다.")
    return tuple(value)


def run_forecast(data: Any, profile: Optional[ForecastProfile] = None) -> ForecastOutputDTO:
    """시간 순서 분할, 선택적 학습 AIC 탐색 및 보류 구간 예측을 수행합니다.

    Args:
        data: 일정 간격의 단변량 원 시계열. 모델의 d/D가 차분을 수행합니다.
        profile: 예측 모형·후보·보류 수 설정.
    Returns:
        ForecastOutputDTO: 학습 모델, 원 단위 예측·구간, 평가 및 후보 이력.
    Raises:
        ValueError: 잘못된 입력·차수·구간 또는 모든 후보의 적합 실패/비수렴.
        ImportError: statsmodels 미설치.

    Notes:
        미리 차분한 입력에 d/D를 다시 적용하면 이중 차분됩니다. 원 시계열을 전달하세요.
        외생변수와 미래 관측을 사용한 재적합은 이 실행기에 포함하지 않습니다.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    p = profile if profile is not None else ForecastProfile()
    series = validate_series(data)
    if p.method not in ("arima", "sarimax") or not isinstance(p.holdout, int) or not 1 <= p.holdout <= len(series)-8 or not 0 < p.alpha < 1:
        raise ValueError("method, holdout 및 alpha를 확인하세요.")
    orders = p.order_candidates if p.order_candidates is not None else [p.order]
    seasons = p.seasonal_candidates if p.seasonal_candidates is not None else [p.seasonal_order]
    if not orders or not seasons:
        raise ValueError("후보 목록은 비어 있을 수 없습니다.")
    if p.method == "arima" and (p.seasonal_candidates is not None or any(p.seasonal_order)):
        raise ValueError("계절 모형은 sarimax method를 사용하세요.")
    orders = [_order(v, 3) for v in orders]
    seasons = [_order(v, 4) for v in seasons] if p.method == "sarimax" else [(0, 0, 0, 0)]
    train, test = series.iloc[:-p.holdout], series.iloc[-p.holdout:]
    best, best_order, best_season = None, None, None
    history = []
    for order in orders:
        for season in seasons:
            record = {"order": order, "seasonal_order": season}
            try:
                fit = ARIMA(train, order=order).fit() if p.method == "arima" else SARIMAX(train, order=order, seasonal_order=season).fit(disp=False)
                converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
                record.update(aic=float(fit.aic), converged=converged, error=None)
                if converged and np.isfinite(fit.aic) and (best is None or fit.aic < best.aic):
                    best, best_order, best_season = fit, order, season
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                record.update(aic=None, converged=False, error=str(exc))
            history.append(record)
    if best is None:
        raise ValueError("수렴한 예측 후보가 없습니다: {}".format(history))
    forecast = best.get_forecast(steps=p.holdout)
    prediction = pd.Series(np.asarray(forecast.predicted_mean), index=test.index, name="forecast")
    interval = pd.DataFrame(np.asarray(forecast.conf_int(alpha=p.alpha)), index=test.index, columns=["lower", "upper"])
    if not np.isfinite(prediction).all() or not np.isfinite(interval.values).all():
        raise ValueError("유한한 예측 및 예측구간을 계산할 수 없습니다.")
    mse = float(mean_squared_error(test, prediction))
    metrics = {"mae": float(mean_absolute_error(test, prediction)), "mse": mse, "rmse": float(np.sqrt(mse))}
    return ForecastOutputDTO(best, train, test, prediction, interval, metrics, best_order, best_season, history)


@dataclass
class SeriesDiagnosticsProfile:
    """정상성·변환 조건.

    Attributes:
        transformation: none, log, boxcox. log/boxcox 입력은 양수.
        difference_order: 반복 차분 횟수 (0 이상의 정수). diff(2)로 대체하지 않습니다.
        lags: ACF/PACF 최대 시차. 변환 후 길이의 절반 미만.
        alpha: ADF 유의수준. 귀무가설은 단위근 존재.
        seasonal_period: 분해 주기. None이면 분해 생략. 최소 두 주기 데이터 필요.
    """
    transformation: str = "none"
    difference_order: int = 0
    lags: int = 5
    alpha: float = 0.05
    seasonal_period: Optional[int] = None


def run_series_diagnostics(data: Any, profile: Optional[SeriesDiagnosticsProfile] = None) -> SeriesDiagnosticsOutputDTO:
    """변환·반복 차분 후 ADF/ACF/PACF와 선택적 계절 분해를 반환합니다.

    Args:
        data: 원 단변량 시계열.
        profile: 변환·차분·시차·주기 설정.
    Returns:
        SeriesDiagnosticsOutputDTO: transformed, boxcox_lambda, adf_statistic, adf_p_value,
            reject_unit_root, acf, pacf, decomposition. ADF 기각은 정상성의 완전한 보장이 아님.
    Raises:
        ValueError: 잘못된 설정, 상수 시계열 또는 길이 부족.
        ImportError: statsmodels 미설치.
    """
    from scipy import stats
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    p = profile if profile is not None else SeriesDiagnosticsProfile()
    series = validate_series(data)
    if not isinstance(p.difference_order, int) or p.difference_order < 0 or not 0 < p.alpha < 1:
        raise ValueError("difference_order 또는 alpha가 유효하지 않습니다.")
    box_lambda = None
    if p.transformation in ("log", "boxcox"):
        if (series <= 0).any():
            raise ValueError("log/boxcox에는 양수 입력이 필요합니다.")
        if p.transformation == "log":
            series = np.log(series)
        else:
            values, box_lambda = stats.boxcox(series.values)
            series = pd.Series(values, index=series.index)
    elif p.transformation != "none":
        raise ValueError("지원하지 않는 transformation입니다.")
    for _ in range(p.difference_order):
        series = series.diff().dropna()
    series = validate_series(series)
    if np.ptp(series.values) == 0 or not isinstance(p.lags, int) or not 1 <= p.lags < len(series)//2:
        raise ValueError("상수 입력은 검정할 수 없으며 lags는 길이의 절반 미만이어야 합니다.")
    adf = adfuller(series.values)
    decomposition = None
    if p.seasonal_period is not None:
        if not isinstance(p.seasonal_period, int) or p.seasonal_period < 2 or len(series) < 2*p.seasonal_period:
            raise ValueError("분해에는 2 이상의 주기와 최소 두 주기 데이터가 필요합니다.")
        decomposition = seasonal_decompose(series, period=p.seasonal_period, model="additive")
    return SeriesDiagnosticsOutputDTO(**{"transformed": series, "boxcox_lambda": box_lambda, "adf_statistic": float(adf[0]),
            "adf_p_value": float(adf[1]), "reject_unit_root": bool(adf[1] < p.alpha),
            "acf": acf(series.values, nlags=p.lags, fft=False),
            "pacf": pacf(series.values, nlags=p.lags, method="ywm"), "decomposition": decomposition, "decomposition_components": {} if decomposition is None else {name: getattr(decomposition, name) for name in ("trend", "seasonal", "resid", "observed")}})


@dataclass
class MultivariateForecastProfile:
    """다변량 VARMAX 조건.

    Attributes:
        order: (p, q). 기본 VAR(1). p/q 동시 양수는 식별·수렴에 유의해야 함.
        holdout: 마지막 테스트 관측 수.
        max_iter: 적합 최대 반복 수.
    """
    order: Tuple[int, int] = (1, 0)
    holdout: int = 12
    max_iter: int = 200


def run_multivariate_forecast(data: pd.DataFrame, profile: Optional[MultivariateForecastProfile] = None) -> MultivariateForecastOutputDTO:
    """여러 시계열을 시간 순서대로 분할하여 VARMAX 예측을 수행합니다.

    Args:
        data: 최소 두 열의 유한한 수치 DataFrame. 고유한 시간 오름차순 인덱스.
        profile: VARMAX 차수·보류 수 설정.
    Returns:
        MultivariateForecastOutputDTO: model, train, test, predictions, metrics (열별 MAE/RMSE), converged.
    Raises:
        ValueError: 잘못된 입력/차수/보류 구간 또는 비수렴.
        ImportError: statsmodels 미설치.
    """
    from statsmodels.tsa.statespace.varmax import VARMAX
    p = profile if profile is not None else MultivariateForecastProfile()
    if not isinstance(data, pd.DataFrame) or data.shape[1] < 2 or not data.columns.is_unique:
        raise ValueError("고유한 이름의 최소 두 시계열 열이 필요합니다.")
    for column in data:
        validate_series(data[column])
    order = _order(p.order, 2)
    if not any(order) or not isinstance(p.holdout, int) or not 1 <= p.holdout <= len(data)-8:
        raise ValueError("order 또는 holdout을 확인하세요.")
    train, test = data.iloc[:-p.holdout].astype(float), data.iloc[-p.holdout:].astype(float)
    model = VARMAX(train, order=order).fit(disp=False, maxiter=p.max_iter)
    converged = bool(model.mle_retvals.get("converged", False))
    if not converged:
        raise ValueError("VARMAX 모형이 수렴하지 않았습니다.")
    predicted = pd.DataFrame(np.asarray(model.forecast(p.holdout)), columns=data.columns, index=test.index)
    metrics = {c: {"mae": float(mean_absolute_error(test[c], predicted[c])),
                   "rmse": float(np.sqrt(mean_squared_error(test[c], predicted[c])))} for c in data}
    return MultivariateForecastOutputDTO(**{"model": model, "train": train, "test": test, "predictions": predicted,
            "metrics": metrics, "converged": converged})
