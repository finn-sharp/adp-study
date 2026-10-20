"""ARIMA/SARIMA 검증·학습 공통부. 보류 목표값을 입력받지 않습니다."""
from dataclasses import asdict
import numpy as np
import pandas as pd


def validate(y, exog, future_exog, p, seasonal):
    """Args: y: 학습 시계열; exog/future_exog: 학습/미래 설명변수; p: Profile; seasonal: SARIMA 여부.
    Returns: tuple: 검증된 y·학습/미래 외생변수·차수 후보.
    Raises: TypeError/ValueError: 설정·자료·시간 격자·미래 가용성 오류.
    """
    def integer(value, name, minimum=0):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum: raise ValueError(name+' 정수 범위 오류')
    integer(p.steps, 'steps', 1); integer(p.maxiter, 'maxiter', 1)
    if isinstance(p.alpha,bool) or not isinstance(p.alpha,(int,float)) or not 0<p.alpha<1: raise ValueError('alpha 오류')
    for flag in ('enforce_stationarity','enforce_invertibility','require_convergence'):
        if not isinstance(getattr(p,flag), bool): raise TypeError(flag+'는 bool입니다.')
    if p.trend not in (None, 'n', 'c', 't', 'ct'): raise ValueError('trend 오류')
    if isinstance(y, pd.Series):
        if not y.index.is_unique or not y.index.is_monotonic_increasing: raise ValueError('학습 시간 인덱스 순서·중복 오류')
        if isinstance(y.index,pd.DatetimeIndex):
            freq = y.index.freq or pd.infer_freq(y.index)
            if freq is None: raise ValueError('불규칙 날짜는 먼저 정규 격자로 정리하세요.')
            y = y.asfreq(freq)
    a = np.asarray(y)
    if a.ndim!=1 or a.size<8 or a.dtype.kind not in 'iuf' or not np.isfinite(a).all(): raise ValueError('8개 이상 유한한 수치형 학습 관측이 필요합니다.')
    orders = p.candidate_orders if p.candidate_orders is not None else [p.order]
    if not isinstance(orders,(list,tuple)) or not orders: raise ValueError('차수 후보 오류')
    for order in orders:
        if not isinstance(order,(list,tuple)) or len(order)!=3: raise ValueError('order=(p,d,q)가 필요합니다.')
        for value in order: integer(value,'order')
    if seasonal:
        if not isinstance(p.seasonal_order,(list,tuple)) or len(p.seasonal_order)!=4: raise ValueError('seasonal_order 오류')
        for value in p.seasonal_order: integer(value,'seasonal_order')
        if p.seasonal_order[3]<2: raise ValueError('SARIMA 주기는 2 이상이어야 합니다.')
    def matrix(x, rows, name):
        a = np.asarray(x)
        if a.ndim!=2 or a.shape[0]!=rows or a.shape[1]==0 or a.dtype.kind not in 'iuf' or not np.isfinite(a).all(): raise ValueError(name+' shape·유한 수치 오류')
        return a
    if exog is None:
        if future_exog is not None: raise ValueError('학습 외생변수 없는 미래 외생변수 오류')
        return y, None, None, orders
    if future_exog is None: raise ValueError('예측 시점에 이용 가능한 미래 외생변수가 필요합니다.')
    if isinstance(exog,pd.DataFrame) and isinstance(y,pd.Series) and not exog.index.equals(y.index): raise ValueError('학습 외생변수 시간 정렬 오류')
    if isinstance(exog,pd.DataFrame) and isinstance(future_exog,pd.DataFrame) and list(exog.columns)!=list(future_exog.columns): raise ValueError('미래 외생변수 열 순서 오류')
    if isinstance(y,pd.Series) and isinstance(y.index,pd.DatetimeIndex) and isinstance(future_exog,pd.DataFrame):
        expected=pd.date_range(start=y.index[-1],periods=p.steps+1,freq=y.index.freq)[1:]
        if not future_exog.index.equals(expected): raise ValueError('미래 외생변수 시간 정렬 오류')
    x, xf = matrix(exog,len(a),'exog'), matrix(future_exog,p.steps,'future_exog')
    if x.shape[1]!=xf.shape[1]: raise ValueError('외생변수 열 수 오류')
    return y, x, xf, orders


def fit(y, exog, future_exog, p, seasonal, dto):
    """Args: y/exog/future_exog: 원 입력; p: 검증할 Profile; seasonal: backend 선택; dto: 전용 DTO.
    Returns: DTO: 학습 AIC 최적 후보와 고정 horizon 예측.
    Raises: ValueError: 적격 후보 없음. backend 수렴 경고는 숨기지 않습니다.
    """
    y, x, xf, orders = validate(y, exog, future_exog, p, seasonal)
    candidates, eligible = [], []
    for order in orders:
        row = dict(order=list(order))
        try:
            if seasonal:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                m = SARIMAX(y, exog=x, order=tuple(order), seasonal_order=tuple(p.seasonal_order),
                            trend=p.trend, enforce_stationarity=p.enforce_stationarity,
                            enforce_invertibility=p.enforce_invertibility).fit(disp=False, maxiter=p.maxiter)
            else:
                from statsmodels.tsa.arima.model import ARIMA
                m = ARIMA(y, exog=x, order=tuple(order), trend=p.trend,
                          enforce_stationarity=p.enforce_stationarity,
                          enforce_invertibility=p.enforce_invertibility).fit(method_kwargs={'maxiter': p.maxiter})
            converged = bool(m.mle_retvals.get('converged',False))
            row.update(aic=float(m.aic), bic=float(m.bic), converged=converged)
            if np.isfinite(m.aic) and (converged or not p.require_convergence): eligible.append((float(m.aic), order, m))
        except (ValueError, np.linalg.LinAlgError) as exc:
            row.update(error=type(exc).__name__+': '+str(exc), converged=False)
        candidates.append(row)
    if not eligible: raise ValueError('수렴한 유한 AIC 후보가 없습니다: '+str(candidates))
    _, order, model = min(eligible,key=lambda item:item[0])
    prediction = model.get_forecast(steps=p.steps, exog=xf)
    mean = np.asarray(prediction.predicted_mean); ci = np.asarray(prediction.conf_int(alpha=p.alpha))
    if not np.isfinite(mean).all() or not np.isfinite(ci).all(): raise ValueError('비유한 예측·구간')
    notes = ['차수 선택은 학습 AIC입니다. 일반화 성능은 별도 보류 구간에서 평가하세요.',
             '예측구간은 모형 조건부 구간이며 차수 선택 불확실성을 모두 반영하지 않습니다.']
    if not model.mle_retvals.get('converged',False): notes.append('미수렴 후보 반환을 명시적으로 허용했습니다.')
    if x is not None: notes.append('미래 외생변수는 예측 시점에 이용 가능해야 합니다. 입력 행 순서에 대응합니다.')
    index = getattr(prediction.predicted_mean,'index',None)
    return dto('SARIMA' if seasonal else 'ARIMA', asdict(p), list(order),
               list(p.seasonal_order) if seasonal else [0,0,0,0], candidates,
               mean, ci[:,0], ci[:,1], None if index is None else index.tolist(),
               np.asarray(model.resid), float(model.aic), float(model.bic),
               bool(model.mle_retvals.get('converged',False)), model, notes)
