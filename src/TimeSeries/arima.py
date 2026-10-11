"""
================================================================================
arima.py - ARIMA·SARIMA 전용 Profile 및 시간 순서 평가
================================================================================
설명:
    - 기존 forecasting.py의 분할·AIC 탐색·예측구간 구현을 재사용합니다.
    - SARIMA는 외생변수 없는 SARIMAX 엔진으로 구현하며 s는 관측 단위의 계절 주기입니다.
사용 예제:
    - python -m examples.extended_workflow
    - docs/EXTENSIONS.md의 import/Profile/run 호출 예제를 참조합니다.
작성 정보: 작성일 2026-10-11, 버전 1.0.0
변경 이력: 2026-10-11 v1.0.0 ARIMA/SARIMA 공개 실행기와 계절 입력 검증 추가
================================================================================
"""
__version__ = "1.0.0"
__status__ = "Development"

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from src.TimeSeries.forecasting import ForecastProfile, ForecastResult, run_forecast, _order


@dataclass
class ARIMAProfile(ForecastProfile):
    """ARIMA 조건.

    Attributes:
        method: arima 고정; order는 (p,d,q). d는 모델 내부 차분 횟수.
        holdout: 마지막 보류 관측 수.
        order_candidates: 선택적 학습 AIC 탐색. 나머지는 ForecastProfile과 동일.
    """
    method: str = "arima"


@dataclass
class SARIMAProfile(ForecastProfile):
    """계절 ARIMA 조건.

    Attributes:
        method: sarimax 고정, 외생변수 없이 SARIMA 적합.
        seasonal_order: (P,D,Q,s), 기본 (1,0,0,12). 최소 하나의 계절항 및 s>=2.
        seasonal_candidates: 각 후보도 계절항과 주기가 유효해야 함.
    """
    method: str = "sarimax"
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 0, 12)


def run_arima(data: Any, profile: Optional[ARIMAProfile] = None) -> ForecastResult:
    """원 시계열에 ARIMA를 적합하고 마지막 구간을 평가합니다.

    Args:
        data: 정렬된 고유 인덱스의 등간격 Series 또는 시간 순서 1차원 배열.
        profile: 비계절 차수·보류 수·후보·예측구간 설정.
    Returns:
        ForecastResult: 학습 모델, 원 단위 예측/구간, 보류 MAE/MSE/RMSE 및 후보 이력.
    Raises:
        ValueError: 잘못된 차수/길이/인덱스/방법 또는 적합 실패.
        ImportError: statsmodels 미설치.
    Notes:
        결측이나 불규칙 간격은 호출 전에 처리하세요. 입력을 미리 d번 차분하지 않습니다.
    """
    p = profile or ARIMAProfile()
    if p.method != "arima":
        raise ValueError("run_arima는 arima method만 지원합니다.")
    return run_forecast(data, p)


def run_sarima(data: Any, profile: Optional[SARIMAProfile] = None) -> ForecastResult:
    """계절항을 갖는 SARIMA를 적합하고 마지막 구간을 평가합니다.

    Args:
        data: 원 단변량 등간격 시계열. 후보별 학습 길이는 최소 두 계절 주기.
        profile: 비계절/계절 차수, holdout, alpha 및 학습 AIC 후보 설정.
    Returns:
        ForecastResult: SARIMAX 결과 객체, 원 단위 예측/구간과 보류 지표.
    Raises:
        ValueError: 계절항/주기/방법 오류, 데이터 부족 또는 수렴 후보 없음.
        ImportError: statsmodels 미설치.
    Notes:
        s=12는 월별 자료의 연간 주기 예입니다. 달력에서 자동 추론하지 않습니다.
        AIC 선택은 학습 구간만 사용하며 최종 결과가 전체 시계열 재적합 모델은 아닙니다.
    """
    p = profile or SARIMAProfile()
    if p.method != "sarimax":
        raise ValueError("SARIMA는 외생변수 없는 sarimax method를 사용합니다.")
    seasons = p.seasonal_candidates if p.seasonal_candidates is not None else [p.seasonal_order]
    if not seasons or isinstance(p.holdout, bool) or not isinstance(p.holdout, int) or p.holdout < 1:
        raise ValueError("유효한 후보 및 양의 정수 holdout이 필요합니다.")
    for season in seasons:
        order = _order(season, 4)
        if not any(order[:3]) or order[3] < 2 or len(data)-p.holdout < 2*order[3]:
            raise ValueError("계절항, s>=2, 학습 데이터 최소 두 주기가 필요합니다.")
    return run_forecast(data, p)
