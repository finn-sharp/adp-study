"""
================================================================================
arima.py - ARIMA 학습·차수 선택·고정 horizon 예측
================================================================================
설명:
    - 기법별 Profile·JSON 레시피·공개 실행기·전용 Output DTO를 제공합니다.
분석 가정:
    - 시간 순 정규 격자·결측 없는 학습 자료. 미래 목표값은 학습·차수 선택에 사용하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-20
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-20  김재현(Finn)     v1.0.0    도메인 공통화·전용 명세·실행 예제 추가
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
from typing import Any, Optional, Dict, List
import numpy as np
from src.output import OutputDTO
from src.Recipes import resolve_profile
from ._common import validate, fit


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class ARIMAProfile:
    """Attributes:
    order (List[int]): 기본 비계절 (p,d,q), 기본 [1,0,0].
    candidate_orders (Optional[List[List[int]]]): 학습 AIC 비교 후보. 기본 None.
    steps (int): 예측 horizon, 기본 12, 양의 정수.
    alpha (float): 예측구간 유의수준, 기본 0.05.
    trend (Optional[str]): None/n/c/t/ct, 기본 None. backend의 적분 차수 제약 적용.
    maxiter (int): 최적화 최대 반복, 기본 100.
    enforce_stationarity (bool): 정상 AR 제약, 기본 True.
    enforce_invertibility (bool): 가역 MA 제약, 기본 True.
    require_convergence (bool): 수렴 후보만 선택, 기본 True.
    """

    order: List[int] = field(default_factory=lambda: [1, 0, 0])
    candidate_orders: Optional[List[List[int]]] = None
    steps: int = 12
    alpha: float = 0.05
    trend: Optional[str] = None
    maxiter: int = 100
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    require_convergence: bool = True


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ARIMAOutputDTO(OutputDTO):
    """Attributes:
    method (str): backend 종류.
    parameters (Dict[str, Any]): 적용한 전체 Profile.
    selected_order (List[int]): 학습 AIC 선택 비계절 차수.
    seasonal_order (List[int]): 계절 차수. ARIMA는 [0,0,0,0].
    candidates (List[Dict[str, Any]]): 후보별 AIC·BIC·수렴 또는 실패 이유.
    predictions (np.ndarray): steps개 미래 시점 예측 평균.
    prediction_lower (np.ndarray): 같은 순서의 예측구간 하한.
    prediction_upper (np.ndarray): 같은 순서의 예측구간 상한.
    forecast_index (Optional[List[Any]]): backend 미래 인덱스, 배열 입력은 None 가능.
    residuals (np.ndarray): 학습 관측 순서의 잔차. 초기화 구간은 별도 고려.
    aic (float): 선택 후보 학습 AIC.
    bic (float): 선택 후보 학습 BIC.
    converged (bool): 최적화 수렴 여부.
    model (Any): 적합 statsmodels 결과. 기본 JSON에서 제외.
    warnings (List[str]): 선택 편향·외생변수 가용성·미수렴 제한.
    """

    method: str
    parameters: Dict[str, Any]
    selected_order: List[int]
    seasonal_order: List[int]
    candidates: List[Dict[str, Any]]
    predictions: np.ndarray
    prediction_lower: np.ndarray
    prediction_upper: np.ndarray
    forecast_index: Optional[List[Any]]
    residuals: np.ndarray
    aic: float
    bic: float
    converged: bool
    model: Any
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate_inputs(y, exog, future_exog, profile):
    """Args: y/exog/future_exog: 학습·미래 입력; profile: 전용 Profile.
    Returns: tuple: 공통 검증 결과.
    Raises: TypeError/ValueError: Profile·차수·시계열·외생변수 오류.
    """
    if not isinstance(profile, ARIMAProfile):
        raise TypeError("ARIMAProfile이 필요합니다.")
    return validate(y, exog, future_exog, profile, False)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _run(y, exog, future_exog, p):
    """Args: y/exog/future_exog: 입력; p: Profile.
    Returns: ARIMAOutputDTO: 학습·예측 결과.
    Raises: ValueError: 적격 후보 없음. ImportError: statsmodels 누락.
    """
    return fit(y, exog, future_exog, p, False, ARIMAOutputDTO)


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_arima(
    y,
    profile: Optional[ARIMAProfile] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    exog=None,
    future_exog=None,
) -> ARIMAOutputDTO:
    """Args:
        y: 시간순 학습 1차원 수치 배열 또는 정규 시간 인덱스 Series.
        profile: 명시적 Profile. JSON 옵션과 혼용 불가.
        recipe_path: 사용자 JSON. 기본 timeseries-arima.json.
        overrides: 일회성 설정.
        exog: 선택 학습 외생변수 (n,k). 행 순서가 y와 대응.
        future_exog: exog 사용 시 필수 미래 외생변수 (steps,k).
    Returns: ARIMAOutputDTO: 학습에서만 선택한 모형과 예측.
    Raises: TypeError/ValueError/OSError: 입력·레시피·적합 오류. ImportError: 의존성 누락.
    """
    p = resolve_profile(
        "timeseries-arima", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_inputs(y, exog, future_exog, p)
    return _run(y, exog, future_exog, p)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    y = 2 * np.sin(np.arange(120) * 2 * np.pi / 12) + rng.normal(size=120)
    print(run_arima(y).to_json(indent=2))
