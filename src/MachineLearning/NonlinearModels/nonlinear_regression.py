"""
================================================================================
nonlinear_regression.py - 명명된 비선형 평균모형 회귀
================================================================================
설명:
    - 기법별 Profile·JSON 레시피·공개 실행기·전용 Output DTO를 제공합니다.
분석 가정:
    - 독립 수치 관측·정확한 평균 함수형태·식별 가능한 파라미터. 학습 잔차와 보류 성능을 구분합니다.
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
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from src.output import OutputDTO
from src.Recipes import resolve_profile


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class NonlinearRegressionProfile:
    """Attributes:
    model_name (str): exponential/logistic/michaelis_menten. 기본 exponential.
        exponential=a*exp(b*x)+c; logistic=L*expit(k*(x-x0)); michaelis=a*x/(b+x).
    initial_parameters (List[float]): 위 순서의 초기값, 기본 [1,0.1,0].
    lower_bounds (Optional[List[float]]): 동일 길이 유한 하한. 기본 None=무한 경계.
    upper_bounds (Optional[List[float]]): 동일 길이 유한 상한. 기본 None=무한 경계.
    maxfev (int): 최대 함수 평가 수, 기본 10000, 양의 정수.
    absolute_sigma (bool): sigma를 알려진 관측 표준편차로 해석, 기본 False.
        False면 공분산을 학습 잔차 분산으로 재조정. True면 sigma 입력 필수.
    """

    model_name: str = "exponential"
    initial_parameters: List[float] = field(default_factory=lambda: [1.0, 0.1, 0.0])
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None
    maxfev: int = 10000
    absolute_sigma: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class NonlinearRegressionOutputDTO(OutputDTO):
    """Attributes:
    model_name (str): 함수 종류.
    parameters (Dict[str, Any]): 전체 실행 설정.
    parameter_names (List[str]): 계수·공분산의 대응 순서.
    coefficients (np.ndarray): 비선형 최소제곱 추정 계수.
    covariance (np.ndarray): 국소 선형화 근사 계수 공분산. 비식별이면 비유한 가능.
    fitted_values (np.ndarray): 학습 입력 순서의 적합값.
    residuals (np.ndarray): 학습 y-적합값.
    training_rmse (float): 학습 RMSE. 일반화 성능이 아님.
    predictions (Optional[np.ndarray]): x_test 순서의 예측값. 미요청 None.
    test_rmse (Optional[float]): y_test를 함께 제공한 경우의 보류 RMSE.
    warnings (List[str]): 함수형태·초기값·공분산 해석 제한.
    """

    model_name: str
    parameters: Dict[str, Any]
    parameter_names: List[str]
    coefficients: np.ndarray
    covariance: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    training_rmse: float
    predictions: Optional[np.ndarray]
    test_rmse: Optional[float]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _vector(values, name):
    """Args: values: 배열 후보; name: 오류 표기.
    Returns: ndarray: 비어 있지 않은 유한 실수 1차원 배열.
    Raises: ValueError: 차원·수치 오류.
    """
    a = np.asarray(values)
    if (
        a.ndim != 1
        or not a.size
        or a.dtype.kind not in "iuf"
        or not np.isfinite(a).all()
    ):
        raise ValueError(name + "은 유한 수치 1차원 배열이어야 합니다.")
    return a.astype(float)


def _validate_inputs(x, y, sigma, x_test, y_test, p):
    """Args: x/y: 학습; sigma: 표준편차; x_test/y_test: 선택 보류; p: Profile.
    Returns: tuple: 정규화한 입력과 경계.
    Raises: TypeError/ValueError: 설정·길이·파라미터·경계 오류.
    """
    if not isinstance(p, NonlinearRegressionProfile):
        raise TypeError("NonlinearRegressionProfile이 필요합니다.")
    n = 2 if p.model_name == "michaelis_menten" else 3
    if p.model_name not in ("exponential", "logistic", "michaelis_menten"):
        raise ValueError("지원하지 않는 함수 종류")
    x, y, initial = (
        _vector(x, "x"),
        _vector(y, "y"),
        _vector(p.initial_parameters, "초기값"),
    )
    if len(x) != len(y) or len(x) <= n or len(initial) != n:
        raise ValueError("학습 길이·초기값 길이 오류")
    if isinstance(p.maxfev, bool) or not isinstance(p.maxfev, int) or p.maxfev <= 0:
        raise ValueError("maxfev 오류")
    if not isinstance(p.absolute_sigma, bool):
        raise TypeError("absolute_sigma는 bool입니다.")
    low = (
        np.full(n, -np.inf)
        if p.lower_bounds is None
        else _vector(p.lower_bounds, "하한")
    )
    high = (
        np.full(n, np.inf)
        if p.upper_bounds is None
        else _vector(p.upper_bounds, "상한")
    )
    if (
        len(low) != n
        or len(high) != n
        or (low >= high).any()
        or (initial < low).any()
        or (initial > high).any()
    ):
        raise ValueError("파라미터 경계·초기값 오류")
    if sigma is not None:
        sigma = _vector(sigma, "sigma")
        if len(sigma) != len(y) or (sigma <= 0).any():
            raise ValueError("sigma 길이·양수 조건 오류")
    elif p.absolute_sigma:
        raise ValueError("absolute_sigma=True는 sigma가 필요합니다.")
    if x_test is not None:
        x_test = _vector(x_test, "x_test")
    if y_test is not None:
        y_test = _vector(y_test, "y_test")
        if x_test is None or len(y_test) != len(x_test):
            raise ValueError("보류 길이 오류")
    return x, y, sigma, x_test, y_test, (low, high)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _function(name):
    """Args: name: 검증된 함수 이름.
    Returns: tuple: callable과 파라미터 이름. 사용자가 함수를 재정의하지 않습니다.
    Raises: ValueError: 지원하지 않는 함수 이름.
    """
    if name == "exponential":
        return lambda x, a, b, c: a * np.exp(b * x) + c, ["a", "b", "c"]
    if name == "logistic":
        return lambda x, L, k, x0: L * expit(k * (x - x0)), ["L", "k", "x0"]
    if name == "michaelis_menten":
        return lambda x, a, b: a * x / (b + x), ["a", "b"]
    raise ValueError("지원하지 않는 함수")


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def build_nonlinear_regressor(profile=None, *, recipe_path=None, overrides=None):
    """미적합 CV adapter를 기존 비선형 Profile/JSON으로 구성합니다.

    Args:
        profile: NonlinearRegressionProfile 또는 None.
        recipe_path: 선택 사용자 JSON.
        overrides: Profile 생략 시 일회성 재정의.
    Returns:
        NonlinearRegressor: 단일 Feature의 미적합 sklearn 추정기.
    Raises:
        ValueError/TypeError/OSError: Profile/레시피 오류. fit에서 정의역 검증.
    """
    from ._estimator import NonlinearRegressor

    p = resolve_profile(
        "nonlinear-regression", profile, recipe_path=recipe_path, overrides=overrides
    )
    if p.absolute_sigma:
        raise ValueError("CV adapter에는 알려진 sigma를 전달하지 않습니다.")
    return NonlinearRegressor(
        p.model_name, p.initial_parameters, p.lower_bounds, p.upper_bounds, p.maxfev
    )


def run_nonlinear_regression(
    x,
    y,
    profile: Optional[NonlinearRegressionProfile] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    sigma=None,
    x_test=None,
    y_test=None,
) -> NonlinearRegressionOutputDTO:
    """Args:
        x/y: 학습 설명변수·반응변수. 단일 설명변수의 비선형 회귀.
        profile: 명시적 Profile, JSON 옵션과 혼용 불가.
        recipe_path: 사용자 JSON, 기본 nonlinear-regression.json.
        overrides: 일회성 설정.
        sigma: 선택 학습 관측 표준편차, 양수 1차원.
        x_test/y_test: 선택 보류 자료. 적합에 사용하지 않음. y_test 생략 시 예측만 반환.
    Returns: NonlinearRegressionOutputDTO: 계수·근사 공분산·학습/보류 결과.
    Raises: TypeError/ValueError/OSError: 입력·JSON 오류. RuntimeError: 최적화 실패.
        FloatingPointError: 함수 정의역·발산. 식별 경고는 숨기지 않습니다.
    """
    p = resolve_profile(
        "nonlinear-regression", profile, recipe_path=recipe_path, overrides=overrides
    )
    x, y, sigma, xt, yt, bounds = _validate_inputs(x, y, sigma, x_test, y_test, p)
    fn, names = _function(p.model_name)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        coef, cov = curve_fit(
            fn,
            x,
            y,
            p0=p.initial_parameters,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=p.absolute_sigma,
            maxfev=p.maxfev,
        )
        fitted = fn(x, *coef)
        predicted = None if xt is None else fn(xt, *coef)
    if (
        not np.isfinite(coef).all()
        or not np.isfinite(fitted).all()
        or (predicted is not None and not np.isfinite(predicted).all())
    ):
        raise ValueError("비유한 계수·예측")
    notes = [
        "초기값과 경계에 따라 다른 국소해가 나올 수 있습니다.",
        "공분산은 국소 선형화 근사이며 계수 p값·예측구간을 제공하지 않습니다.",
    ]
    if not np.isfinite(cov).all():
        notes.append("비식별·공분산 계산 실패 가능성이 있습니다.")
    return NonlinearRegressionOutputDTO(
        p.model_name,
        asdict(p),
        names,
        coef,
        cov,
        fitted,
        y - fitted,
        float(np.sqrt(np.mean((y - fitted) ** 2))),
        predicted,
        None if yt is None else float(np.sqrt(np.mean((yt - predicted) ** 2))),
        notes,
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x = np.linspace(0, 3, 40)
    y = 2 * np.exp(0.4 * x) + 1
    print(
        run_nonlinear_regression(
            x, y, overrides={"initial_parameters": [1.0, 0.2, 0.0]}
        ).to_json(indent=2)
    )
