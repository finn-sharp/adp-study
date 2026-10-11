"""
================================================================================
regression.py - 다중 선형·다항·비선형 곡선 회귀
================================================================================
설명:
    - 여러 설명변수의 선형 회귀와 PolynomialFeatures 기반 다항 회귀를 제공합니다.
    - 임의 곡선 f(X.T, *parameters)를 scipy curve_fit으로 학습합니다.
    - 다항 회귀는 입력에 비선형이나 계수에는 선형입니다. 곡선 회귀와 구분합니다.
사용 예제:
    - python -m examples.extended_workflow
    - docs/EXTENSIONS.md의 import/Profile/run 호출 예제를 참조합니다.
작성 정보: 작성일 2026-10-11, 버전 1.0.0
변경 이력: 2026-10-11 v1.0.0 Profile/Result 공통 구조의 회귀 실행기 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 모듈 전용 결과 DTO 및 저장용 변환 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import numpy as np
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted
from src.MachineLearning.common import SupervisedProfile, SupervisedResult, run_supervised, validate_features


@dataclass
class MultipleRegressionProfile(SupervisedProfile):
    """다중 선형 회귀 설정.

    Attributes:
        fit_intercept: 절편 적합 여부. 기본 True.
        task: regression만 허용. 나머지 분할/CV 필드는 SupervisedProfile과 동일.
    """
    fit_intercept: bool = True


@dataclass
class PolynomialRegressionProfile(MultipleRegressionProfile):
    """다항 회귀 설정.

    Attributes:
        degree: 양의 정수 다항 차수. 기본 2; 교호항 포함.
        interaction_only: True이면 같은 변수의 거듭제곱을 제외.
        ridge_alpha: None이면 OLS, 0 이상의 수이면 Ridge 정규화.
        param_grid: polynomial__degree, model__alpha 등 Pipeline 이름 사용 가능.
    """
    degree: int = 2
    interaction_only: bool = False
    ridge_alpha: Optional[float] = None


@dataclass
class NonlinearRegressionProfile(SupervisedProfile):
    """비선형 최소제곱 곡선 적합 설정.

    Attributes:
        initial_params: 초기 계수 튜플. 지역 최적해/수렴에 영향을 줌; 필수.
        bounds: (하한, 상한), scalar 또는 초기 계수와 같은 길이 배열.
        max_evaluations: curve_fit 최대 함수 평가 수. 양의 정수.
        scaling: 기본 none. 다른 설정은 함수에 변환된 좌표를 전달함.
    """
    initial_params: Optional[Tuple[float, ...]] = None
    bounds: Any = (-np.inf, np.inf)
    max_evaluations: int = 10000


class CurveRegressor(RegressorMixin, BaseEstimator):
    """Pipeline/CV에서 사용할 sklearn 호환 곡선 회귀 추정기.

    Attributes:
        function: f(X.T, *parameters) -> (n_samples,) callable.
        initial_params: 계수 초기값; 비어 있지 않은 유한한 1차원 배열.
        bounds: curve_fit 계수 범위.
        max_evaluations: 최대 평가 수.
        parameters_: fit 이후 추정된 계수.
        covariance_: fit 이후 근사 계수 공분산. 식별 불가능하면 무한대일 수 있음.
        n_features_in_: fit 입력의 특성 수.
    """
    def __init__(self, function, initial_params=None, bounds=(-np.inf, np.inf), max_evaluations=10000):
        self.function = function
        self.initial_params = initial_params
        self.bounds = bounds
        self.max_evaluations = max_evaluations

    def fit(self, X, y):
        """학습 입력만으로 계수를 적합합니다.

        Args:
            X: 유한한 2차원 수치 행렬.
            y: X와 길이가 같은 유한한 수치 타깃.
        Returns:
            CurveRegressor: 적합된 self.
        Raises:
            ValueError: 입력/함수 출력/초기값이 부적절한 경우.
            RuntimeError: 지정한 평가 수 안에 최적화가 수렴하지 않는 경우.
        """
        arr = validate_features(X)
        target = np.asarray(y, dtype=float)
        params = np.asarray(self.initial_params, dtype=float)
        if not callable(self.function) or params.ndim != 1 or not len(params) or not np.isfinite(params).all():
            raise ValueError("callable 함수와 유한한 1차원 initial_params가 필요합니다.")
        if target.shape != (len(arr),) or not np.isfinite(target).all() or len(arr) <= len(params):
            raise ValueError("타깃 길이/유한성 또는 계수 대비 표본 수를 확인하세요.")
        if isinstance(self.max_evaluations, bool) or not isinstance(self.max_evaluations, int) or self.max_evaluations < 1:
            raise ValueError("max_evaluations는 양의 정수여야 합니다.")
        self.n_features_in_ = arr.shape[1]
        def checked(coordinates, *values):
            pred = np.asarray(self.function(coordinates, *values), dtype=float)
            if pred.shape != (coordinates.shape[1],) or not np.isfinite(pred).all():
                raise ValueError("함수는 유한한 (n_samples,) 출력을 반환해야 합니다.")
            return pred
        self.parameters_, self.covariance_ = curve_fit(checked, arr.T, target,
            p0=params, bounds=self.bounds, maxfev=self.max_evaluations)
        return self

    def predict(self, X):
        """학습과 같은 열 순서의 새 입력을 예측합니다.

        Args:
            X: 학습과 같은 열 수의 유한한 2차원 수치 행렬.
        Returns:
            np.ndarray: 길이 n_samples의 예측값.
        Raises:
            ValueError: 열 수 불일치 또는 잘못된 함수 출력.
            NotFittedError: fit 전에 호출.
        """
        check_is_fitted(self, "parameters_")
        arr = validate_features(X, min_samples=1)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("학습 입력과 특성 수가 다릅니다.")
        pred = np.asarray(self.function(arr.T, *self.parameters_), dtype=float)
        if pred.shape != (len(arr),) or not np.isfinite(pred).all():
            raise ValueError("함수 예측 출력이 유효하지 않습니다.")
        return pred


@dataclass
class MultipleRegressionOutputDTO(SupervisedResult):
    """MultipleRegression 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

@dataclass
class PolynomialRegressionOutputDTO(SupervisedResult):
    """PolynomialRegression 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

@dataclass
class NonlinearRegressionOutputDTO(SupervisedResult):
    """NonlinearRegression 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def run_multiple_regression(X: Any, y: Any, profile: Optional[MultipleRegressionProfile] = None) -> MultipleRegressionOutputDTO:
    """최소 두 설명변수로 다중 선형 회귀를 학습·평가합니다.

    Args:
        X: (n_samples, n_features>=2) 수치 행렬 또는 DataFrame.
        y: 1차원 연속 타깃.
        profile: 절편/분할/스케일링/CV 설정.
    Returns:
        MultipleRegressionOutputDTO: 적합 Pipeline과 보류 평가. 계수는 model.named_steps['model'].coef_.
    Raises:
        ValueError: 특성 부족 또는 regression 이외의 task.
    """
    p = profile or MultipleRegressionProfile()
    if validate_features(X).shape[1] < 2 or p.task != "regression":
        raise ValueError("다중 회귀에는 두 개 이상의 특성과 regression task가 필요합니다.")
    return MultipleRegressionOutputDTO.from_result(run_supervised(X, y, LinearRegression(fit_intercept=p.fit_intercept), p))


def run_polynomial_regression(X: Any, y: Any, profile: Optional[PolynomialRegressionProfile] = None) -> PolynomialRegressionOutputDTO:
    """학습 폴드 내부에서 다항 특성을 생성하여 회귀를 평가합니다.

    Args:
        X: 한 개 이상의 수치 특성.
        y: 연속 타깃.
        profile: 차수·교호항·선택적 정규화 및 공통 학습 조건.
    Returns:
        PolynomialRegressionOutputDTO: scaler → polynomial → model Pipeline 및 보류 지표.
    Raises:
        ValueError: 잘못된 task, 차수 또는 정규화 값.
    """
    p = profile or PolynomialRegressionProfile()
    if p.task != "regression" or isinstance(p.degree, bool) or not isinstance(p.degree, int) or p.degree < 1:
        raise ValueError("regression task 및 양의 정수 degree가 필요합니다.")
    if p.ridge_alpha is not None and (not np.isfinite(p.ridge_alpha) or p.ridge_alpha < 0):
        raise ValueError("ridge_alpha는 유한한 비음수여야 합니다.")
    model = LinearRegression(fit_intercept=p.fit_intercept) if p.ridge_alpha is None else Ridge(alpha=p.ridge_alpha, fit_intercept=p.fit_intercept)
    return PolynomialRegressionOutputDTO.from_result(run_supervised(X, y, model, p, [("polynomial", PolynomialFeatures(
        degree=p.degree, interaction_only=p.interaction_only, include_bias=False))]))


def run_nonlinear_regression(X: Any, y: Any, function: Callable, profile: NonlinearRegressionProfile) -> NonlinearRegressionOutputDTO:
    """사용자 지정 함수의 계수에 대해 비선형 최소제곱 적합·평가를 수행합니다.

    Args:
        X: 수치 특성 행렬. 함수에는 X.T 형태로 전달.
        y: 연속 타깃.
        function: f(coordinates, *parameters) -> 1차원 예측. 좌표 첫 행은 coordinates[0].
        profile: 초기 계수·범위·평가 제한 및 공통 학습 조건.
    Returns:
        NonlinearRegressionOutputDTO: 적합 Pipeline. 계수/공분산은 마지막 model의 parameters_/covariance_.
    Raises:
        ValueError: 초기 계수/입력/함수 출력 오류.
        RuntimeError: 최적화 비수렴.
    Notes:
        지역 최적해를 반환할 수 있으며 전역 최적성이나 계수 식별성을 보장하지 않습니다.
        Examples: lambda z,a,b,c: a*np.exp(b*z[0])+c; initial_params=(1., .1, 0.).
    """
    if profile is None or profile.task != "regression":
        raise ValueError("NonlinearRegressionProfile의 regression task가 필요합니다.")
    return NonlinearRegressionOutputDTO.from_result(run_supervised(X, y, CurveRegressor(function, profile.initial_params,
        profile.bounds, profile.max_evaluations), profile))
