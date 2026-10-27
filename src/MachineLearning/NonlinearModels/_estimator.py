"""
================================================================================
_estimator.py - 명명된 비선형 회귀를 sklearn CV 안에서 재적합하는 얇은 adapter.
================================================================================
설명:
    - 명명된 비선형 회귀를 sklearn CV 안에서 재적합하는 얇은 adapter.
분석 가정:
    - 입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class NonlinearRegressor(RegressorMixin, BaseEstimator):
    """단일 설명변수 curve_fit; 전처리는 외부 Pipeline의 학습 폴드에 둡니다."""

    def __init__(
        self,
        model_name="exponential",
        initial_parameters=(1.0, 0.1, 0.0),
        lower_bounds=None,
        upper_bounds=None,
        maxfev=10000,
    ):
        """Args: model_name/initial_parameters/lower_bounds/upper_bounds/maxfev: 기존 NonlinearRegressionProfile의 같은 의미.
        Returns: None: clone 가능한 미적합 설정.
        Raises: None: 적합 시 검증.
        """
        self.model_name = model_name
        self.initial_parameters = initial_parameters
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.maxfev = maxfev

    def _x(self, X):
        """Args: X: (n,1) 유한 단일 설명변수.
        Returns: ndarray: 1차원 x.
        Raises: ValueError: 차원/정의역 오류.
        """
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or a.shape[1] != 1 or not np.isfinite(a).all():
            raise ValueError("단일 Feature의 유한 2차원 행렬이 필요합니다.")
        return a[:, 0]

    def fit(self, X, y):
        """Args: X/y: 현재 CV 학습 폴드.
        Returns: NonlinearRegressor: 적합한 자신.
        Raises: ValueError/RuntimeError: 기존 비선형 runner 적합 오류.
        """
        from .nonlinear_regression import run_nonlinear_regression

        self.result_ = run_nonlinear_regression(
            self._x(X),
            y,
            overrides=dict(
                model_name=self.model_name,
                initial_parameters=list(self.initial_parameters),
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                maxfev=self.maxfev,
            ),
        )
        self.n_features_in_ = 1
        return self

    def predict(self, X):
        """Args: X: 같은 단일 Feature의 새 관측.
        Returns: ndarray: 원 y 단위 평균 예측.
        Raises: ValueError/NotFittedError: 미적합/비유한 예측.
        """
        from .nonlinear_regression import _function

        check_is_fitted(self, "result_")
        values = _function(self.model_name)[0](self._x(X), *self.result_.coefficients)
        if not np.isfinite(values).all():
            raise ValueError("비선형 예측 overflow")
        return values
