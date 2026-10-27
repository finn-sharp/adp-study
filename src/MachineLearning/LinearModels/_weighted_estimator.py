"""
================================================================================
_weighted_estimator.py - CV 학습 내 Feature 처리와 log 잔차분산 추정을 수행하는 FGLS 추정기.
================================================================================
설명:
    - CV 학습 내 Feature 처리와 log 잔차분산 추정을 수행하는 FGLS 추정기.
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
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from src.Preprocessing._common import feature_array


class FeasibleWLSRegressor(RegressorMixin, BaseEstimator):
    """원 y 척도 회귀; 모든 전처리·분산 모형은 fit 학습 행에서만 추정."""

    def __init__(
        self, iterations=2, variance_floor=1e-4, weight_ratio=100.0, cov_type="HC3"
    ):
        """학습/분석 단계의 계산.

        Args:
        self: 명세의 단계 입력.
        iterations: 명세의 단계 입력.
        variance_floor: 명세의 단계 입력.
        weight_ratio: 명세의 단계 입력.
        cov_type: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        self.iterations = iterations
        self.variance_floor = variance_floor
        self.weight_ratio = weight_ratio
        self.cov_type = cov_type

    def _matrix(self, X, fit=False):
        """학습/분석 단계의 계산.

        Args:
        self: 명세의 단계 입력.
        X: 명세의 단계 입력.
        fit: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        a, names = feature_array(X, True, True)
        if fit:
            if np.isnan(a).all(axis=0).any():
                raise ValueError("전체 결측 Feature를 먼저 처리하세요.")
            self.feature_names_in_ = np.asarray(names, dtype=object)
            self.n_features_in_ = len(names)
            self.imputer_ = SimpleImputer(strategy="median").fit(a)
            self.scaler_ = StandardScaler().fit(self.imputer_.transform(a))
        elif names != self.feature_names_in_.tolist():
            raise ValueError("새 Feature 이름/순서 불일치")
        matrix = self.scaler_.transform(self.imputer_.transform(a))
        return sm.add_constant(matrix, has_constant="add")

    def fit(self, X, y):
        """학습/분석 단계의 계산.

        Args:
        self: 명세의 단계 입력.
        X: 명세의 단계 입력.
        y: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        from src.MachineLearning.LinearModels.weighted_regression import (
            WeightedRegressionProfile,
            _validate,
        )

        _validate(
            WeightedRegressionProfile(
                self.iterations, self.variance_floor, self.weight_ratio, self.cov_type
            )
        )
        if (
            isinstance(X, pd.DataFrame)
            and isinstance(y, pd.Series)
            and not X.index.equals(y.index)
        ):
            raise ValueError("Feature/반응 인덱스 오류")
        design = self._matrix(X, True)
        target = np.asarray(y, dtype=float)
        if (
            target.shape != (len(design),)
            or not np.isfinite(target).all()
            or np.ptp(target) == 0
            or len(design) <= design.shape[1] + 1
            or np.linalg.matrix_rank(design) != design.shape[1]
        ):
            raise ValueError("타깃/rank/자유도 오류")
        model = sm.OLS(target, design).fit()
        weights = np.ones(len(target))
        for _ in range(self.iterations):
            self.variance_model_ = sm.OLS(
                np.log(np.maximum(np.asarray(model.resid) ** 2, self.variance_floor)),
                design,
            ).fit()
            logvariance = np.asarray(self.variance_model_.predict(design))
            logvariance = np.clip(
                logvariance,
                logvariance.min(),
                logvariance.min() + np.log(self.weight_ratio),
            )
            variance = np.maximum(
                np.exp(np.clip(logvariance, -50, 50)), self.variance_floor
            )
            weights = 1 / variance
            weights /= weights.mean()
            model = sm.WLS(target, design, weights=weights).fit(
                cov_type=self.cov_type, use_t=True
            )
        self.weights_ = weights
        self.model_ = model
        return self

    def predict(self, X):
        """학습/분석 단계의 계산.

        Args:
        self: 명세의 단계 입력.
        X: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        check_is_fitted(self, "model_")
        values = np.asarray(self.model_.predict(self._matrix(X)))
        if not np.isfinite(values).all():
            raise ValueError("非유한 예측")
        return values
