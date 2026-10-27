"""
================================================================================
_frequency.py - 학습 상대빈도 기반 범주 변환기. 범주 순서 또는 타깃을 학습하지 않습니다.
================================================================================
설명:
    - 학습 상대빈도 기반 범주 변환기. 범주 순서 또는 타깃을 학습하지 않습니다.
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """범주별 학습 빈도를 재사용합니다.

    Attributes:
        handle_unknown (str): ignore/error.
        unknown_frequency (float): 미지 범주 값.
        frequencies_ (list): 열별 상대빈도 사전.
    """

    def __init__(self, handle_unknown="ignore", unknown_frequency=0.0):
        """설정 보존.

        Args:
            handle_unknown: 미지 범주 정책.
            unknown_frequency: 미지 범주 값.
        Returns:
            None: 미적합 변환기.
        Raises:
            None: Profile에서 검증.
        """
        self.handle_unknown = handle_unknown
        self.unknown_frequency = unknown_frequency

    def fit(self, X, y=None):
        """학습 빈도를 적합합니다.

        Args:
            X: 검증한 문자열 행렬.
            y: 사용하지 않는 타깃.
        Returns:
            FrequencyEncoder: 적합한 자신.
        Raises:
            ValueError: 입력 shape 오류.
        """
        a = np.asarray(X, dtype=object)
        self.n_features_in_ = a.shape[1]
        self.frequencies_ = [
            (pd.Series(column).value_counts() / len(a)).to_dict() for column in a.T
        ]
        return self

    def transform(self, X):
        """학습 빈도로 새 범주를 변환합니다.

        Args:
            X: 같은 열 수의 범주 행렬.
        Returns:
            np.ndarray: 상대빈도 행렬.
        Raises:
            ValueError/NotFittedError: 미지 범주·열·미적합 오류.
        """
        check_is_fitted(self, "frequencies_")
        a = np.asarray(X, dtype=object)
        if a.ndim != 2 or a.shape[1] != self.n_features_in_:
            raise ValueError("열 수 불일치")
        result = np.empty(a.shape, dtype=float)
        for index, mapping in enumerate(self.frequencies_):
            if self.handle_unknown == "error" and any(
                value not in mapping for value in a[:, index]
            ):
                raise ValueError("미지 범주")
            result[:, index] = [
                mapping.get(value, self.unknown_frequency) for value in a[:, index]
            ]
        return result
