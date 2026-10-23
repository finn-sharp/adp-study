"""
================================================================================
polynomial.py - 학습 전용 polynomial·상호작용 Feature 설계
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 학습 적합과 선택은 학습/CV 안에서 수행하고 추론 가정을 확인합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.stats.regression._common import validate_profile, numeric_data, vif_values
from src.Preprocessing._common import ProfileTransformer, execute_preprocessing


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class PolynomialProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        degree (int): 다항식 최대 차수, 1~3. 기본 2.
        interaction_only (bool): 상호작용만 생성; 절편은 OLS가 추가. 기본 False.
        max_features (int): 생성 Feature 수 제한. 기본 500.
    """

    degree: int = 2
    interaction_only: bool = False
    max_features: int = 500


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class PolynomialOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        preprocessed_feature (Any): 학습 Feature/타깃 변환.
        new_preprocessed_feature (Any): 새 자료 transform 또는 None.
        input_features (List[str]): 원 열 순서.
        output_features (List[str]): 생성/선택한 열 순서.
        parameters (Dict[str,Any]): 적용 설정.
        transformer (Any): 학습 자료에만 fit한 재사용 변환기; JSON 제외.
        n_samples (int): 학습 행 수.
        warnings (List[str]): 추론·선택 해석 정보.
    """

    preprocessed_feature: Any
    new_preprocessed_feature: Any
    input_features: List[str]
    output_features: List[str]
    parameters: Dict[str, Any]
    transformer: Any
    n_samples: int
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate(profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
    Returns:
        object: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    validate_profile(profile, PolynomialProfile)
    if not 1 <= profile.degree <= 3 or profile.max_features < 1:
        raise ValueError("다항 차수/한도 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def build_polynomial_transformer(profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
        recipe_path: 전용 명세의 recipe_path 입력.
        overrides: 전용 명세의 overrides 입력.
    Returns:
        PolynomialDesignTransformer: 미적합 sklearn clone 호환 변환기.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-polynomial", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    from sklearn.preprocessing import PolynomialFeatures

    return PolynomialDesignTransformer(profile)


class PolynomialDesignTransformer(TransformerMixin, BaseEstimator):
    """절편 없는 polynomial과 생성 열 한도를 관리합니다."""

    def __init__(self, profile=None):
        """설정 보존.

        Args:
            profile: PolynomialProfile 또는 None.
        Returns:
            None: 미적합 설정.
        Raises:
            None: fit에서 검증.
        """
        self.profile = profile

    def fit(self, X, y=None):
        """학습 Feature 차원에서 생성 규칙을 적합합니다.

        Args:
            X: 학습 2차원 Feature.
            y: 미사용 sklearn 타깃.
        Returns:
            PolynomialDesignTransformer: 자신.
        Raises:
            ValueError: 차수/열 한도 오류.
        """
        import math

        raw, _ = numeric_data(X)
        profile = self.profile or PolynomialProfile()
        _validate(profile)
        d = raw.shape[1]
        if profile.interaction_only:
            count = sum(
                math.factorial(d) // (math.factorial(k) * math.factorial(d - k))
                for k in range(1, min(d, profile.degree) + 1)
            )
        else:
            count = (
                math.factorial(d + profile.degree)
                // (math.factorial(profile.degree) * math.factorial(d))
                - 1
            )
        if count > profile.max_features:
            raise ValueError("생성 Feature 한도 초과")
        from sklearn.preprocessing import PolynomialFeatures

        self.transformer_ = ProfileTransformer(
            PolynomialFeatures(
                degree=profile.degree,
                interaction_only=profile.interaction_only,
                include_bias=False,
            ),
            numeric=True,
        ).fit(X)
        self.feature_names_in_ = self.transformer_.feature_names_in_
        self.n_features_in_ = d
        return self

    def transform(self, X):
        """같은 Feature 순서에 학습 규칙을 적용합니다.

        Args:
            X: 새 Feature.
        Returns:
            ndarray: 절편 없는 생성 Feature.
        Raises:
            ValueError/NotFittedError: 미적합·열·비유한 결과.
        """
        check_is_fitted(self, "transformer_")
        result = self.transformer_.transform(X)
        if not np.isfinite(result).all():
            raise ValueError("다항 Feature overflow")
        return result

    def get_feature_names_out(self, input_features=None):
        """Feature 변환 이름을 전달합니다.

        Args:
            input_features: 상위 Pipeline의 열 이름.
        Returns:
            ndarray: 출력 열 이름.
        Raises:
            ValueError/NotFittedError: 미적합/이름 오류.
        """
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out(input_features)


def _run(X_train, X_new, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X_train: 전용 명세의 X_train 입력.
        X_new: 전용 명세의 X_new 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        PolynomialOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    transformer = build_polynomial_transformer(profile)
    return execute_preprocessing(
        X_train, X_new, profile, transformer, PolynomialOutputDTO
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_polynomial(
    X_train, X_new=None, profile=None, *, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X_train: 전용 명세의 X_train 입력.
        X_new: 전용 명세의 X_new 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        PolynomialOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-polynomial", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(X_train, X_new, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
