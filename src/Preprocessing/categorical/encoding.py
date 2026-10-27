"""
================================================================================
encoding.py - CategoricalEncoding 전용 전처리
================================================================================
설명:
    - 전용 Profile·Output DTO·JSON·FeatureSpec/Pipeline 호환 변환기를 제공합니다.
분석 가정:
    - fit은 명시한 학습 자료만 사용하며 새 입력에서는 transform만 수행합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    전처리·명세·레시피 추가
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import inspect
import numpy as np
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.MachineLearning.analysis_common import finite_number
from src.Preprocessing._common import (
    ProfileTransformer,
    feature_array,
    execute_preprocessing,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class CategoricalEncodingProfile:
    """전용 전처리 설정.

    Attributes:
        method (str): onehot/ordinal/frequency; 기본 'onehot'.
        handle_unknown (str): ignore/error; 기본 'ignore'.
        drop_first (bool): OneHot 첫 학습 범주 제거; 기본 False.
        sparse_output (bool): OneHot sparse 출력; 기본 False.
        unknown_value (int): Ordinal 미지 범주 값; 음수; 기본 -1.
        category_orders (Optional[List[List[str]]]): 원 열 순서의 명시적 범주 목록; None이면 문자열 학습 순서; 기본 None.
        unknown_frequency (float): Frequency 미지 범주 값 [0,1]; 기본 0.
    """

    method: str = "onehot"
    handle_unknown: str = "ignore"
    drop_first: bool = False
    sparse_output: bool = False
    unknown_value: int = -1
    category_orders: Optional[List[List[str]]] = None
    unknown_frequency: float = 0


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class CategoricalEncodingOutputDTO(OutputDTO):
    """전처리 자료·열과 재사용 변환기.

    Attributes:
        preprocessed_feature (Any): 학습 자료 변환 행렬; OneHot는 sparse일 수 있음.
        new_preprocessed_feature (Any): 새 자료 변환 또는 None; 재적합하지 않음.
        input_features (List[str]): 원 열 이름과 순서.
        output_features (List[str]): 출력 열 이름과 순서.
        parameters (Dict[str, Any]): 적용한 Profile.
        transformer (Any): 적합 변환기; JSON에서 제외.
        n_samples (int): 학습 행 수.
        warnings (List[str]): 해석 정보.
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


def _validate_profile(profile):
    """전용 설정의 정의역을 확인합니다.

    Args:
        profile: CategoricalEncodingProfile.
    Returns:
        None: 유효한 설정.
    Raises:
        ValueError/TypeError: Profile·설정 오류.
    """
    if not isinstance(profile, CategoricalEncodingProfile):
        raise TypeError("CategoricalEncodingProfile이 필요합니다.")
    if profile.method not in ("onehot", "ordinal", "frequency"):
        raise ValueError("method 오류")
    if profile.handle_unknown not in ("ignore", "error"):
        raise ValueError("handle_unknown 오류")
    for field in ("drop_first", "sparse_output"):
        if not isinstance(getattr(profile, field), bool):
            raise TypeError(field + "는 bool입니다.")
    if (
        isinstance(profile.unknown_value, bool)
        or not isinstance(profile.unknown_value, int)
        or profile.unknown_value >= 0
    ):
        raise ValueError("unknown_value는 음수 정수입니다.")
    finite_number(profile.unknown_frequency, "unknown_frequency")
    if not 0 <= profile.unknown_frequency <= 1:
        raise ValueError("unknown_frequency 오류")
    if profile.category_orders is not None:
        if not isinstance(profile.category_orders, list) or not profile.category_orders:
            raise ValueError("category_orders 오류")
        for order in profile.category_orders:
            if (
                not isinstance(order, list)
                or not order
                or any(not isinstance(v, str) for v in order)
                or len(set(order)) != len(order)
            ):
                raise ValueError("category_orders 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def build_categorical_encoding_transformer(
    profile: Optional[CategoricalEncodingProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """FeatureSpec에 사용할 미적합 변환기를 구성합니다.

    Args:
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: Profile 생략 시 재정의.
    Returns:
        ProfileTransformer: clone 가능한 미적합 변환기.
    Raises:
        ValueError/TypeError/OSError: 설정·레시피 오류.
    """
    profile = resolve_profile(
        "categorical-encoding", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    from ._frequency import FrequencyEncoder

    categories = "auto" if profile.category_orders is None else profile.category_orders
    if profile.method == "onehot":
        params = dict(
            categories=categories,
            handle_unknown=profile.handle_unknown,
            drop="first" if profile.drop_first else None,
        )
        params[
            (
                "sparse_output"
                if "sparse_output" in inspect.signature(OneHotEncoder).parameters
                else "sparse"
            )
        ] = profile.sparse_output
        estimator = OneHotEncoder(**params)
    elif profile.method == "ordinal":
        params = dict(
            categories=categories,
            handle_unknown=(
                "use_encoded_value" if profile.handle_unknown == "ignore" else "error"
            ),
        )
        if profile.handle_unknown == "ignore":
            params["unknown_value"] = profile.unknown_value
        estimator = OrdinalEncoder(**params)
    else:
        estimator = FrequencyEncoder(profile.handle_unknown, profile.unknown_frequency)
    return ProfileTransformer(estimator, numeric=False)


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_categorical_encoding(
    X_train: Any,
    X_new: Any = None,
    profile: Optional[CategoricalEncodingProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CategoricalEncodingOutputDTO:
    """학습 구간만 적합하고 원/새 자료를 변환합니다.

    Args:
        X_train: 학습 2차원 자료.
        X_new: 같은 열의 새 자료 또는 None.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: Profile 생략 시 재정의.
    Returns:
        CategoricalEncodingOutputDTO: 전처리 행렬·열·변환기.
    Raises:
        ValueError/TypeError/OSError: 자료·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "categorical-encoding", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    transformer = build_categorical_encoding_transformer(profile)
    return execute_preprocessing(
        X_train, X_new, profile, transformer, CategoricalEncodingOutputDTO
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_categorical_encoding([["a"], ["b"], ["a"]]).to_json(indent=2))
