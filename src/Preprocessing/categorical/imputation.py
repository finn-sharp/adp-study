"""
================================================================================
imputation.py - CategoricalImputation 전용 전처리
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
class CategoricalImputationProfile:
    """전용 전처리 설정.

    Attributes:
        strategy (str): most_frequent/constant; 기본 'most_frequent'.
        fill_value (str): constant의 명시적 대치 범주; 기본 '(missing)'.
    """

    strategy: str = "most_frequent"
    fill_value: str = "(missing)"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class CategoricalImputationOutputDTO(OutputDTO):
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
        profile: CategoricalImputationProfile.
    Returns:
        None: 유효한 설정.
    Raises:
        ValueError/TypeError: Profile·설정 오류.
    """
    if not isinstance(profile, CategoricalImputationProfile):
        raise TypeError("CategoricalImputationProfile이 필요합니다.")
    if profile.strategy not in ("most_frequent", "constant"):
        raise ValueError("strategy 오류")
    if not isinstance(profile.fill_value, str) or not profile.fill_value:
        raise ValueError("fill_value 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def build_categorical_imputation_transformer(
    profile: Optional[CategoricalImputationProfile] = None,
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
        "categorical-imputation", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    from sklearn.impute import SimpleImputer

    params = dict(strategy=profile.strategy, fill_value=profile.fill_value)
    if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
        params["keep_empty_features"] = True
    return ProfileTransformer(
        SimpleImputer(**params),
        numeric=False,
        allow_nan=True,
        reject_empty=profile.strategy != "constant",
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_categorical_imputation(
    X_train: Any,
    X_new: Any = None,
    profile: Optional[CategoricalImputationProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CategoricalImputationOutputDTO:
    """학습 구간만 적합하고 원/새 자료를 변환합니다.

    Args:
        X_train: 학습 2차원 자료.
        X_new: 같은 열의 새 자료 또는 None.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: Profile 생략 시 재정의.
    Returns:
        CategoricalImputationOutputDTO: 전처리 행렬·열·변환기.
    Raises:
        ValueError/TypeError/OSError: 자료·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "categorical-imputation", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    transformer = build_categorical_imputation_transformer(profile)
    return execute_preprocessing(
        X_train, X_new, profile, transformer, CategoricalImputationOutputDTO
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_categorical_imputation([["a"], [None], ["b"]]).to_json(indent=2))
