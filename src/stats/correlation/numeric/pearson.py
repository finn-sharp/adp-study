"""
================================================================================
pearson.py - Pearson 전용 상관분석
================================================================================
설명:
    - 전용 Profile·Output DTO·레시피·단일 쌍 및 행렬 API를 제공합니다.
분석 가정:
    - 대응 관측·기법별 귀무가설/추론 가정을 확인합니다. 상관은 인과를 뜻하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    상관분석·명세·레시피 추가
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
import inspect, warnings as pywarnings
import numpy as np
import pandas as pd
from scipy import stats
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.MachineLearning.analysis_common import positive_int
from src.stats.correlation._common import (
    validate_profile,
    numeric_pair,
    correlation_matrix,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class PearsonProfile:
    """전용 상관분석 설정.

    Attributes:
        alpha (float): 유의수준 (0,1); 기본 0.05.
        alternative (str): two-sided/less/greater; 기본 'two-sided'.
        nan_policy (str): raise/omit; omit은 쌍별 제거; 기본 'raise'.
        min_samples (int): 쌍별 최소 유효 표본, 3 이상; 기본 3.
        max_pairs (int): 행렬의 상삼각 쌍 한도; 기본 10000.
        p_adjust (str): 행렬 다중비교 none/bonferroni/holm/fdr_bh; 기본 'holm'.
        confidence_interval (bool): n>3에서 Fisher 근사 신뢰구간; 기본 True.
    """

    alpha: float = 0.05
    alternative: str = "two-sided"
    nan_policy: str = "raise"
    min_samples: int = 3
    max_pairs: int = 10000
    p_adjust: str = "holm"
    confidence_interval: bool = True


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class PearsonOutputDTO(OutputDTO):
    """단일 쌍의 상관계수·추론 결과.

    Attributes:
        method (str): 적용한 기법/추론 방법.
        coefficient (float): 상관계수 [-1,1].
        p_value (float): 선택 대립가설의 p 값.
        reject_h0 (bool): p_value < alpha.
        alpha (float): 적용 유의수준.
        alternative (str): 대립가설 방향.
        n_samples (int): 쌍별 유효 표본 수.
        confidence_interval (Optional[Tuple[float, float]]): Pearson Fisher 근사 양측 구간; 다른 기법 None.
        warnings (List[str]): 제거·추론 가정·수치 경고.
    """

    method: str
    coefficient: float
    p_value: float
    reject_h0: bool
    alpha: float
    alternative: str
    n_samples: int
    confidence_interval: Optional[Tuple[float, float]]
    warnings: List[str]


@dataclass
class PearsonMatrixOutputDTO(OutputDTO):
    """상관행렬과 중복 없는 상삼각 검정의 보정 결과.

    Attributes:
        method (str): 기법 이름.
        correlation_matrix (pd.DataFrame): 계수 행렬; 미정의 값 NaN.
        p_value_matrix (pd.DataFrame): 원 p 행렬; 대각선 NaN.
        adjusted_p_value_matrix (pd.DataFrame): 지정 방법으로 보정한 p 행렬.
        n_matrix (pd.DataFrame): 각 쌍의 유효 표본 수.
        significance_matrix (pd.DataFrame): 보정 p < alpha, 대각선 False.
        feature_names (List[str]): 원 열 순서.
        alpha (float): 유의수준.
        warnings (List[str]): 미정의 쌍·결측·추론 경고.
    """

    method: str
    correlation_matrix: pd.DataFrame
    p_value_matrix: pd.DataFrame
    adjusted_p_value_matrix: pd.DataFrame
    n_matrix: pd.DataFrame
    significance_matrix: pd.DataFrame
    feature_names: List[str]
    alpha: float
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_profile(profile):
    """전용 Profile 및 기법 조건을 확인합니다.

    Args:
        profile: PearsonProfile.
    Returns:
        None: 유효한 설정.
    Raises:
        ValueError/TypeError: Profile·설정 오류.
    """
    if not isinstance(profile, PearsonProfile):
        raise TypeError("PearsonProfile이 필요합니다.")
    validate_profile(profile)
    if not isinstance(profile.confidence_interval, bool):
        raise TypeError("confidence_interval은 bool입니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_pair(a, b, removed, profile):
    """해당 기법의 계수와 추론을 계산합니다.

    Args:
        a: 검증한 첫 수치 벡터.
        b: 대응 두번째 수치 벡터.
        removed: 제거한 결측 쌍 수.
        profile: 전용 설정.
    Returns:
        PearsonOutputDTO: 계수·p·유효 표본 수.
    Raises:
        ValueError: 퇴화 자료·수치 오류·미지원 exact 동률.
    """
    notes = []
    if removed:
        notes.append(str(removed) + "개 대응 결측 쌍을 제거했습니다.")
    kwargs = (
        {"alternative": profile.alternative}
        if "alternative" in inspect.signature(stats.pearsonr).parameters
        else {}
    )
    coefficient, p = stats.pearsonr(a, b, **kwargs)
    coefficient = float(coefficient)
    p = float(p)
    if not kwargs and profile.alternative != "two-sided":
        p = (
            p / 2
            if (coefficient > 0) == (profile.alternative == "greater")
            else 1 - p / 2
        )
    interval = None
    if profile.confidence_interval:
        if len(a) > 3:
            if abs(coefficient) == 1:
                interval = (coefficient, coefficient)
            else:
                z = np.arctanh(coefficient)
                critical = stats.norm.ppf(1 - profile.alpha / 2)
                width = critical / np.sqrt(len(a) - 3)
                interval = (float(np.tanh(z - width)), float(np.tanh(z + width)))
        else:
            notes.append("Fisher 근사 신뢰구간은 n>3에서만 반환합니다.")
    method = "pearson-exact-beta"
    if not np.isfinite(coefficient) or not np.isfinite(p):
        raise ValueError("비유한 상관 결과")
    return PearsonOutputDTO(
        method,
        coefficient,
        p,
        bool(p < profile.alpha),
        profile.alpha,
        profile.alternative,
        len(a),
        interval,
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_pearson(
    x: Any,
    y: Any,
    profile: Optional[PearsonProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> PearsonOutputDTO:
    """단일 대응 수치 쌍의 상관분석을 실행합니다.

    Args:
        x: 첫 수치 벡터.
        y: 같은 위치의 대응 수치 벡터.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        PearsonOutputDTO: 전용 추론 결과.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·레시피 오류.
    """
    profile = resolve_profile(
        "correlation-pearson", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    a, b, removed = numeric_pair(x, y, profile)
    with pywarnings.catch_warnings(record=True) as caught:
        result = _run_pair(a, b, removed, profile)
    result.warnings.extend(str(w.message) for w in caught)
    return result


def run_pearson_matrix(
    X: Any,
    profile: Optional[PearsonProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> PearsonMatrixOutputDTO:
    """열별 상관·유효 표본·상삼각 다중검정을 함께 반환합니다.

    Args:
        X: 2차원 수치 자료; 행이 관측, 열이 변수.
        profile: 전용 설정 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        PearsonMatrixOutputDTO: 원 열 순서의 행렬들.
    Raises:
        ValueError/TypeError/OSError: shape·한도·설정·레시피 오류.
    """
    profile = resolve_profile(
        "correlation-pearson", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    return correlation_matrix(
        X, profile, run_pearson, PearsonMatrixOutputDTO, "pearson"
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_pearson([1, 2, 4, 5, 6], [2, 5, 3, 7, 8]).to_json(indent=2))
