"""
================================================================================
spearman.py - Spearman 전용 상관분석
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
class SpearmanProfile:
    """전용 상관분석 설정.

    Attributes:
        alpha (float): 유의수준 (0,1); 기본 0.05.
        alternative (str): two-sided/less/greater; 기본 'two-sided'.
        nan_policy (str): raise/omit; omit은 쌍별 제거; 기본 'raise'.
        min_samples (int): 쌍별 최소 유효 표본, 3 이상; 기본 3.
        max_pairs (int): 행렬의 상삼각 쌍 한도; 기본 10000.
        p_adjust (str): 행렬 다중비교 none/bonferroni/holm/fdr_bh; 기본 'holm'.
        p_value_method (str): asymptotic/permutation; 기본 'asymptotic'.
        n_resamples (int): permutation 무작위 표본 수, 양수; 기본 9999.
        random_state (int): permutation 시드, 0 이상; 기본 1234.
    """

    alpha: float = 0.05
    alternative: str = "two-sided"
    nan_policy: str = "raise"
    min_samples: int = 3
    max_pairs: int = 10000
    p_adjust: str = "holm"
    p_value_method: str = "asymptotic"
    n_resamples: int = 9999
    random_state: int = 1234


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class SpearmanOutputDTO(OutputDTO):
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
class SpearmanMatrixOutputDTO(OutputDTO):
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
        profile: SpearmanProfile.
    Returns:
        None: 유효한 설정.
    Raises:
        ValueError/TypeError: Profile·설정 오류.
    """
    if not isinstance(profile, SpearmanProfile):
        raise TypeError("SpearmanProfile이 필요합니다.")
    validate_profile(profile)
    if profile.p_value_method not in ("asymptotic", "permutation"):
        raise ValueError("p_value_method 오류")
    positive_int(profile.n_resamples, "n_resamples")
    positive_int(profile.random_state, "random_state", 0)


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
        SpearmanOutputDTO: 계수·p·유효 표본 수.
    Raises:
        ValueError: 퇴화 자료·수치 오류·미지원 exact 동률.
    """
    notes = []
    if removed:
        notes.append(str(removed) + "개 대응 결측 쌍을 제거했습니다.")
    kwargs = (
        {"alternative": profile.alternative}
        if "alternative" in inspect.signature(stats.spearmanr).parameters
        else {}
    )
    coefficient, p = stats.spearmanr(a, b, **kwargs)
    coefficient = float(coefficient)
    p = float(p)
    interval = None
    if not kwargs and profile.alternative != "two-sided":
        p = (
            p / 2
            if (coefficient > 0) == (profile.alternative == "greater")
            else 1 - p / 2
        )
    if profile.p_value_method == "permutation":
        first = stats.rankdata(a)
        second = stats.rankdata(b)
        first -= first.mean()
        second -= second.mean()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        rng = np.random.RandomState(profile.random_state)
        hits = 0
        for _ in range(profile.n_resamples):
            value = float(first @ rng.permutation(second) / denominator)
            hits += (
                abs(value) >= abs(coefficient) - 1e-12
                if profile.alternative == "two-sided"
                else (
                    value >= coefficient - 1e-12
                    if profile.alternative == "greater"
                    else value <= coefficient + 1e-12
                )
            )
        p = (hits + 1) / (profile.n_resamples + 1)
    elif len(a) <= 500:
        notes.append(
            "Spearman 점근 p 값은 작은 표본에서 부정확할 수 있습니다. permutation을 고려하세요."
        )
    method = "spearman-" + profile.p_value_method
    if not np.isfinite(coefficient) or not np.isfinite(p):
        raise ValueError("비유한 상관 결과")
    return SpearmanOutputDTO(
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


def run_spearman(
    x: Any,
    y: Any,
    profile: Optional[SpearmanProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SpearmanOutputDTO:
    """단일 대응 수치 쌍의 상관분석을 실행합니다.

    Args:
        x: 첫 수치 벡터.
        y: 같은 위치의 대응 수치 벡터.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        SpearmanOutputDTO: 전용 추론 결과.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·레시피 오류.
    """
    profile = resolve_profile(
        "correlation-spearman", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    a, b, removed = numeric_pair(x, y, profile)
    with pywarnings.catch_warnings(record=True) as caught:
        result = _run_pair(a, b, removed, profile)
    result.warnings.extend(str(w.message) for w in caught)
    return result


def run_spearman_matrix(
    X: Any,
    profile: Optional[SpearmanProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SpearmanMatrixOutputDTO:
    """열별 상관·유효 표본·상삼각 다중검정을 함께 반환합니다.

    Args:
        X: 2차원 수치 자료; 행이 관측, 열이 변수.
        profile: 전용 설정 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        SpearmanMatrixOutputDTO: 원 열 순서의 행렬들.
    Raises:
        ValueError/TypeError/OSError: shape·한도·설정·레시피 오류.
    """
    profile = resolve_profile(
        "correlation-spearman", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    return correlation_matrix(
        X, profile, run_spearman, SpearmanMatrixOutputDTO, "spearman"
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_spearman([1, 2, 4, 5, 6], [2, 5, 3, 7, 8]).to_json(indent=2))
