"""
================================================================================
matrix_transform.py - 상관행렬 정리·삼각 추출·거리 변환
================================================================================
설명:
    - 상관계수를 Fisher z로 바꾸지 않고 행렬/쌍 표 구조와 거리를 변환합니다.
분석 가정:
    - 열/행이 같은 변수의 대칭 계수 행렬입니다. 결측을 무연관 0으로 대체하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    상관행렬 변환·명세·레시피 추가
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
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.MachineLearning.analysis_common import finite_number, positive_int


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class CorrelationMatrixTransformProfile:
    """행렬/쌍 표 변환 설정.

    Attributes:
        mode (str): tidy/upper/lower/distance; 기본 tidy.
        distance_kind (str): signed는 sqrt(2(1-r)), absolute는 sqrt(2(1-|r|)); 기본 signed.
        min_abs_coefficient (float): 보존할 최소 |r| [0,1]; 기본 0.
        max_p_value (Optional[float]): p 필터 [0,1] 또는 None; 지정 시 p 행렬 필요.
        include_diagonal (bool): tidy/upper/lower에서 자기 쌍 포함; distance 대각은 항상0. 기본 False.
        allow_missing (bool): NaN 허용 여부; 허용 시 정리 표에서 제외, 행렬에서는 NaN 유지. 기본 False.
        tolerance (float): 대칭/대각/r 범위 오차 허용, 0 이상; 기본 1e-8.
        max_pairs (int): 중복 없는 쌍 수 한도; 기본 10000.
    """

    mode: str = "tidy"
    distance_kind: str = "signed"
    min_abs_coefficient: float = 0.0
    max_p_value: Optional[float] = None
    include_diagonal: bool = False
    allow_missing: bool = False
    tolerance: float = 1e-8
    max_pairs: int = 10000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CorrelationMatrixTransformOutputDTO(OutputDTO):
    """열 이름을 보존한 행렬 및 쌍 표 결과.

    Attributes:
        mode (str): 적용한 변환.
        transformed_matrix (Optional[pd.DataFrame]): 삼각/거리 행렬; tidy는 None.
        pairs (pd.DataFrame): feature_x/feature_y/coefficient/abs_coefficient/p_value, |r| 내림차순.
        n_features (int): 원 변수 수.
        parameters (Dict[str, Any]): 적용 Profile.
        warnings (List[str]): PSD·결측·거리 해석 경고.
    """

    mode: str
    transformed_matrix: Optional[pd.DataFrame]
    pairs: pd.DataFrame
    n_features: int
    parameters: Dict[str, Any]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate(matrix, p_values, profile):
    """대칭·행/열 이름·계수/p 범위와 변환 조건을 확인합니다.

    Args:
        matrix: 계수 행렬.
        p_values: 같은 shape/열 순서의 p 행렬 또는 None.
        profile: 전용 설정.
    Returns:
        tuple: 계수, p 행렬, 변수 이름.
    Raises:
        ValueError/TypeError: 정의역/열/설정 오류.
    """
    if not isinstance(profile, CorrelationMatrixTransformProfile):
        raise TypeError("전용 Profile이 필요합니다.")
    if profile.mode not in (
        "tidy",
        "upper",
        "lower",
        "distance",
    ) or profile.distance_kind not in ("signed", "absolute"):
        raise ValueError("변환 방법 오류")
    for value, name in [
        (profile.min_abs_coefficient, "min_abs_coefficient"),
        (profile.tolerance, "tolerance"),
    ]:
        finite_number(value, name)
    if not 0 <= profile.min_abs_coefficient <= 1 or profile.tolerance < 0:
        raise ValueError("변환 임계값 오류")
    if not isinstance(profile.include_diagonal, bool) or not isinstance(
        profile.allow_missing, bool
    ):
        raise TypeError("포함/결측 설정은 bool입니다.")
    if profile.max_p_value is not None:
        finite_number(profile.max_p_value, "max_p_value")
        if not 0 <= profile.max_p_value <= 1 or p_values is None:
            raise ValueError("p 필터 범위/행렬 오류")
    positive_int(profile.max_pairs, "max_pairs")
    if np.iscomplexobj(matrix):
        raise ValueError("실수 행렬이 필요합니다.")
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 2 or np.isinf(a).any():
        raise ValueError("정사각 계수 행렬이 필요합니다.")
    if not profile.allow_missing and np.isnan(a).any():
        raise ValueError("계수 행렬 결측")
    if not np.allclose(a, a.T, rtol=0, atol=profile.tolerance, equal_nan=True):
        raise ValueError("대칭 계수 행렬이 필요합니다.")
    if (np.abs(a[np.isfinite(a)]) > 1 + profile.tolerance).any():
        raise ValueError("계수는 [-1,1]입니다.")
    diagonal = np.diag(a)
    finite = diagonal[np.isfinite(diagonal)]
    if not np.allclose(finite, 1.0, rtol=0, atol=profile.tolerance):
        raise ValueError("유효 대각선은1이어야 합니다.")
    names = (
        [str(c) for c in matrix.columns]
        if isinstance(matrix, pd.DataFrame)
        else ["feature_" + str(i) for i in range(len(a))]
    )
    if len(set(names)) != len(names) or (
        isinstance(matrix, pd.DataFrame) and [str(v) for v in matrix.index] != names
    ):
        raise ValueError("같은 순서의 고유 행/열 이름이 필요합니다.")
    if len(a) * (len(a) - 1) // 2 > profile.max_pairs:
        raise ValueError("행렬 쌍 한도 초과")
    if p_values is not None and np.iscomplexobj(p_values):
        raise ValueError("p 값은 실수입니다.")
    p = (
        np.full(a.shape, np.nan)
        if p_values is None
        else np.asarray(p_values, dtype=float)
    )
    if (
        p.shape != a.shape
        or np.isinf(p).any()
        or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any()
    ):
        raise ValueError("p 행렬 정의역 오류")
    if p_values is not None and not np.allclose(
        p, p.T, rtol=0, atol=profile.tolerance, equal_nan=True
    ):
        raise ValueError("대칭 p 행렬이 필요합니다.")
    if isinstance(p_values, pd.DataFrame) and (
        [str(v) for v in p_values.index] != names
        or [str(v) for v in p_values.columns] != names
    ):
        raise ValueError("p 행렬 열/행 순서 불일치")
    return np.clip(a, -1, 1), p, names


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _run_transform(a, p, names, profile):
    """중복 쌍 제거·필터·구조/거리 변환을 계산합니다.

    Args:
        a: 검증한 계수 행렬.
        p: 대응 p 행렬.
        names: 원 변수 순서.
        profile: 변환 조건.
    Returns:
        CorrelationMatrixTransformOutputDTO: 쌍 표 및 선택 행렬.
    Raises:
        ValueError: 수치 행렬 연산 오류.
    """
    keep = np.isfinite(a) & (np.abs(a) >= profile.min_abs_coefficient)
    if profile.max_p_value is not None:
        keep &= np.isfinite(p) & (p <= profile.max_p_value)
    rows = []
    for i in range(len(a)):
        for j in range(i if profile.include_diagonal else i + 1, len(a)):
            if keep[i, j]:
                rows.append(
                    dict(
                        feature_x=names[i],
                        feature_y=names[j],
                        coefficient=float(a[i, j]),
                        abs_coefficient=float(abs(a[i, j])),
                        p_value=float(p[i, j]),
                    )
                )
    pairs = pd.DataFrame(
        rows,
        columns=["feature_x", "feature_y", "coefficient", "abs_coefficient", "p_value"],
    )
    if len(pairs):
        pairs = pairs.sort_values(
            ["abs_coefficient", "feature_x", "feature_y"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
    warnings = []
    transformed = None
    if profile.mode in ("upper", "lower"):
        triangle = (
            np.triu(
                np.ones(a.shape, dtype=bool), k=0 if profile.include_diagonal else 1
            )
            if profile.mode == "upper"
            else np.tril(
                np.ones(a.shape, dtype=bool), k=0 if profile.include_diagonal else -1
            )
        )
        transformed = pd.DataFrame(
            np.where(triangle & keep, a, np.nan), index=names, columns=names
        )
    elif profile.mode == "distance":
        value = a if profile.distance_kind == "signed" else np.abs(a)
        distance = np.sqrt(np.maximum(0.0, 2 * (1 - value)))
        distance = np.where(keep, distance, np.nan)
        np.fill_diagonal(distance, 0.0)
        transformed = pd.DataFrame(distance, index=names, columns=names)
        if not np.isfinite(distance).all():
            warnings.append(
                "결측/필터로 거리 쌍이 빠져 완전한 거리 metric을 보장하지 않습니다."
            )
        if profile.distance_kind == "absolute":
            warnings.append(
                "절댓값 상관 거리는 부호를 제거합니다. 원 변수에 대한 거리 metric을 자동 보장하지 않습니다."
            )
        if np.isfinite(a).all():
            if np.linalg.eigvalsh(a).min() < -profile.tolerance:
                warnings.append(
                    "입력 행렬은 PSD가 아니므로 거리의 유클리드 임베딩 성질을 보장하지 않습니다."
                )
        else:
            warnings.append(
                "결측 계수의 거리는 NaN입니다. 거리 metric 성질을 보장하지 않습니다."
            )
    if np.isnan(a).any():
        warnings.append("미정의 상관을0으로 대체하지 않았습니다.")
    return CorrelationMatrixTransformOutputDTO(
        profile.mode, transformed, pairs, len(a), asdict(profile), warnings
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_correlation_matrix_transform(
    matrix, p_values=None, profile=None, *, recipe_path=None, overrides=None
):
    """상관행렬을 표·삼각·거리 형식으로 정리합니다.

    Args:
        matrix: 대칭 계수 행렬/DataFrame.
        p_values: 대응 p 행렬 또는 None.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        CorrelationMatrixTransformOutputDTO: 변환 결과와 해석 정보.
    Raises:
        ValueError/TypeError/OSError: 행렬·설정·레시피 오류.
    """
    profile = resolve_profile(
        "correlation-matrix-transform",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    a, p, names = _validate(matrix, p_values, profile)
    return _run_transform(a, p, names, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(run_correlation_matrix_transform([[1, 0.5], [0.5, 1]]).to_json(indent=2))
