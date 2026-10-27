"""
================================================================================
cramers_v.py - CramersV 전용 상관분석
================================================================================
설명:
    - 전용 Profile·Output DTO·레시피·원 범주 쌍의 분할표 분석을 제공합니다.
분석 가정:
    - 대응 관측·기법별 귀무가설/추론 가정을 확인합니다. 상관은 인과를 뜻하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    상관분석·명세·레시피 추가
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
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.stats.correlation._common import validate_profile, categorical_pair


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class CramersVProfile:
    """범주형 연관 분석 설정.

    Attributes:
        alpha (float): 유의수준 (0,1); 기본 0.05.
        nan_policy (str): raise/omit; 대응 쌍 함께 제거; 기본 'raise'.
        bias_correction (bool): 작은 표본 편향 보정; p 값은 원 Pearson chi-square; 기본 True.
    """

    alpha: float = 0.05
    nan_policy: str = "raise"
    bias_correction: bool = True


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class CramersVOutputDTO(OutputDTO):
    """범주의 임의 순서 숫자 변환 없이 계산한 연관 결과.

    Attributes:
        method (str): 계수/검정 계산 방법.
        coefficient (float): Cramers V [0,1] 또는 양성 수준 기준 signed Phi [-1,1].
        p_value (float): 독립성 chi-square 또는 Phi Fisher 양측 p.
        reject_h0 (bool): p < alpha.
        alpha (float): 적용 유의수준.
        n_samples (int): 대응 유효 표본 수.
        observed (pd.DataFrame): 원 범주 레벨 순서의 분할표.
        expected (pd.DataFrame): 독립성 귀무 기대 빈도.
        statistic (float): 원 Pearson chi-square 통계량.
        df (int): 독립성 자유도.
        levels_x (List[Any]): 행 범주 수준.
        levels_y (List[Any]): 열 범주 수준.
        warnings (List[str]): 희소 표·결측·해석 경고.
    """

    method: str
    coefficient: float
    p_value: float
    reject_h0: bool
    alpha: float
    n_samples: int
    observed: pd.DataFrame
    expected: pd.DataFrame
    statistic: float
    df: int
    levels_x: List[Any]
    levels_y: List[Any]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_profile(profile):
    """전용 범주형 설정을 확인합니다.

    Args:
        profile: CramersVProfile.
    Returns:
        None: 유효한 설정.
    Raises:
        ValueError/TypeError: Profile·설정 오류.
    """
    if not isinstance(profile, CramersVProfile):
        raise TypeError("CramersVProfile이 필요합니다.")
    validate_profile(profile)
    if not isinstance(profile.bias_correction, bool):
        raise TypeError("bias_correction은 bool입니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_pair(a, b, removed, profile):
    """분할표에서 전용 연관계수·검정을 계산합니다.

    Args:
        a: 첫 범주 벡터.
        b: 대응 둘째 범주 벡터.
        removed: 제거한 결측 쌍 수.
        profile: 전용 설정.
    Returns:
        CramersVOutputDTO: 계수·p·원/기대 빈도·수준.
    Raises:
        ValueError: 단일 수준·희소 보정 차원 오류.
    """
    x_codes, x_levels = pd.factorize(a, sort=False)
    y_codes, y_levels = pd.factorize(b, sort=False)
    if len(x_levels) < 2 or len(y_levels) < 2:
        raise ValueError("두 변수 모두 2개 이상 관측 범주가 필요합니다.")
    observed = np.zeros((len(x_levels), len(y_levels)), dtype=int)
    np.add.at(observed, (x_codes, y_codes), 1)
    statistic, p, df, expected = stats.chi2_contingency(observed, correction=False)
    n = len(a)
    r, c = observed.shape
    phi2 = statistic / n
    if profile.bias_correction:
        phi2 = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        c = c - (c - 1) ** 2 / (n - 1)
    denominator = min(r - 1, c - 1)
    if denominator <= 0:
        raise ValueError("편향 보정 Cramers V의 유효 차원이 양수가 아닙니다.")
    coefficient = float(np.sqrt(phi2 / denominator))
    method = "cramers-v-bias-corrected" if profile.bias_correction else "cramers-v"
    notes = []
    if removed:
        notes.append(str(removed) + "개 결측 쌍 제거")
    if (expected < 5).any():
        notes.append(
            "기대 빈도 5 미만 셀이 있습니다. chi-square 근사의 적절성을 확인하세요."
        )
    frame = pd.DataFrame(observed, index=x_levels, columns=y_levels)
    expected_frame = pd.DataFrame(expected, index=x_levels, columns=y_levels)
    return CramersVOutputDTO(
        method,
        coefficient,
        float(p),
        bool(p < profile.alpha),
        profile.alpha,
        len(a),
        frame,
        expected_frame,
        float(statistic),
        int(df),
        list(x_levels),
        list(y_levels),
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_cramers_v(
    x: Any,
    y: Any,
    profile: Optional[CramersVProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CramersVOutputDTO:
    """대응 범주 변수의 전용 연관분석을 실행합니다.

    Args:
        x: 첫 범주 벡터.
        y: 대응 범주 벡터.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        CramersVOutputDTO: 원 범주 수준을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·레시피 오류.
    """
    profile = resolve_profile(
        "correlation-cramers-v", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(profile)
    a, b, removed = categorical_pair(x, y, profile)
    return _run_pair(a, b, removed, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_cramers_v([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1]).to_json(indent=2))
