"""
================================================================================
PostHoc.py - 독립 집단 평균 사후검정
================================================================================
설명:
    - 독립 집단 평균 사후검정을 수행하고 Profile·Output DTO로 입력과 결과를 정의합니다.

분석 가정:
    - 관측은 독립입니다. Tukey는 정규성·등분산, t 검정은 집단 내 정규성을 가정하며 equal_var=False이면 Welch t를 사용합니다.
    - 결측값·무한대는 사전에 처리하며 가정 검정은 자동 수행하지 않습니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.1.0    통계 명세·검증·Output DTO 통일 및 검정 구현
    2026-10-11  김재현(Finn)     v1.2.0    모델별 JSON 레시피 로딩과 공개 실행기 연결
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.2.0"
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
from src.Recipes import load_profile, resolve_profile
from src.stats.tests.parametric.Average.Multivariate._common import (
    validate_alpha,
    validate_groups,
    validate_labels,
)
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class PostHocProfile:
    """독립 집단 평균 사후검정 입력 조건.

    Attributes:
        method (str): tukey/holm/bonferroni. 기본 tukey. Holm·Bonferroni는 모든 집단 쌍 t 검정 p-value를 보정.
        equal_var (bool): t 검정 등분산 여부. 기본 True. False는 Welch t; Tukey에는 False 사용 불가.
        alpha (float): 유의수준. 기본 0.05, 유한한 0<값<1. 기각 기준은 p_value<alpha.
    """

    method: str = "tukey"
    equal_var: bool = True
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class PostHocOutputDTO(OutputDTO):
    """독립 집단 평균 사후검정 검정 결과.

    Attributes:
        test_type (str): 실행한 독립 집단 평균 사후검정 이름.
        method (str): 다중검정 보정 방법.
        alpha (float): 전체 쌍 비교 family의 유의수준.
        equal_var (bool): 등분산 가정 여부.
        group_names (List[str]): 입력 집단 순서의 표시 이름.
        group_means (List[float]): 입력 순서의 집단 평균.
        group_ns (List[int]): 입력 순서의 집단 표본 수.
        comparisons (pd.DataFrame): 모든 쌍의 group1/group2, mean_diff(집단2-집단1), statistic(t 또는 NaN), df(t 자유도 또는 NaN), p_raw(Tukey는 NaN), p_adjusted, ci_lower/ci_upper, reject_h0. Tukey·Bonferroni CI는 family-wise, Holm CI는 NaN.
        num_comparisons (int): 비교 쌍 수 k*(k-1)/2.
    """

    test_type: str
    method: str
    alpha: float
    equal_var: bool
    group_names: List[str]
    group_means: List[float]
    group_ns: List[int]
    comparisons: pd.DataFrame
    num_comparisons: int


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    groups: Any, profile: PostHocProfile, group_names: Optional[List[str]]
) -> Any:
    """사후검정의 입력과 보정 조건을 검증합니다.

    Args:
        groups: 최소 두 집단의 유한한 수치형 1차원 배열 목록. 각 집단 최소 2행.
        profile: PostHocProfile.
        group_names: 선택적 고유한 문자열 집단 이름. None이면 Group0 등의 이름.
    Returns:
        Any: (집단 배열 목록, 입력 순서 집단 이름).
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·유의수준·방법·분산·이름 오류.
    """
    if not isinstance(profile, PostHocProfile):
        raise TypeError("PostHocProfile이 필요합니다.")
    validate_alpha(profile.alpha)
    if profile.method not in ("tukey", "holm", "bonferroni") or not isinstance(
        profile.equal_var, bool
    ):
        raise ValueError(
            "method는 tukey/holm/bonferroni, equal_var는 bool이어야 합니다."
        )
    if profile.method == "tukey" and not profile.equal_var:
        raise ValueError(
            "Tukey HSD는 등분산을 가정합니다. 이분산은 holm/bonferroni와 equal_var=False를 사용하세요."
        )
    arrays = validate_groups(groups)
    if any(g.var(ddof=1) <= 0 for g in arrays):
        raise ValueError("모든 집단에 양의 표본 분산이 필요합니다.")
    names = (
        list(group_names)
        if group_names is not None
        else ["Group" + str(i) for i in range(len(arrays))]
    )
    if (
        len(names) != len(arrays)
        or any(not isinstance(n, str) or not n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "group_names는 집단 수와 같은 고유한 비어 있지 않은 문자열 목록이어야 합니다."
        )
    return arrays, names


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_posthoc(
    groups: List[np.ndarray], names: List[str], profile: PostHocProfile
) -> PostHocOutputDTO:
    """전체 집단 쌍의 평균 비교 및 family 보정을 수행합니다.

    Args:
        groups: 검증된 집단 배열.
        names: 입력 순서의 고유한 집단 이름.
        profile: 검정 방법·분산·유의수준.
    Returns:
        PostHocOutputDTO: 쌍별 비교표와 집단 요약.
    Raises:
        ValueError: 유한하지 않은 통계량 또는 p-value.
    Notes:
        Holm·Bonferroni는 쌍별 Student/Welch t를 사용합니다. Tukey의 pooled 오차항과 다릅니다.
        Holm 조정 신뢰구간은 계산하지 않으며 NaN(JSON에서는 null)으로 표시합니다.
    """
    pairs = list(combinations(range(len(groups)), 2))
    m = len(pairs)
    rows = []
    if profile.method == "tukey":
        # 숫자 집단 코드는 입력 순서를 보존합니다. 문자열 정렬 때문에 부호가 바뀌지 않습니다.
        labels = np.concatenate([np.repeat(i, len(g)) for i, g in enumerate(groups)])
        result = pairwise_tukeyhsd(np.concatenate(groups), labels, alpha=profile.alpha)
        for k, (i, j) in enumerate(pairs):
            rows.append(
                [
                    names[i],
                    names[j],
                    float(result.meandiffs[k]),
                    np.nan,
                    np.nan,
                    np.nan,
                    float(result.pvalues[k]),
                    float(result.confint[k, 0]),
                    float(result.confint[k, 1]),
                    bool(result.reject[k]),
                ]
            )
    else:
        raw = []
        for i, j in pairs:
            a, b = groups[i], groups[j]
            na, nb = len(a), len(b)
            va, vb = a.var(ddof=1), b.var(ddof=1)
            diff = float(b.mean() - a.mean())
            if profile.equal_var:
                df = float(na + nb - 2)
                variance = ((na - 1) * va + (nb - 1) * vb) / df
                se = float(np.sqrt(variance * (1 / na + 1 / nb)))
            else:
                se = float(np.sqrt(va / na + vb / nb))
                df = float(
                    (va / na + vb / nb) ** 2
                    / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
                )
            statistic = diff / se
            p = float(2 * stats.t.sf(abs(statistic), df))
            raw.append(p)
            if profile.method == "bonferroni":
                margin = float(stats.t.ppf(1 - profile.alpha / (2 * m), df) * se)
                lo, hi = diff - margin, diff + margin
            else:
                lo, hi = np.nan, np.nan
            rows.append(
                [names[i], names[j], diff, statistic, df, p, np.nan, lo, hi, False]
            )
        reject, adjusted, _, _ = multipletests(
            raw, alpha=profile.alpha, method=profile.method
        )
        for k in range(m):
            rows[k][6], rows[k][9] = float(adjusted[k]), bool(reject[k])
    table = pd.DataFrame(
        rows,
        columns=[
            "group1",
            "group2",
            "mean_diff",
            "statistic",
            "df",
            "p_raw",
            "p_adjusted",
            "ci_lower",
            "ci_upper",
            "reject_h0",
        ],
    )
    if not np.isfinite(table[["mean_diff", "p_adjusted"]].to_numpy()).all():
        raise ValueError("사후검정 결과가 유한하지 않습니다.")
    return PostHocOutputDTO(
        "Independent-group post-hoc",
        profile.method,
        profile.alpha,
        profile.equal_var,
        names,
        [float(g.mean()) for g in groups],
        [len(g) for g in groups],
        table,
        m,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_posthoc(
    groups: Any,
    profile: Optional[PostHocProfile] = None,
    group_names: Optional[List[str]] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> PostHocOutputDTO:
    """한 반응변수의 모든 독립 집단 쌍을 비교합니다.

    Args:
        groups: 집단별 1차원 수치 배열 목록. 반복측정·대응 자료는 지원하지 않음.
        profile: None이면 posthoc.json에서 구성.
        group_names: 입력 순서의 집단 표시 이름. None이면 자동 이름.
        recipe_path: 사용자 JSON 경로. None이면 src/Recipes의 모델별 기본 파일.
        overrides: Profile이 없을 때 레시피 위에 적용할 명시적 설정 사전.
    Returns:
        PostHocOutputDTO: 평균 차이·조정 p-value·선택적 CI·기각 판단 비교표.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·분산·이름 또는 수치 계산 오류.
        FileNotFoundError/OSError: 레시피 파일 누락 또는 읽기 실패.
    Notes:
        전체 ANOVA의 유의성에 따라 자동으로 차단하지 않습니다. 전체 검정 결과와 함께 보고하세요.
        이원 ANOVA 교호작용의 단순효과/조정 주변평균 검정은 이 함수의 범위가 아닙니다.
        MANOVA 후 종속변수별 반복 비교는 이 함수의 쌍별 보정 외에 추가 보정이 필요합니다.
    """
    profile = resolve_profile(
        "posthoc", profile, recipe_path=recipe_path, overrides=overrides
    )
    arrays, names = _validate_inputs(groups, profile, group_names)
    return _run_posthoc(arrays, names, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    groups = [[1, 2, 3, 4], [4, 5, 6, 8], [8, 9, 10, 12]]
    for method in ("tukey", "holm", "bonferroni"):
        result = run_posthoc(
            groups,
            load_profile("posthoc", overrides={"method": method}),
            ["A", "B", "C"],
        )
        print(result.comparisons)
        print(result.to_json()[:200])
