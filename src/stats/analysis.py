"""
================================================================================
analysis.py - 기술통계, 정규성·등분산·상관·범주형·분산 검정
================================================================================
설명:
    - 통계분석 및 카이제곱/F 분포 PDF의 절차를 수치형 실행기로 구현합니다.
    - reject_h0=False는 귀무가설이 참이라는 뜻이 아니라 기각 근거 부족입니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 통계 Profile/Result와 입력 검증 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
from scipy import stats


@dataclass
class StatisticalProfile:
    """통계 검정 조건.

    Attributes:
        test: shapiro, normaltest, levene, bartlett, pearson, spearman, kendall,
            paired_t, wilcoxon, mannwhitney, kruskal, friedman,
            chi2_independence, chi2_goodness, variance_chi2, variance_f.
        alpha: 유의수준 (0 < alpha < 1).
        alternative: two-sided, less, greater. 방향 지원 검정에만 사용.
        null_value: variance_chi2는 귀무 모분산, variance_f는 귀무 분산비.
        correction: chi2_independence에서 2x2 Yates 보정 여부.
        center: Levene 중심값 (median, mean, trimmed).
    """
    test: str = "shapiro"
    alpha: float = 0.05
    alternative: str = "two-sided"
    null_value: float = 1.0
    correction: bool = True
    center: str = "median"


@dataclass
class StatisticalOutputDTO(OutputDTO):
    """통계 검정 결과.

    Attributes:
        test_type: 수행 검정 이름.
        statistic: 검정통계량 (상관 검정은 상관계수).
        p_value: 지정한 대립가설의 p-value.
        reject_h0: p_value < alpha 여부. NaN 결과는 반환하지 않음.
        details: 표본 수, 자유도, 기대도수, 귀무가설 등 검정별 보조 정보.
    """
    test_type: str
    statistic: float
    p_value: float
    reject_h0: bool
    details: Dict[str, Any] = field(default_factory=dict)


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
StatisticalResult = StatisticalOutputDTO


@dataclass
class DescriptiveOutputDTO(OutputDTO):
    """기술통계 반환 DTO.

    Attributes:
        n: 표본 수.
        mean, median, modes: 평균·중앙값·최빈값 배열.
        sample_variance, sample_std: ddof=1 표본 분산/표준편차.
        q1, q3, iqr, quartile_deviation: 사분위수·IQR·반사분위수.
        range: 최대값과 최소값 차이.
        skewness, excess_kurtosis: 왜도·초과첨도. 정의되지 않으면 None/NaN.
        outlier_indices: 원본 표본 순서의 IQR 이상치 위치.
    """
    n: int
    mean: float
    median: float
    modes: np.ndarray
    sample_variance: float
    sample_std: float
    q1: float
    q3: float
    iqr: float
    quartile_deviation: float
    range: float
    skewness: Optional[float]
    excess_kurtosis: Optional[float]
    outlier_indices: np.ndarray

def numeric_sample(data: Any, min_size: int = 2) -> np.ndarray:
    """유한한 1차원 수치 표본을 검증합니다.

    Args:
        data: 표본 시퀀스.
        min_size: 최소 표본 수.
    Returns:
        np.ndarray: 1차원 float 표본.
    Raises:
        ValueError: 차원/크기 오류 또는 NaN·무한대.
    """
    a = np.asarray(data, dtype=float)
    if a.ndim != 1 or len(a) < min_size or not np.isfinite(a).all():
        raise ValueError("표본은 최소 {}개의 유한한 1차원 수치여야 합니다.".format(min_size))
    return a


def validate_test_profile(profile: Any) -> None:
    """공통 유의수준과 검정 방향을 검증합니다.

    Args:
        profile: alpha 및 선택적 alternative 필드가 있는 Profile.
    Returns:
        None: 유효한 경우 정상 종료.
    Raises:
        ValueError: Profile 누락, 잘못된 alpha 또는 alternative.
    """
    if profile is None or not np.isfinite(profile.alpha) or not 0 < profile.alpha < 1:
        raise ValueError("profile.alpha는 0과 1 사이여야 합니다.")
    if getattr(profile, "alternative", "two-sided") not in ("two-sided", "less", "greater"):
        raise ValueError("alternative는 two-sided, less, greater 중 하나여야 합니다.")


def run_descriptive(data: Any, iqr_weight: float = 1.5) -> DescriptiveOutputDTO:
    """기술통계와 IQR 이상치 위치를 반환합니다.

    Args:
        data: 최소 2개의 유한한 1차원 수치 표본.
        iqr_weight: 이상치 경계의 IQR 배수 (0 이상).
    Returns:
        DescriptiveOutputDTO: n/mean/median/modes/sample_variance/sample_std/q1/q3/iqr/
            quartile_deviation/range/skewness/excess_kurtosis/outlier_indices.
            상수 표본의 왜도·첨도는 정의되지 않으므로 None.
    Raises:
        ValueError: 잘못된 표본 또는 iqr_weight.
    """
    a = numeric_sample(data)
    if not np.isfinite(iqr_weight) or iqr_weight < 0:
        raise ValueError("iqr_weight는 0 이상의 유한한 값이어야 합니다.")
    q1, q3 = np.quantile(a, [0.25, 0.75]); iqr = q3 - q1
    values, counts = np.unique(a, return_counts=True)
    return DescriptiveOutputDTO(**{"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "modes": values[counts == counts.max()], "sample_variance": float(a.var(ddof=1)),
            "sample_std": float(a.std(ddof=1)), "q1": float(q1), "q3": float(q3),
            "iqr": float(iqr), "quartile_deviation": float(iqr / 2), "range": float(np.ptp(a)),
            "skewness": float(stats.skew(a, bias=False)) if np.ptp(a) else None,
            "excess_kurtosis": float(stats.kurtosis(a, fisher=True, bias=False)) if np.ptp(a) else None,
            "outlier_indices": np.flatnonzero((a < q1 - iqr_weight * iqr) | (a > q3 + iqr_weight * iqr))})


def run_statistical_test(data: Any, profile: Optional[StatisticalProfile] = None,
                         other: Any = None, expected: Any = None) -> StatisticalOutputDTO:
    """검정 유형에 따라 표본·집단·도수 입력을 처리합니다.

    Args:
        data: 단일 검정은 1차원 표본; 두 표본 검정은 첫 표본; 집단 검정은 표본 목록;
            chi2_independence는 2차원 교차표, chi2_goodness는 1차원 관측도수.
        profile: 검정 이름과 조건. None이면 Shapiro 정규성 검정.
        other: 두 표본·상관·대응 검정의 두 번째 표본.
        expected: chi2_goodness의 기대도수 (관측도수와 합이 같아야 함). None이면 균등 도수.
    Returns:
        StatisticalOutputDTO: 유한한 통계량·p-value와 보조 정보.
    Raises:
        ValueError: 잘못된 입력/프로필, 상수·퇴화 표본 또는 정의 불가능한 검정.

    Notes:
        paired_t/wilcoxon/friedman 입력은 같은 관측 대상 순서로 정렬되어야 합니다.
        상관 검정은 scipy 1.7 호환성을 위해 양측만 지원합니다.
        분산 검정은 독립성과 정규성을 가정합니다. 기대도수가 작은 교차표는 해석에 주의합니다.
    """
    p = profile if profile is not None else StatisticalProfile()
    validate_test_profile(p)
    kind = p.test
    two_sided_only = ("shapiro", "normaltest", "levene", "bartlett", "pearson", "spearman",
                      "kendall", "kruskal", "friedman", "chi2_independence", "chi2_goodness")
    if kind in two_sided_only and p.alternative != "two-sided":
        raise ValueError("이 실행기는 해당 검정의 단측 대립가설을 지원하지 않습니다.")
    details = {}
    if kind in ("levene", "bartlett", "kruskal", "friedman"):
        groups = [numeric_sample(g) for g in data]
        if len(groups) < (3 if kind == "friedman" else 2):
            raise ValueError("검정을 위한 집단 수가 부족합니다.")
        if kind == "friedman" and len(set(map(len, groups))) != 1:
            raise ValueError("Friedman 검정은 동일 길이의 대응 표본이 필요합니다.")
        runner = {"levene": stats.levene, "bartlett": stats.bartlett,
                  "kruskal": stats.kruskal, "friedman": stats.friedmanchisquare}[kind]
        result = runner(*groups, **({"center": p.center} if kind == "levene" else {}))
        statistic, pvalue = float(result.statistic), float(result.pvalue)
        details["group_ns"] = [len(g) for g in groups]
    elif kind == "chi2_independence":
        table = np.asarray(data, dtype=float)
        if table.ndim != 2 or min(table.shape) < 2 or not np.isfinite(table).all() or (table < 0).any():
            raise ValueError("교차표는 최소 2x2의 유한한 비음수 도수여야 합니다.")
        if not np.equal(table, np.floor(table)).all():
            raise ValueError("교차표에는 실제 정수 도수를 입력하세요.")
        statistic, pvalue, df, exp = stats.chi2_contingency(table, correction=p.correction)
        details.update(df=df, expected=exp, expected_below_5=int((exp < 5).sum()))
    elif kind == "chi2_goodness":
        observed = numeric_sample(data)
        if (observed < 0).any() or observed.sum() <= 0 or not np.equal(observed, np.floor(observed)).all():
            raise ValueError("관측도수는 합이 양수인 비음수 정수여야 합니다.")
        exp = np.full(len(observed), observed.mean()) if expected is None else numeric_sample(expected)
        if len(exp) != len(observed) or (exp <= 0).any() or not np.isclose(exp.sum(), observed.sum()):
            raise ValueError("기대도수는 양수이고 관측도수와 길이·합이 같아야 합니다.")
        result = stats.chisquare(observed, f_exp=exp)
        statistic, pvalue = float(result.statistic), float(result.pvalue)
        details.update(df=len(observed)-1, expected=exp)
    else:
        a = numeric_sample(data, min_size=8 if kind == "normaltest" else 3 if kind == "shapiro" else 2)
        details["n"] = len(a)
        if kind in ("shapiro", "normaltest"):
            if not np.ptp(a):
                raise ValueError("상수 표본에는 정규성 검정을 적용할 수 없습니다.")
            result = getattr(stats, kind)(a)
        elif kind in ("paired_t", "wilcoxon", "mannwhitney", "pearson", "spearman", "kendall"):
            b = numeric_sample(other)
            if kind != "mannwhitney" and len(a) != len(b):
                raise ValueError("대응/상관 검정은 같은 길이의 표본이 필요합니다.")
            if kind in ("pearson", "spearman", "kendall"):
                if not np.ptp(a) or not np.ptp(b):
                    raise ValueError("상수 표본의 상관계수는 정의되지 않습니다.")
                result = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[kind](a, b)
            else:
                result = {"paired_t": stats.ttest_rel, "wilcoxon": stats.wilcoxon,
                          "mannwhitney": stats.mannwhitneyu}[kind](a, b, alternative=p.alternative)
            details["n_other"] = len(b)
        elif kind in ("variance_chi2", "variance_f"):
            if not np.isfinite(p.null_value) or p.null_value <= 0:
                raise ValueError("귀무 분산/분산비는 양수여야 합니다.")
            df1 = len(a) - 1
            if a.var(ddof=1) <= 0:
                raise ValueError("분산 검정은 양의 표본분산이 필요합니다.")
            if kind == "variance_chi2":
                statistic = df1 * a.var(ddof=1) / p.null_value
                cdf, sf = stats.chi2.cdf(statistic, df1), stats.chi2.sf(statistic, df1)
                details.update(df=df1, null_variance=p.null_value)
            else:
                b = numeric_sample(other); df2 = len(b) - 1
                if b.var(ddof=1) <= 0:
                    raise ValueError("두 번째 표본분산은 양수여야 합니다.")
                statistic = (a.var(ddof=1) / b.var(ddof=1)) / p.null_value
                cdf, sf = stats.f.cdf(statistic, df1, df2), stats.f.sf(statistic, df1, df2)
                details.update(df1=df1, df2=df2, null_ratio=p.null_value)
            pvalue = min(1.0, 2 * min(cdf, sf)) if p.alternative == "two-sided" else sf if p.alternative == "greater" else cdf
            return StatisticalOutputDTO(kind, float(statistic), float(pvalue), bool(pvalue < p.alpha), details)
        else:
            raise ValueError("지원하지 않는 통계 검정입니다: " + kind)
        statistic, pvalue = float(result[0]), float(result[1])
    if not np.isfinite(statistic) or not np.isfinite(pvalue):
        raise ValueError("퇴화 데이터로 인해 유한한 검정 결과를 계산할 수 없습니다.")
    return StatisticalOutputDTO(kind, float(statistic), float(pvalue), bool(pvalue < p.alpha), details)


if __name__ == "__main__":
    print(run_statistical_test([1, 2, 3, 4, 6]))
