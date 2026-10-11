"""
================================================================================
posthoc.py - Tukey HSD·Games-Howell·보정된 쌍별 t 사후검정
================================================================================
설명:
    - 독립 집단의 모든 쌍을 비교하고 평균차(group2-group1), p-value 및 구간을 반환합니다.
    - 등분산은 Tukey, 이분산은 Games-Howell 또는 Welch 쌍별 t 검정을 지원합니다.
    - 쌍별 t는 Holm/Bonferroni 보정을 적용합니다. Games-Howell은 scipy studentized_range 사용.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-11
    - 버전: 1.0.0
변경 이력: 2026-10-11 v1.0.0 사후검정 3종·다중성·구간의 의미 명세 추가
사용 예제: python -m examples.multivariate_workflow
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, List
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PostHocProfile:
    """독립 집단 사후검정 조건.

    Attributes:
        method: tukey, games_howell, pairwise_t.
        equal_var: tukey는 True, games_howell은 False 필수. pairwise_t는 pooled/Welch 분기.
        correction: pairwise_t에서 holm 또는 bonferroni. 다른 방법은 자체 familywise 보정.
        alpha: 가족 단위 유의수준 (0,1).
    """
    method: str = "tukey"
    equal_var: bool = True
    correction: str = "holm"
    alpha: float = 0.05


@dataclass
class PostHocOutputDTO(OutputDTO):
    """모든 집단 쌍의 비교 결과.

    Attributes:
        method: 실행 기법 이름.
        table: group1/group2/mean_diff/statistic/df/p_raw/p_adjusted/ci_lower/ci_upper/reject_h0.
            평균차는 group2-group1. Tukey statistic은 NaN, Games-Howell statistic은 양의 q,
            pairwise_t는 부호 있는 t. Holm의 구간은 계산하지 않아 NaN.
            Tukey/Games-Howell p_raw는 자체 보정된 검정이라 NaN.
        groups: 최초 등장 순서의 원래 집단 라벨 목록.
        alpha: 가족 단위 유의수준.
        correction: tukey, games_howell, holm 또는 bonferroni.
    """
    method: str
    table: pd.DataFrame
    groups: List[Any]
    alpha: float
    correction: str


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
PostHocResult = PostHocOutputDTO


def run_posthoc(y: Any, groups: Any, profile: PostHocProfile = None) -> PostHocOutputDTO:
    """독립 집단 전체 쌍에 대해 사후검정과 다중성 보정을 수행합니다.

    Args:
        y: 결측/무한대 없는 1차원 수치 반응.
        groups: 같은 길이의 결측 없는 이산 집단 라벨. 각 집단 최소 두 관측.
        profile: 기법·등분산·다중성·유의수준 설정. None이면 Tukey HSD.
    Returns:
        PostHocOutputDTO: 원래 라벨의 모든 쌍 비교 DataFrame. reject_h0는 p_adjusted < alpha.
    Raises:
        ValueError: 입력 길이/집단 수/표본 수/분산 부족 또는 기법·가정 불일치.
        ImportError: statsmodels 미설치. Games-Howell은 scipy >=1.7 필요.
    Notes:
        Tukey HSD는 독립성·정규성·등분산을, Games-Howell은 독립성 및 집단별 정규성을
        가정합니다. 후자는 이분산·불균형을 허용합니다. Games-Howell 구간은 근사 simultaneous 구간.
        pairwise_t의 Bonferroni 구간은 가족 단위 보정 구간, Holm은 보정 p만 제공합니다.
        이 함수는 ANOVA 결과로 실행 여부를 자동 결정하지 않습니다. 사후 비교 계획을 명시하세요.
        반복측정/이원 ANOVA의 단순효과를 독립 일원 비교로 대체하지 않습니다.
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
    p = profile or PostHocProfile()
    if p.method not in ("tukey","games_howell","pairwise_t") or not isinstance(p.equal_var,bool) or not 0 < p.alpha < 1 or p.correction not in ("holm","bonferroni"):
        raise ValueError("method/equal_var/correction/alpha를 확인하세요.")
    if (p.method == "tukey" and not p.equal_var) or (p.method == "games_howell" and p.equal_var):
        raise ValueError("Tukey는 equal_var=True, Games-Howell은 equal_var=False로 지정하세요.")
    values = np.asarray(y,dtype=float)
    labels = np.asarray(groups)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels) or not np.isfinite(values).all() or pd.isna(labels).any():
        raise ValueError("반응과 집단은 같은 길이의 결측 없는 1차원 데이터여야 합니다.")
    codes, levels = pd.factorize(labels,sort=False)
    samples = [values[codes==i] for i in range(len(levels))]
    if len(samples) < 2 or any(len(g)<2 for g in samples):
        raise ValueError("각각 최소 두 관측을 갖는 두 집단 이상이 필요합니다.")
    ns=np.asarray([len(g) for g in samples]);means=np.asarray([np.mean(g) for g in samples]);variances=np.asarray([np.var(g,ddof=1) for g in samples])
    if not np.isfinite(variances).all() or (p.method != "tukey" and (variances <= 0).any()) or np.sum((ns-1)*variances) <= 0:
        raise ValueError("검정에 필요한 양의 유한 분산이 부족합니다.")
    pairs=list(combinations(range(len(samples)),2));records=[]
    if p.method == "tukey":
        # 숫자 위치로 검정하므로 문자열 변환에 따른 원래 라벨 충돌을 피함.
        result=pairwise_tukeyhsd(values,codes,alpha=p.alpha)
        for j,(a,b) in enumerate(pairs):
            records.append(dict(group1=levels[a],group2=levels[b],mean_diff=float(result.meandiffs[j]),
                statistic=np.nan,df=float(len(values)-len(samples)),p_raw=np.nan,
                p_adjusted=float(result.pvalues[j]),ci_lower=float(result.confint[j,0]),
                ci_upper=float(result.confint[j,1]),reject_h0=bool(result.pvalues[j]<p.alpha)))
    else:
        for a,b in pairs:
            diff=float(means[b]-means[a])
            if p.method == "games_howell" or not p.equal_var:
                va,vb=variances[a]/ns[a],variances[b]/ns[b]
                se=np.sqrt(va+vb)
                df=float((va+vb)**2/(va**2/(ns[a]-1)+vb**2/(ns[b]-1)))
            else:
                df=float(ns[a]+ns[b]-2)
                pooled=((ns[a]-1)*variances[a]+(ns[b]-1)*variances[b])/df
                se=np.sqrt(pooled*(1/ns[a]+1/ns[b]))
            if p.method == "games_howell":
                q=float(abs(diff)/se*np.sqrt(2))
                adjusted=float(stats.studentized_range.sf(q,len(samples),df))
                margin=float(stats.studentized_range.ppf(1-p.alpha,len(samples),df)*se/np.sqrt(2))
                raw=np.nan;statistic=q;lower=diff-margin;upper=diff+margin
            else:
                statistic=float(diff/se);raw=float(2*stats.t.sf(abs(statistic),df));adjusted=np.nan
                if p.correction == "bonferroni":
                    margin=float(stats.t.ppf(1-p.alpha/(2*len(pairs)),df)*se)
                    lower,upper=diff-margin,diff+margin
                else:lower=upper=np.nan
            records.append(dict(group1=levels[a],group2=levels[b],mean_diff=diff,statistic=statistic,
                df=df,p_raw=raw,p_adjusted=adjusted,ci_lower=lower,ci_upper=upper,reject_h0=False))
        if p.method == "pairwise_t":
            adjusted=multipletests([r['p_raw'] for r in records],alpha=p.alpha,method=p.correction)[1]
            for row,value in zip(records,adjusted):row['p_adjusted']=float(value)
        for row in records:row['reject_h0']=bool(row['p_adjusted']<p.alpha)
    table=pd.DataFrame(records)
    if not np.isfinite(table[['mean_diff','df','p_adjusted']].values).all():
        raise ValueError("유한한 사후검정 결과를 계산할 수 없습니다.")
    correction=p.correction if p.method == 'pairwise_t' else p.method
    return PostHocOutputDTO(p.method,table,list(levels),p.alpha,correction)
