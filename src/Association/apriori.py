"""
================================================================================
apriori.py - 빈발 항목집합과 연관규칙
================================================================================
설명: 연관분석 PDF의 TransactionEncoder → Apriori → association_rules를 모듈화합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 거래 인코딩과 규칙 결과 명세 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

import inspect
from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd


@dataclass
class AssociationProfile:
    """Apriori/연관규칙 조건.

    Attributes:
        min_support: 빈발 항목집합의 최소 지지도 (0 < 값 <= 1).
        metric: 규칙 필터 기준 confidence, lift, support.
        min_threshold: 기준 지표의 최소 값 (0 이상).
        max_len: 항목집합 최대 길이. None이면 제한 없음.
    """
    min_support: float = 0.2
    metric: str = "confidence"
    min_threshold: float = 0.5
    max_len: Optional[int] = None


@dataclass
class AssociationOutputDTO(OutputDTO):
    """연관분석 결과.

    Attributes:
        encoder: 적합 TransactionEncoder.
        encoded: 거래 x 항목 bool DataFrame. 한 거래의 중복 항목은 존재 여부로 처리.
        frequent_itemsets: support, itemsets를 가진 DataFrame.
        rules: antecedents/consequents/support/confidence/lift 등의 규칙 표.
            규칙이 없으면 빈 표. conviction은 완전 신뢰 규칙에서 inf일 수 있음.
    """
    encoder: Any
    encoded: pd.DataFrame
    frequent_itemsets: pd.DataFrame
    rules: pd.DataFrame


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
AssociationResult = AssociationOutputDTO


def run_association(transactions: Any, profile: Optional[AssociationProfile] = None) -> AssociationOutputDTO:
    """거래 목록으로 빈발 항목집합과 규칙을 추출합니다.

    Args:
        transactions: 문자열 항목 목록들의 목록. 빈 거래는 전체 거래 수에 포함됩니다.
        profile: 연관분석 설정.
    Returns:
        AssociationOutputDTO: 인코더, bool 거래표, 항목집합 및 규칙.
    Raises:
        ValueError: 비어 있는 전체 입력, 비문자 항목, 잘못된 임계값·지표.
        ImportError: mlxtend 미설치.

    Notes:
        연관규칙은 관측된 동시 발생 관계이며 인과관계를 의미하지 않습니다.
    """
    import numpy as np
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    p = profile if profile is not None else AssociationProfile()
    if not np.isfinite(p.min_support) or not 0 < p.min_support <= 1:
        raise ValueError("min_support는 0보다 크고 1 이하여야 합니다.")
    if p.metric not in ("confidence", "lift", "support") or not np.isfinite(p.min_threshold) or p.min_threshold < 0:
        raise ValueError("metric 또는 min_threshold가 유효하지 않습니다.")
    if p.metric != "lift" and p.min_threshold > 1:
        raise ValueError("confidence/support 임계값은 1 이하여야 합니다.")
    if p.max_len is not None and (not isinstance(p.max_len, int) or p.max_len < 1):
        raise ValueError("max_len은 양의 정수 또는 None이어야 합니다.")
    rows = []
    if isinstance(transactions, str):
        raise ValueError("거래 목록을 입력하세요.")
    for row in transactions:
        if isinstance(row, str):
            raise ValueError("각 거래는 문자열 하나가 아니라 항목 목록이어야 합니다.")
        items = list(row)
        if any(not isinstance(v, str) or not v for v in items):
            raise ValueError("항목은 비어 있지 않은 문자열이어야 합니다.")
        rows.append(items)
    if not rows or not any(rows):
        raise ValueError("최소 한 거래와 한 항목이 필요합니다.")
    encoder = TransactionEncoder()
    matrix = encoder.fit(rows).transform(rows)
    encoded = pd.DataFrame(matrix, columns=encoder.columns_)
    itemsets = apriori(encoded, min_support=p.min_support, use_colnames=True, max_len=p.max_len)
    columns = ["antecedents", "consequents", "antecedent support", "consequent support", "support",
               "confidence", "lift", "leverage", "conviction"]
    if itemsets.empty or not itemsets.itemsets.map(len).gt(1).any():
        rules = pd.DataFrame(columns=columns)
    else:
        kwargs = {"metric": p.metric, "min_threshold": p.min_threshold}
        if "num_itemsets" in inspect.signature(association_rules).parameters:
            kwargs["num_itemsets"] = len(rows)
        rules = association_rules(itemsets, **kwargs)
        rules = rules.sort_values([p.metric, "support"], ascending=False).reset_index(drop=True) if p.metric != "support" else rules.sort_values("support", ascending=False).reset_index(drop=True)
    return AssociationOutputDTO(encoder, encoded, itemsets, rules)


if __name__ == "__main__":
    print(run_association([["Milk", "Eggs"], ["Milk"], ["Eggs", "Bread"]]).rules)
