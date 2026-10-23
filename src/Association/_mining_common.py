"""
================================================================================
_mining_common.py - 새 빈발 패턴 알고리즘의 거래 검증·결과 테이블·규칙 생성 계약.
================================================================================
설명:
    - 새 알고리즘 파일의 공통 검증·학습·자료 로딩 계약을 제공합니다.
분석 가정:
    - 입력 정의역·학습 전용 전처리·거래 분모와 알고리즘별 한도를 유지합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    별도 알고리즘의 공통 계약 추가
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
from itertools import combinations
import math
import numpy as np
import pandas as pd
from src.MachineLearning.analysis_common import positive_int, finite_number


def validate_transactions(transactions, profile):
    """빈 거래를 분모에 유지하고 중복 항목을 제거합니다.

    Args:
        transactions: 문자열 목록/tuple/set의 비어 있지 않은 거래 목록.
        profile: support/confidence/lift 및 탐색 한도를 가진 전용 Profile.
    Returns:
        list: 문자열 집합 거래들.
    Raises:
        ValueError/TypeError: 자료·임계값·한도 오류.
    """
    for name in ("min_support", "min_confidence", "min_lift"):
        finite_number(getattr(profile, name), name)
    if (
        not 0 < profile.min_support <= 1
        or not 0 <= profile.min_confidence <= 1
        or profile.min_lift < 0
    ):
        raise ValueError("support는 (0,1], confidence는 [0,1], lift는 0 이상입니다.")
    for name in ("max_length", "max_itemsets", "max_rules", "max_rule_candidates"):
        positive_int(getattr(profile, name), name)
    if not isinstance(transactions, (list, tuple)) or not transactions:
        raise ValueError("비어 있지 않은 거래 목록이 필요합니다.")
    result = []
    for transaction in transactions:
        if not isinstance(transaction, (list, tuple, set, frozenset)):
            raise ValueError("각 거래는 문자열 항목 목록/집합입니다.")
        if any(not isinstance(item, str) or not item.strip() for item in transaction):
            raise ValueError("항목은 비어 있지 않은 문자열입니다.")
        result.append(set(transaction))
    return result


def minimum_count(support, n):
    """부동소수점 곱셈 대신 기존 count/n 비교와 같은 정수 경계를 찾습니다.

    Args:
        support: 최소 지지도.
        n: 빈 거래 포함 전체 거래 수.
    Returns:
        int: 최소 허용 거래 수.
    Raises:
        ValueError: 잘못된 수치.
    """
    count = max(1, int(math.ceil(support * n)))
    while count > 1 and (count - 1) / n >= support:
        count -= 1
    while count / n < support:
        count += 1
    return count


def build_tables(counts, n, profile):
    """전수 빈발 항목집합의 지지도에서 동일한 규칙 지표를 계산합니다.

    Args:
        counts: 정렬된 항목 tuple → 거래 수 사전.
        n: 빈 거래 포함 전체 거래 수.
        profile: 규칙 임계값·출력/후보 한도.
    Returns:
        tuple: itemsets DataFrame, rules DataFrame, 검사한 규칙 후보 수.
    Raises:
        ValueError: 후보·규칙 한도 초과; 부분 결과를 반환하지 않음.
    """
    ordered = sorted(counts, key=lambda items: (len(items), items))
    rows = [
        dict(
            items=items,
            count=counts[items],
            support=counts[items] / n,
            length=len(items),
        )
        for items in ordered
    ]
    rules = []
    evaluated = 0
    for items in ordered:
        possible = (1 << len(items)) - 2
        if evaluated + possible > profile.max_rule_candidates:
            raise ValueError("규칙 후보 한도를 초과했습니다.")
        evaluated += possible
        support = counts[items] / n
        for size in range(1, len(items)):
            for left in combinations(items, size):
                right = tuple(item for item in items if item not in left)
                antecedent_support, consequent_support = (
                    counts[left] / n,
                    counts[right] / n,
                )
                confidence = support / antecedent_support
                lift = confidence / consequent_support
                if confidence >= profile.min_confidence and lift >= profile.min_lift:
                    if len(rules) >= profile.max_rules:
                        raise ValueError("연관규칙 출력 한도를 초과했습니다.")
                    conviction = (
                        (1 - consequent_support) / (1 - confidence)
                        if confidence < 1
                        else np.inf
                    )
                    rules.append(
                        dict(
                            antecedents=left,
                            consequents=right,
                            support=support,
                            confidence=confidence,
                            lift=lift,
                            leverage=support - antecedent_support * consequent_support,
                            conviction=conviction,
                        )
                    )
    return (
        pd.DataFrame(rows, columns=["items", "count", "support", "length"]),
        pd.DataFrame(
            rules,
            columns=[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ],
        ),
        evaluated,
    )
