"""
================================================================================
fp_growth.py - FPGrowth 빈발 항목집합·연관규칙
================================================================================
설명:
    - FPGrowth 빈발 항목집합·연관규칙을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 각 거래의 중복 항목은 한 번만 집계합니다. 연관은 인과 관계를 뜻하지 않습니다.
    - 결측값과 범주형 인코딩은 실행 전에 처리합니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.0.0    모델별 레시피·전체 명세·실행 예제 작성
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
from typing import Any, Dict, Optional
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.MachineLearning.analysis_common import positive_int
from ._mining_common import validate_transactions, minimum_count, build_tables


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class FPGrowthProfile:
    """FPGrowth 탐색과 규칙 설정.

    Attributes:
        min_support (float): 최소 지지도 (0,1]; 기본 0.2, 빈 거래 포함 분모.
        min_confidence (float): 최소 신뢰도 [0,1]; 기본 0.5.
        min_lift (float): 유한한 최소 lift, 0 이상; 기본 1.
        max_length (int): 최대 패턴 길이, 양수; 기본 3.
        max_itemsets (int): 빈발 집합 출력 한도; 초과 시 오류, 기본 100000.
        max_rules (int): 규칙 출력 한도; 초과 시 오류, 기본 100000.
        max_rule_candidates (int): 전체 규칙 후보 검사 한도; 기본 100000.
        max_nodes (int): 조건부 트리 포함 생성 노드 한도; 기본 100000.
    """

    min_support: float = 0.2
    min_confidence: float = 0.5
    min_lift: float = 1.0
    max_length: int = 3
    max_itemsets: int = 100000
    max_rules: int = 100000
    max_rule_candidates: int = 100000
    max_nodes: int = 100000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class FPGrowthOutputDTO(OutputDTO):
    """FPGrowth 빈발 집합·규칙 결과.

    Attributes:
        algorithm (str): 실행 알고리즘 이름.
        n_transactions (int): 빈 거래 포함 분모.
        itemsets (pd.DataFrame): items(tuple), count, support, length; 길이/항목 사전 순서.
        rules (pd.DataFrame): antecedents, consequents, support, confidence, lift, leverage, conviction.
        diagnostics (Dict[str, int]): 빈발 집합·규칙 후보 및 tree_nodes 수.
    """

    algorithm: str
    n_transactions: int
    itemsets: pd.DataFrame
    rules: pd.DataFrame
    diagnostics: Dict[str, int]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_inputs(transactions, profile):
    """전용 Profile과 문자열 거래를 검증합니다.

    Args:
        transactions: 거래별 문자열 항목 목록.
        profile: FPGrowthProfile.
    Returns:
        list: 거래별 고유 항목 집합.
    Raises:
        TypeError/ValueError: Profile·자료·임계값·한도 오류.
    """
    if not isinstance(profile, FPGrowthProfile):
        raise TypeError("FPGrowthProfile이 필요합니다.")
    positive_int(profile.max_nodes, "max_nodes")
    return validate_transactions(transactions, profile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


class _FPNode:
    """FP-tree의 항목 노드.

    Attributes:
        item (object): 문자열 항목; 루트는 None.
        count (int): 해당 경로의 거래 가중치 합.
        parent (object): 부모 노드 또는 None.
        children (dict): 항목별 자식 노드.
    """

    def __init__(self, item, parent):
        """비어 있는 FP 노드를 생성합니다.

        Args:
            item: 항목 또는 루트 None.
            parent: 부모 노드 또는 None.
        Returns:
            None: 새 노드.
        Raises:
            None: 검증한 항목에서만 호출.
        """
        self.item = item
        self.parent = parent
        self.count = 0
        self.children = {}


def _build_tree(paths, threshold, budget, profile):
    """가중 거래의 빈발 항목을 빈도 순으로 삽입한 FP-tree를 만듭니다.

    Args:
        paths: (고유 항목 iterable, 거래 가중치) 목록.
        threshold: 전체 원 거래 기준 최소 도수.
        budget: 모든 조건부 트리의 누적 노드 수.
        profile: FP-tree 노드 한도 설정.
    Returns:
        tuple: header 항목별 노드 목록, 항목별 가중 빈도.
    Raises:
        ValueError: 노드 한도 초과; 부분 결과 반환하지 않음.
    """
    counts = {}
    for items, weight in paths:
        for item in items:
            counts[item] = counts.get(item, 0) + weight
    counts = {item: count for item, count in counts.items() if count >= threshold}
    root = _FPNode(None, None)
    headers = {item: [] for item in counts}
    for items, weight in paths:
        ordered = sorted(
            (item for item in items if item in counts),
            key=lambda item: (-counts[item], item),
        )
        node = root
        for item in ordered:
            if item not in node.children:
                budget[0] += 1
                if budget[0] > profile.max_nodes:
                    raise ValueError("FP-tree 누적 노드 한도 초과")
                child = _FPNode(item, node)
                node.children[item] = child
                headers[item].append(child)
            node = node.children[item]
            node.count += weight
    return headers, counts


def _mine_patterns(transactions, profile):
    """조건부 패턴 기반 FP-tree 성장으로 빈발 집합을 탐색합니다.

    Args:
        transactions: 중복 없는 문자열 거래.
        profile: FP-Growth 조건.
    Returns:
        tuple: 항목 tuple별 도수와 누적 생성 노드 수.
    Raises:
        ValueError: 노드·항목집합 한도 초과.
    """
    threshold = minimum_count(profile.min_support, len(transactions))
    budget = [0]
    headers, counts = _build_tree(
        [(items, 1) for items in transactions], threshold, budget, profile
    )
    stack = [((), headers, counts)]
    patterns = {}
    while stack:
        prefix, headers, counts = stack.pop()
        for item in sorted(counts, key=lambda item: (counts[item], item)):
            pattern = tuple(sorted(prefix + (item,)))
            if len(patterns) >= profile.max_itemsets:
                raise ValueError("빈발 항목집합 한도 초과")
            patterns[pattern] = counts[item]
            if len(pattern) >= profile.max_length:
                continue
            paths = []
            for node in headers[item]:
                items = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    items.append(parent.item)
                    parent = parent.parent
                if items:
                    paths.append((items, node.count))
            if paths:
                conditional, conditional_counts = _build_tree(
                    paths, threshold, budget, profile
                )
                if conditional_counts:
                    stack.append((prefix + (item,), conditional, conditional_counts))
    return patterns, budget[0]


def _run_model(transactions, profile):
    """전용 탐색과 공통 규칙 계산을 연결합니다.

    Args:
        transactions: 검증한 고유 항목 거래.
        profile: 전용 설정.
    Returns:
        FPGrowthOutputDTO: 완전한 패턴·규칙 결과.
    Raises:
        ValueError: 탐색·결과 한도 초과.
    """
    counts, work = _mine_patterns(transactions, profile)
    itemsets, rules, rule_candidates = build_tables(counts, len(transactions), profile)
    return FPGrowthOutputDTO(
        "fp-growth",
        len(transactions),
        itemsets,
        rules,
        {
            "n_itemsets": len(counts),
            "rule_candidates": rule_candidates,
            "tree_nodes": work,
        },
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_fp_growth(
    transactions: Any,
    profile: Optional[FPGrowthProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> FPGrowthOutputDTO:
    """FPGrowth를 전용 JSON/Profile로 실행합니다.

    Args:
        transactions: 문자열 항목 집합/목록 거래들. 빈 거래는 분모에 포함.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: Profile 생략 시 명시적 설정 재정의.
    Returns:
        FPGrowthOutputDTO: 순서가 재현되는 수치·표 결과.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·한도·레시피 오류.
    """
    profile = resolve_profile(
        "fp-growth", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs(transactions, profile)
    return _run_model(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_fp_growth(
        [["bread", "milk"], ["bread", "butter"], ["bread", "milk", "butter"], []],
        overrides={"min_support": 0.25},
    )
    print(result.to_json(indent=2))
