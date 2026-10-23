"""
================================================================================
eclat.py - Eclat 빈발 항목집합·연관규칙
================================================================================
설명:
    - Eclat 빈발 항목집합·연관규칙을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
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
class EclatProfile:
    """Eclat 탐색과 규칙 설정.

    Attributes:
        min_support (float): 최소 지지도 (0,1]; 기본 0.2, 빈 거래 포함 분모.
        min_confidence (float): 최소 신뢰도 [0,1]; 기본 0.5.
        min_lift (float): 유한한 최소 lift, 0 이상; 기본 1.
        max_length (int): 최대 패턴 길이, 양수; 기본 3.
        max_itemsets (int): 빈발 집합 출력 한도; 초과 시 오류, 기본 100000.
        max_rules (int): 규칙 출력 한도; 초과 시 오류, 기본 100000.
        max_rule_candidates (int): 전체 규칙 후보 검사 한도; 기본 100000.
        max_intersections (int): TID 교집합 검사 한도; 기본 100000.
    """

    min_support: float = 0.2
    min_confidence: float = 0.5
    min_lift: float = 1.0
    max_length: int = 3
    max_itemsets: int = 100000
    max_rules: int = 100000
    max_rule_candidates: int = 100000
    max_intersections: int = 100000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class EclatOutputDTO(OutputDTO):
    """Eclat 빈발 집합·규칙 결과.

    Attributes:
        algorithm (str): 실행 알고리즘 이름.
        n_transactions (int): 빈 거래 포함 분모.
        itemsets (pd.DataFrame): items(tuple), count, support, length; 길이/항목 사전 순서.
        rules (pd.DataFrame): antecedents, consequents, support, confidence, lift, leverage, conviction.
        diagnostics (Dict[str, int]): 빈발 집합·규칙 후보 및 intersections 수.
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
        profile: EclatProfile.
    Returns:
        list: 거래별 고유 항목 집합.
    Raises:
        TypeError/ValueError: Profile·자료·임계값·한도 오류.
    """
    if not isinstance(profile, EclatProfile):
        raise TypeError("EclatProfile이 필요합니다.")
    positive_int(profile.max_intersections, "max_intersections")
    return validate_transactions(transactions, profile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _mine_patterns(transactions, profile):
    """수직 TID 집합을 교집합하여 빈발 항목집합을 탐색합니다.

    Args:
        transactions: 중복 없는 문자열 거래.
        profile: Eclat 조건과 교집합 한도.
    Returns:
        tuple: 항목 tuple별 도수와 교집합 검사 수.
    Raises:
        ValueError: 항목집합·교집합 한도 초과.
    """
    vertical = {}
    for index, items in enumerate(transactions):
        for item in items:
            vertical.setdefault(item, set()).add(index)
    threshold = minimum_count(profile.min_support, len(transactions))
    suffixes = [
        (item, vertical[item])
        for item in sorted(vertical)
        if len(vertical[item]) >= threshold
    ]
    stack = [((), suffixes)]
    patterns = {}
    intersections = 0
    while stack:
        prefix, suffixes = stack.pop()
        for index, (item, tids) in enumerate(suffixes):
            pattern = prefix + (item,)
            if len(patterns) >= profile.max_itemsets:
                raise ValueError("빈발 항목집합 한도 초과")
            patterns[pattern] = len(tids)
            if len(pattern) >= profile.max_length:
                continue
            children = []
            for other, other_tids in suffixes[index + 1 :]:
                intersections += 1
                if intersections > profile.max_intersections:
                    raise ValueError("Eclat 교집합 한도 초과")
                common = tids & other_tids
                if len(common) >= threshold:
                    children.append((other, common))
            if children:
                stack.append((pattern, children))
    return patterns, intersections


def _run_model(transactions, profile):
    """전용 탐색과 공통 규칙 계산을 연결합니다.

    Args:
        transactions: 검증한 고유 항목 거래.
        profile: 전용 설정.
    Returns:
        EclatOutputDTO: 완전한 패턴·규칙 결과.
    Raises:
        ValueError: 탐색·결과 한도 초과.
    """
    counts, work = _mine_patterns(transactions, profile)
    itemsets, rules, rule_candidates = build_tables(counts, len(transactions), profile)
    return EclatOutputDTO(
        "eclat",
        len(transactions),
        itemsets,
        rules,
        {
            "n_itemsets": len(counts),
            "rule_candidates": rule_candidates,
            "intersections": work,
        },
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_eclat(
    transactions: Any,
    profile: Optional[EclatProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> EclatOutputDTO:
    """Eclat를 전용 JSON/Profile로 실행합니다.

    Args:
        transactions: 문자열 항목 집합/목록 거래들. 빈 거래는 분모에 포함.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로.
        overrides: Profile 생략 시 명시적 설정 재정의.
    Returns:
        EclatOutputDTO: 순서가 재현되는 수치·표 결과.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·한도·레시피 오류.
    """
    profile = resolve_profile(
        "eclat", profile, recipe_path=recipe_path, overrides=overrides
    )
    data = _validate_inputs(transactions, profile)
    return _run_model(data, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_eclat(
        [["bread", "milk"], ["bread", "butter"], ["bread", "milk", "butter"], []],
        overrides={"min_support": 0.25},
    )
    print(result.to_json(indent=2))
