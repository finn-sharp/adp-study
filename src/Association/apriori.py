"""
================================================================================
apriori.py - Apriori 빈발 항목집합·연관규칙
================================================================================
설명:
    - Apriori 빈발 항목집합·연관규칙을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.output import OutputDTO
from src.Recipes import load_profile, resolve_profile
from src.MachineLearning.common import validate_features, make_scaler
from src.MachineLearning.analysis_common import positive_int, finite_number, validate_scaling, feature_names



# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class AprioriProfile:
    """Apriori 빈발 항목집합·연관규칙 입력 조건.

    Attributes:
        min_support (float): 최소 지지도. 0 초과 1 이하. 기본 0.2.
        min_confidence (float): 최소 신뢰도. 0 이상 1 이하. 기본 0.5.
        min_lift (float): 최소 향상도. 유한한 0 이상. 기본 1.0.
        max_length (int): 최대 항목집합 길이. 양의 정수. 기본 3.
        max_rules (int): 연관규칙 출력 한도. 양의 정수. 초과 시 오류. 기본 100000.
        max_candidates (int): 전체 후보 평가 한도. 초과 시 일부 결과 반환 대신 오류. 기본 100000.
    """

    min_support: float = 0.2
    min_confidence: float = 0.5
    min_lift: float = 1.0
    max_length: int = 3
    max_rules: int = 100000
    max_candidates: int = 100000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class AprioriOutputDTO(OutputDTO):
    """Apriori 빈발 항목집합·연관규칙 결과. 배열은 원 입력 행 순서.

    Attributes:
        n_transactions (int): 전체 거래 수. 빈 거래도 분모에 포함.
        itemsets (pd.DataFrame): items(tuple), count, support, length 열. 지지도는 거래 비율.
        rules (pd.DataFrame): antecedents, consequents, support, confidence, lift, leverage, conviction 열.
    """

    n_transactions: int
    itemsets: pd.DataFrame
    rules: pd.DataFrame


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _transactions(transactions: Any) -> List[set]:
    """거래 자료를 중복 없는 문자열 항목 집합으로 정규화합니다.

    Args:
        transactions: 거래별 문자열 목록/tuple/set의 목록. 빈 거래는 허용.
    Returns:
        List[set]: 입력 거래 순서의 집합 목록.
    Raises:
        ValueError: 빈 전체 입력, 잘못된 거래 타입 또는 빈/비문자열 항목.
    """
    if not isinstance(transactions, (list, tuple)) or not transactions:
        raise ValueError("transactions는 비어 있지 않은 거래 목록이어야 합니다.")
    result = []
    for transaction in transactions:
        if not isinstance(transaction, (list, tuple, set, frozenset)):
            raise ValueError("각 거래는 문자열 항목 목록 또는 집합이어야 합니다.")
        if any(not isinstance(item, str) or not item.strip() for item in transaction):
            raise ValueError("항목은 비어 있지 않은 문자열이어야 합니다.")
        result.append(set(transaction))
    return result


def _validate_inputs(transactions: Any, profile: AprioriProfile) -> List[set]:
    """자료와 Profile 조건을 검증합니다.

    Args:
        transactions: 거래별 문자열 항목 목록.
        profile: AprioriProfile 조건.
    Returns:
        List[set]: 중복 없는 거래 목록.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, AprioriProfile):
        raise TypeError("AprioriProfile이 필요합니다.")
    finite_number(profile.min_support, "min_support", positive=True)
    finite_number(profile.min_confidence, "min_confidence")
    finite_number(profile.min_lift, "min_lift")
    positive_int(profile.max_length, "max_length")
    positive_int(profile.max_candidates, "max_candidates")
    positive_int(profile.max_rules, "max_rules")
    if profile.min_support > 1 or profile.min_confidence > 1: raise ValueError("support/confidence는 1 이하여야 합니다.")
    arr = _transactions(transactions)
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_model(transactions: Any, profile: AprioriProfile) -> AprioriOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        transactions: 문자열 거래 목록.
        profile: 검증된 AprioriProfile.
    Returns:
        AprioriOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    from itertools import combinations
    transactions = _transactions(transactions)
    items = sorted(set().union(*transactions))
    if len(items) > profile.max_candidates: raise ValueError("Apriori 후보 한도를 초과했습니다.")
    supports = {}
    rows = []
    level = [(item,) for item in items]
    evaluated = 0
    for length in range(1, profile.max_length+1):
        if not level: break
        evaluated += len(level)
        if evaluated > profile.max_candidates: raise ValueError("Apriori 후보 한도를 초과했습니다.")
        frequent = []
        for candidate in level:
            count = sum(set(candidate).issubset(transaction) for transaction in transactions)
            support = count / len(transactions)
            if support >= profile.min_support:
                supports[candidate] = support
                rows.append(dict(items=candidate, count=count, support=support, length=length))
                frequent.append(candidate)
        if length == profile.max_length: break
        candidates = set()
        previous = set(frequent)
        for i, left in enumerate(frequent):
            for right in frequent[i+1:]:
                if left[:-1] != right[:-1]: continue
                joined = tuple(sorted(set(left) | set(right)))
                if len(joined) == length+1 and all(part in previous for part in combinations(joined, length)):
                    candidates.add(joined)
                    if evaluated + len(candidates) > profile.max_candidates: raise ValueError("Apriori 후보 한도를 초과했습니다.")
        level = sorted(candidates)
    rules = []
    for itemset, support in supports.items():
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                consequent = tuple(item for item in itemset if item not in antecedent)
                confidence = support / supports[antecedent]
                lift = confidence / supports[consequent]
                if confidence >= profile.min_confidence and lift >= profile.min_lift:
                    if len(rules) >= profile.max_rules: raise ValueError("Apriori 규칙 한도를 초과했습니다.")
                    conviction = (1-supports[consequent])/(1-confidence) if confidence < 1 else np.inf
                    rules.append(dict(antecedents=antecedent, consequents=consequent, support=support,
                        confidence=confidence, lift=lift, leverage=support-supports[antecedent]*supports[consequent], conviction=conviction))
    return AprioriOutputDTO(len(transactions), pd.DataFrame(rows, columns=["items","count","support","length"]),
        pd.DataFrame(rules, columns=["antecedents","consequents","support","confidence","lift","leverage","conviction"]))


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_apriori(transactions: Any, profile: Optional[AprioriProfile] = None, *,
              recipe_path: Optional[str] = None,
              overrides: Optional[Dict[str, Any]] = None) -> AprioriOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        transactions: 거래별 문자열 항목 목록. 빈 거래는 분모에 포함하며 중복 항목은 한 번 집계.
        profile: 명시적 AprioriProfile. None이면 apriori.json을 읽음.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        AprioriOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile("apriori", profile, recipe_path=recipe_path, overrides=overrides)
    _validate_inputs(transactions, profile)
    return _run_model(transactions, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transactions = [["bread", "milk"], ["bread", "eggs"], ["milk", "eggs"], ["bread", "milk", "eggs"]]
    result = run_apriori(transactions, load_profile("apriori"))
    print(result.itemsets)
    print(result.rules)
    print(result.to_json(indent=2)[:500])
