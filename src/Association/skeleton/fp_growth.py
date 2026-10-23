"""
================================================================================
fp_growth.py - fp_growth UCI 스켈레톤
================================================================================
설명:
    - 데이터 로딩 → JSON/Profile → 전처리/모델 연결 → 학습/분석 → 수치 평가.
분석 가정:
    - 영수증의 고유 상품 공동 등장만 분석하며 판매량 가중/인과적 관계나 사회적 친구 관계로 해석하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin-klg@naver.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary (UCI 자료의 라이선스는 별도 출처 명시)
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-12  김재현(Finn)       v1.0.0    기존 파이프라인 기반 UCI 수치 예제
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

from collections import Counter
from itertools import combinations
from src.Recipes import load_profile
from src.Association.fp_growth import run_fp_growth
from src.datasets.uci import load_baskets

UCI_DEFAULT_OVERRIDES = {"min_support": 0.05}


def load_data(cache_dir=None, max_transactions=1000, max_items=30):
    """취소·음수 수량·비상품 행을 제외한 UCI Retail 영수증을 읽습니다.

    Args:
        cache_dir: 선택 UCI 캐시.
        max_transactions: 완전한 영수증 수 한도.
        max_items: 거래 빈도에 따른 상품 수 한도.
    Returns:
        list: 영수증별 고유 상품 집합.
    Raises:
        ValueError/OSError/ImportError: 자료·다운로드·Excel 엔진 오류.
    """
    return load_baskets(cache_dir, max_transactions, max_items)


def load_recipe(overrides=None):
    """기존 fp_growth JSON을 전용 Profile로 읽습니다.

    Args:
        overrides: 명시적 레시피 재정의.
    Returns:
        object: 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 레시피 오류.
    """
    settings = dict(UCI_DEFAULT_OVERRIDES)
    settings.update(overrides or {})
    return load_profile("fp-growth", overrides=settings)


def build_pipeline(profile=None, overrides=None):
    """장바구니 구조를 유지하는 분석 실행 함수를 구성합니다.

    Args:
        profile: 전용 Profile 또는 None.
        overrides: Profile 생략 시 재정의.
    Returns:
        callable: 정제한 영수증을 분석하는 함수.
    Raises:
        ValueError: Profile과 overrides 동시 지정.
    """
    if profile is not None and overrides is not None:
        raise ValueError("profile과 overrides를 동시에 지정하지 않습니다.")
    profile = load_recipe(overrides) if profile is None else profile
    return lambda baskets: run_fp_growth(baskets, profile=profile)


def main(cache_dir=None, overrides=None, max_transactions=1000, max_items=30):
    """UCI Retail을 정제하고 수치/표 결과를 출력합니다.

    Args:
        cache_dir: 선택 UCI 캐시.
        overrides: 명시적 레시피 재정의.
        max_transactions: 원 시간 순 완전한 영수증 한도.
        max_items: 상품 한도.
    Returns:
        OutputDTO: 기존 분석 전용 결과.
    Raises:
        ValueError/OSError/ImportError: 자료·설정·라이브러리 오류.
    """
    data = load_data(cache_dir, max_transactions, max_items)
    profile = load_recipe(overrides)
    pipeline = build_pipeline(profile=profile)
    result = pipeline(data)
    print("=== Example : UCI Online Retail fp_growth ===")
    print("transactions:", result.n_transactions)
    print("itemsets:", len(result.itemsets), "rules:", len(result.rules))
    print(result.rules.head(10).to_string(index=False))
    return result


if __name__ == "__main__":
    main()
