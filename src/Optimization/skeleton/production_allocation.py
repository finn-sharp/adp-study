"""
================================================================================
production_allocation.py - 블로그 조건 → JSON/Profile → 별도 생산 모듈 → GA·전수 최적해 대조.
================================================================================
설명:
    - 블로그 조건 → JSON/Profile → 별도 생산 모듈 → GA·전수 최적해 대조.
    - 학습용 시나리오이며 실측 UCI 자료로 표시하지 않습니다. 비용 단위는 만원입니다.
분석 가정:
    - 자료의 열·타깃·분할 조건을 확인하고, 학습 전처리는 학습 구간에서만 적합합니다. 실행은 명시한 진입점에서 수행합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from functools import partial
from src.Recipes import load_profile
from src.Optimization.production_allocation import run_production_allocation


def load_data():
    """독자가 제공한 생산 시나리오 상수를 읽습니다.

    Args:
        없음: 관측 데이터가 아닌 공정 설계 상수입니다.
    Returns:
        tuple: 처리분, 구매 비용, 연간 운영 비용.
    Raises:
        None: 고정 상수 반환.
    """
    return [3, 2, 4], [5000, 7000, 4000], [1000, 1200, 800]


def load_recipe(overrides=None):
    """기본 생산 레시피와 선택 재정의를 적용합니다.

    Args:
        overrides: 일회성 설정.
    Returns:
        ProductionAllocationProfile: 미실행 설정.
    Raises:
        ValueError/TypeError/OSError: 레시피 오류.
    """
    return load_profile("production-allocation", overrides=overrides)


def build_pipeline(profile):
    """설정된 생산 최적화 실행기를 조합합니다.

    Args:
        profile: 전용 생산 설정.
    Returns:
        callable: 세 공정 벡터를 받는 실행기.
    Raises:
        TypeError: 실행 시 설정 오류.
    """
    return partial(run_production_allocation, profile=profile)


def main(overrides=None, data=None):
    """생산량 제약·비용과 GA/전수 차이를 보고합니다.

    Args:
        overrides: Profile 재정의.
        data: 선택 공정 세 벡터, None이면 블로그 시나리오.
    Returns:
        ProductionAllocationOutputDTO: 탐색과 독립 검증 결과.
    Raises:
        ValueError/TypeError/OSError: 공정·설정·레시피 오류.
    """
    return build_pipeline(load_recipe(overrides))(
        *(load_data() if data is None else data)
    )


if __name__ == "__main__":
    print(main().to_json(indent=2))
