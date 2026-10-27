"""
================================================================================
information_selection.py - 실제 UCI 자료 → JSON/Profile → 분야별 runner → DTO. import에서 실행 없음.
================================================================================
설명:
    - 실제 UCI 자료 → JSON/Profile → 분야별 runner → DTO. import에서 실행 없음.
    - 작성자: 김재현(Finn), 2026-10-23; Proprietary, UCI 라이선스는 원자료에 따름.
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

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from functools import partial
from src.Recipes import load_profile
from src.datasets.uci import load_tabular, load_bike
from src.stats.regression.information_selection import run_information_selection


def load_data(cache_dir=None):
    """UCI의 원 척도·시간순·반응 정의역을 보존합니다.

    Args:
        cache_dir: 실제 UCI 캐시.
    Returns:
        tuple: (X,y) 또는 (시간순 series,None).
    Raises:
        OSError/ValueError: 다운로드/자료 오류.
    """
    d = load_tabular("regression", cache_dir=cache_dir)
    return d[["alcohol", "pH", "sulphates", "fixed acidity"]], d.target


def load_recipe(overrides=None):
    """전용 JSON을 Profile로 읽습니다.

    Args:
        overrides: 선택 설정 재정의.
    Returns:
        Profile: 전용 설정.
    Raises:
        OSError/ValueError/TypeError: 레시피 오류.
    """
    return load_profile("regression-information-selection", overrides=overrides)


def build_pipeline(profile):
    """분석 runner와 명시 설정을 연결합니다.

    Args:
        profile: 전용 설정.
    Returns:
        callable: 미실행 runner.
    Raises:
        TypeError: 실행 때 설정 오류.
    """
    return partial(run_information_selection, profile=profile)


def main(cache_dir=None, overrides=None):
    """정의역에 맞는 실제 UCI 실행 예제.

    Args:
        cache_dir: UCI 캐시.
        overrides: 설정 재정의.
    Returns:
        OutputDTO: 적합·진단 결과. 보류가 없는 추론은 일반화 성능 아님.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·적합 오류.
    """
    X, y = load_data(cache_dir)
    runner = build_pipeline(load_recipe(overrides))
    return runner(X, y)


if __name__ == "__main__":
    print(main().to_json(indent=2))
