"""
================================================================================
calinski_harabasz.py - UCI 자료·예측·독립 평가 skeleton
================================================================================
설명:
    - 원 자료 → JSON/Profile → 학습 구간 전처리·모델 → 보류 예측 → 독립 메트릭.
분석 가정:
    - 보류 자료는 학습 적합에 사용하지 않습니다. 군집은 전체 자료 탐색입니다.
    - 시간 순 예제는 직전 관측 naive 예측이며 import 시 다운로드·학습하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-13
    - 라이선스: Proprietary (UCI 자료의 라이선스는 별도)
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-13  김재현(Finn)      v1.0.0    메트릭별 데이터·레시피·평가 흐름
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from src.Recipes import load_profile
from src.Evaluation.clustering.calinski_harabasz import run_calinski_harabasz
from src.Evaluation.skeleton._common import load_data as _load_data, prepare


def load_data(cache_dir=None):
    """분야별 공식 UCI 자료를 읽습니다.

    Args:
        cache_dir: 선택 UCI 캐시 경로.
    Returns:
        DataFrame: 원 자료.
    Raises:
        ValueError/OSError: 자료/다운로드 오류.
    """
    return _load_data("clustering", cache_dir)


def load_recipe(overrides=None):
    """메트릭별 JSON을 전용 Profile로 읽습니다.

    Args:
        overrides: 평가 설정 재정의.
    Returns:
        ClusteringCalinskiHarabaszProfile: 평가 설정.
    Raises:
        ValueError/OSError: 레시피 오류.
    """
    return load_profile("metric-clustering-calinski-harabasz", overrides=overrides)


def build_pipeline(profile=None, overrides=None):
    """모델 적합/보류 예측 이후 평가하는 실행 함수를 구성합니다.

    Args:
        profile: 전용 평가 Profile 또는 None.
        overrides: Profile 생략 시 재정의.
    Returns:
        callable: 원 자료에서 전용 평가 DTO를 반환하는 함수.
    Raises:
        ValueError: Profile과 overrides 동시 지정.
    """
    if profile is not None and overrides is not None:
        raise ValueError("profile과 overrides를 함께 지정하지 않습니다.")
    p = load_recipe(overrides) if profile is None else profile

    def execute(data):
        """보류 예측을 만들고 해당 메트릭을 실행합니다.

        Args:
            data: load_data 원 자료 또는 동일 schema 사용자 자료.
        Returns:
            OutputDTO: 독립 메트릭 결과.
        Raises:
            ValueError: 입력·지표 정의역 오류.
        """
        return run_calinski_harabasz(**prepare("clustering", data), profile=p)

    return execute


def main(cache_dir=None, overrides=None, data=None):
    """자료·레시피·예측·평가를 연결하고 수치 결과를 출력합니다.

    Args:
        cache_dir: 선택 UCI 캐시.
        overrides: 평가 레시피 재정의.
        data: 오프라인 실행용 원 DataFrame; None이면 공식 UCI 로더.
    Returns:
        OutputDTO: value·정의·클래스/행 순서·JSON 결과.
    Raises:
        ValueError/OSError: 자료/평가/레시피 오류.
    """
    result = build_pipeline(overrides=overrides)(
        load_data(cache_dir) if data is None else data
    )
    print(result.to_json(indent=2))
    return result


if __name__ == "__main__":
    main()
