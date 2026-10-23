"""phi UCI 스켈레톤.

데이터 로딩 → JSON/Profile → Feature/분석 연결 → 실행 → 수치 확인.
전처리는 학습 Feature에서만 적합하며 타깃을 제외합니다. 시각화를 포함하지 않습니다.
작성자: 김재현(Finn), 2026-10-12. 변경 이력: v1.0.0 최초 추가.
라이선스: Proprietary; UCI 자료는 별도 CC BY 4.0.
"""

__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

from functools import partial
from src.Recipes import load_profile
from src.datasets.adult import load_adult
from src.stats.correlation.categorical.phi import run_phi


def load_data(cache_dir=None):
    """UCI 원 Feature를 탐색 분석용으로 선택합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시 경로 또는 None.
    Returns:
        DataFrame: 수치 또는 원 범주; 범주를 임의 순위로 치환하지 않습니다.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    train, test = load_adult(cache_dir)
    return train[["sex", "income"]]


def load_recipe(overrides=None):
    """전용 JSON/Profile을 읽습니다.

    Args:
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        object: 분석 또는 행렬 변환 설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    settings = {"positive_x": "Male", "positive_y": ">50K"}
    settings.update(overrides or {})
    return load_profile("correlation-phi", overrides=settings)


def build_pipeline(profile=None, overrides=None):
    """Profile을 전용 분석 실행기에 연결합니다.

    Args:
        profile: 전용 Profile 또는 None; overrides와 동시 지정 불가.
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        callable: 탐색 분석; 일반화 성능을 평가하는 모델이 아닙니다.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    if profile is not None and overrides is not None:
        raise ValueError("profile과 overrides는 함께 지정하지 않습니다.")
    profile = load_recipe(overrides) if profile is None else profile
    return partial(run_phi, profile=profile)


def main(cache_dir=None, overrides=None):
    """원본 로딩·레시피·분석 연결·수치 출력을 실행합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시 경로 또는 None.
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        OutputDTO: 계수·검정·행렬 또는 정리 결과.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    data = load_data(cache_dir)
    pipeline = build_pipeline(overrides=overrides)
    result = pipeline(data.iloc[:, 0], data.iloc[:, 1])
    print(result.to_json(indent=2))
    return result


if __name__ == "__main__":
    main()
