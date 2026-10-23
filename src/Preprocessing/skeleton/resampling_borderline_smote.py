"""고정 합성 자료 → 레시피 → 분석 callable → 전용 DTO 예제.

외부 통신과 import 시 학습이 없으며 관측 자료와 설정을 분리합니다.
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from functools import partial
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.Preprocessing.resampling.borderline_smote import run_borderline_smote


def load_data():
    """Args: 없음. 고정 시드·자료로 실행합니다.
    Returns: tuple: 공개 runner의 위치 입력 인자.
    Raises: ValueError: 예제 자료 생성 실패.
    """
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=160,
        n_features=4,
        n_redundant=0,
        weights=[0.75, 0.25],
        random_state=1234,
    )


def load_recipe(overrides=None):
    """Args: overrides: 선택 일회성 Profile 설정.
    Returns: object: 기법별 JSON에서 구성한 전용 Profile.
    Raises: ValueError/TypeError/OSError: 설정·레시피 오류.
    """
    return load_profile("resampling-borderline-smote", overrides=overrides)


def build_pipeline(profile=None):
    """Args: profile: 명시적 설정, 생략하면 기법별 JSON 로드.
    Returns: callable: 자료를 받아 전용 DTO를 반환하는 분석 실행 흐름.
        sklearn Pipeline이 아닌 함수 조합입니다. CV 리샘플링은 load_resampler를 사용합니다.
    Raises: ValueError/TypeError/OSError: 설정·레시피 오류. 자료 검증은 호출 시 수행.
    """
    return partial(
        run_borderline_smote, profile=load_recipe() if profile is None else profile
    )


def main():
    """Args: 없음. 고정 예제 자료와 기본 레시피를 사용합니다.
    Returns: object: 실제 실행한 전용 Output DTO.
    Raises: ValueError/TypeError/ImportError/OSError: 입력·설정·선택 의존성·적합 실패.
    """
    result = build_pipeline(load_recipe())(*load_data())
    print(result.to_json(indent=2))
    return result


if __name__ == "__main__":
    main()
