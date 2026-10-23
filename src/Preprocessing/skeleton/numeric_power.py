"""numeric_power UCI 스켈레톤.

데이터 로딩 → JSON/Profile → Feature/분석 연결 → 실행 → 수치 확인.
전처리는 학습 Feature에서만 적합하며 타깃을 제외합니다. 시각화를 포함하지 않습니다.
작성자: 김재현(Finn), 2026-10-12. 변경 이력: v1.0.0 최초 추가.
라이선스: Proprietary; UCI 자료는 별도 CC BY 4.0.
"""

__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

from src.Recipes import load_profile
from src.Preprocessing.numeric.power import build_numeric_power_transformer
from src.Preprocessing.feature_pipeline import (
    make_feature_spec,
    run_feature_preprocessing,
)
from src.datasets.uci import load_tabular
from sklearn.model_selection import train_test_split


def load_data(cache_dir=None):
    """UCI 원본에서 타깃을 제외하고 학습/보류 Feature를 먼저 분리합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시 경로 또는 None.
    Returns:
        tuple: 학습 DataFrame, 보류 DataFrame.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    data = load_tabular("regression", cache_dir=cache_dir).drop(columns="target")
    return train_test_split(data, test_size=0.2, random_state=1234)


def load_recipe(overrides=None):
    """전용 JSON을 Profile로 검증합니다.

    Args:
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        object: 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    return load_profile("numeric-power", overrides=overrides)


def build_pipeline(feature_columns, profile=None, overrides=None):
    """FeatureSpec의 대상 열과 미적합 처리 순서를 선언합니다.

    Args:
        feature_columns: 타깃을 제외한 원 Feature 열 이름과 순서.
        profile: 전용 Profile 또는 None; overrides와 동시 지정 불가.
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        list: FeatureSpec 목록.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    if profile is not None and overrides is not None:
        raise ValueError("profile과 overrides는 함께 지정하지 않습니다.")
    profile = load_recipe(overrides) if profile is None else profile
    return [
        make_feature_spec(
            "numeric",
            list(feature_columns),
            [("power", build_numeric_power_transformer(profile))],
        )
    ]


def main(cache_dir=None, overrides=None):
    """학습 Feature만 적합하고 동일 구성으로 보류 Feature를 변환합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시 경로 또는 None.
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        FeaturePreprocessingOutputDTO: 변환 Feature·이름·재사용 변환기.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    train, test = load_data(cache_dir)
    profile = load_recipe(overrides)
    features = build_pipeline(train.columns, profile=profile)
    result = run_feature_preprocessing(train, features, test)
    print(
        "train/new shapes:",
        result.preprocessed_feature.shape,
        result.new_preprocessed_feature.shape,
    )
    print("feature steps:", result.feature_specs)
    return result


if __name__ == "__main__":
    main()
