"""feature_pipeline UCI 스켈레톤.

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
from src.datasets.adult import load_adult
from src.Preprocessing.feature_pipeline import make_feature_spec, build_feature_preprocessor, run_feature_preprocessing
from src.Preprocessing.numeric.imputation import build_numeric_imputation_transformer
from src.Preprocessing.numeric.scaling import build_numeric_scaling_transformer
from src.Preprocessing.categorical.imputation import build_categorical_imputation_transformer
from src.Preprocessing.categorical.encoding import build_categorical_encoding_transformer
NUMERIC = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

def load_data(cache_dir=None):
    """Adult 공식 train/test를 유지하고 타깃을 Feature에서 분리합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시 경로 또는 None.
    Returns:
        tuple: X_train, X_test, y_train, y_test.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    train, test = load_adult(cache_dir)
    columns = NUMERIC + CATEGORICAL
    return train[columns], test[columns], train.income.copy(), test.income.copy()

def load_recipe(overrides=None):
    """Feature 조합 JSON을 읽고 타깃 제외 규칙을 지정합니다.

    Args:
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        FeaturePreprocessingProfile: 조합 설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    settings = {'target_columns': ['income']}
    settings.update(overrides or {})
    return load_profile('feature-preprocessing', overrides=settings)

def build_pipeline(profile=None, overrides=None):
    """Feature별 순서를 정의하여 모델/CV에 연결할 미적합 변환기를 만듭니다.

    Args:
        profile: 전용 Profile 또는 None; overrides와 동시 지정 불가.
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        tuple: FeatureSpec 목록, 미적합 ColumnTransformer.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    if profile is not None and overrides is not None:
        raise ValueError('profile과 overrides는 함께 지정하지 않습니다.')
    profile = load_recipe(overrides) if profile is None else profile
    features = [
        make_feature_spec('numeric', NUMERIC, [('impute', build_numeric_imputation_transformer()), ('scale', build_numeric_scaling_transformer())]),
        make_feature_spec('categorical', CATEGORICAL, [('impute', build_categorical_imputation_transformer()), ('encode', build_categorical_encoding_transformer())]),
    ]
    return features, build_feature_preprocessor(features, profile)

def main(cache_dir=None, overrides=None):
    """원본 순서대로 타깃 분리·Feature 선언·학습 전용 적합·수치 확인을 실행합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시 경로 또는 None.
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        FeaturePreprocessingOutputDTO: Feature 행렬·이름·재사용 전처리.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    train, test, y_train, y_test = load_data(cache_dir)
    profile = load_recipe(overrides)
    features, unfitted = build_pipeline(profile=profile)
    # 모델 학습 시에는 Pipeline([('features', unfitted), ('model', estimator)])를 CV에 전달합니다.
    result = run_feature_preprocessing(train, features, test, profile)
    print('train/new shapes:', result.preprocessed_feature.shape, result.new_preprocessed_feature.shape)
    print('feature steps:', result.feature_specs)
    print('target rows:', len(y_train), len(y_test))
    return result


if __name__ == '__main__':
    main()
