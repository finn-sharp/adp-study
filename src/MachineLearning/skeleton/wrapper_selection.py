"""
================================================================================
wrapper_selection.py - Wrapper Selection UCI 스켈레톤
================================================================================
설명:
    - 데이터 로딩 → JSON/Profile → 전처리/모델 연결 → 학습/분석 → 수치 평가.
분석 가정:
    - 보류 자료는 전처리 적합·변수 선택·모델 선택에 사용하지 않습니다.
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

from src.Recipes import load_profile
from sklearn.pipeline import Pipeline
from src.MachineLearning.common import make_scaler
from src.MachineLearning.FeatureSelection.wrapper import _base_estimator
from src.MachineLearning.FeatureSelection.wrapper import WrapperSelector
from src.datasets.uci import load_tabular
from ._common import connect_pipeline, fit_evaluate, report


def load_data(task="classification", cache_dir=None):
    """UCI 자료를 원 특성과 target 열로 불러옵니다.

    Args:
        task: 지원하는 분류/회귀 목적.
        cache_dir: 선택 UCI 원본 캐시.
    Returns:
        pd.DataFrame: Iris 또는 Wine Quality; Adaline은 이진 Iris.
    Raises:
        ValueError/OSError: 목적·자료·다운로드 오류.
    """
    if task not in ("classification", "regression"):
        raise ValueError("이 모델 스켈레톤이 지원하지 않는 task입니다.")
    return load_tabular(task, binary=False, cache_dir=cache_dir)


def load_recipe(task="classification", overrides=None):
    """기존 모델별 JSON을 검증하여 전용 Profile로 읽습니다.

    Args:
        task: 모델 목적; 파일 선택과 일치해야 함.
        overrides: 명시적인 레시피 재정의.
    Returns:
        object: 기존 구현의 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 레시피 필드·별칭·파일 오류.
    """
    return load_profile("wrapper-selection", task=task, overrides=overrides)


def build_pipeline(
    feature_columns, task="classification", profile=None, overrides=None
):
    """FeatureSpec 전처리와 기존 모델을 연결합니다.

    Args:
        feature_columns: target을 제외한 원 특성 열 순서.
        task: 모델 목적.
        profile: load_recipe로 읽은 전용 Profile 또는 None.
        overrides: Profile 생략 시 적용할 레시피 재정의.
    Returns:
        Basepipeline: 미적합 전처리·모델 연결 객체.
    Raises:
        ValueError/ImportError: 목적·전처리·모델 라이브러리 오류.
    """
    if profile is not None and overrides is not None:
        raise ValueError("profile과 overrides를 동시에 지정하지 않습니다.")
    profile = load_recipe(task, overrides) if profile is None else profile
    if profile.task != task:
        raise ValueError("Profile task와 요청 목적이 다릅니다.")
    selector = WrapperSelector(
        task=profile.task,
        scaling=profile.scaling,
        random_state=profile.random_state,
        C=profile.C,
        max_iter=profile.max_iter,
        method=profile.method,
        n_features_to_select=profile.n_features_to_select,
        min_improvement=profile.min_improvement,
        max_steps=profile.max_steps,
        cv=profile.cv,
        scoring=profile.scoring,
        n_jobs=profile.n_jobs,
    )
    estimator = Pipeline(
        [
            ("selector", selector),
            ("selected_scaler", make_scaler(profile.scaling)),
            ("model", _base_estimator(profile.task, profile.C, profile.max_iter)),
        ]
    )
    return connect_pipeline(feature_columns, estimator, profile, scaling="none")


def main(task="classification", cache_dir=None, overrides=None):
    """원 참고 코드의 순서로 학습·예측·수치 평가를 실행합니다.

    Args:
        task: 분류/회귀 목적.
        cache_dir: 선택 UCI 원본 캐시.
        overrides: 명시적인 레시피 재정의.
    Returns:
        SkeletonOutputDTO: 보류 평가·예측·재사용 파이프라인.
    Raises:
        ValueError/OSError/ImportError: 자료·설정·라이브러리 오류.
    """
    data = load_data(task, cache_dir)
    profile = load_recipe(task, overrides)
    feature_columns = [column for column in data.columns if column != "target"]
    pipeline = build_pipeline(feature_columns, task=task, profile=profile)
    return report(fit_evaluate(data, pipeline, "Wrapper Selection"))


if __name__ == "__main__":
    main()
