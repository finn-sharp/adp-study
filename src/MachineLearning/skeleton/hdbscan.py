"""
================================================================================
hdbscan.py - HDBSCAN UCI 스켈레톤
================================================================================
설명:
    - 데이터 로딩 → JSON/Profile → 전처리/모델 연결 → 학습/분석 → 수치 평가.
분석 가정:
    - 전체 자료를 사용하는 탐색 분석입니다. LDA만 라벨을 적합에 사용하며 보류 성능을 주장하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary (UCI 자료의 라이선스는 별도 출처 명시)
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)       v1.0.0    기존 파이프라인 기반 UCI 수치 예제
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

import numpy as np
from src.Evaluation.clustering.metrics import run_clustering_metrics
from src.Recipes import load_profile
from src.datasets.uci import load_tabular
from ._common import AnalysisEstimator, SkeletonOutputDTO, connect_pipeline, report


def load_data(cache_dir=None):
    """UCI Iris의 네 수치 특성과 원 라벨을 읽습니다.

    Args:
        cache_dir: 선택 UCI 캐시.
    Returns:
        pd.DataFrame: 수치 특성과 target 라벨.
    Raises:
        ValueError/OSError: 자료·다운로드 오류.
    """
    return load_tabular("classification", cache_dir=cache_dir)


def load_recipe(overrides=None):
    """기존 분석 JSON을 전용 Profile로 읽습니다.

    Args:
        overrides: 명시적 레시피 재정의.
    Returns:
        object: 모델 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 레시피 오류.
    """
    return load_profile("hdbscan", overrides=overrides)


def build_pipeline(feature_columns, profile=None, overrides=None):
    """FeatureSpec 전처리와 기존 분석 runner를 연결합니다.

    Args:
        feature_columns: 원 특성 열 순서.
        profile: 전용 Profile 또는 None.
        overrides: Profile 생략 시 재정의.
    Returns:
        Basepipeline: 미적합 분석 파이프라인.
    Raises:
        ValueError: Profile과 overrides 동시 지정.
    """
    if profile is not None and overrides is not None:
        raise ValueError("profile과 overrides를 동시에 지정하지 않습니다.")
    profile = load_recipe(overrides) if profile is None else profile
    estimator = AnalysisEstimator(
        "src.MachineLearning.Clustering.hdbscan", "run_hdbscan", profile
    )
    return connect_pipeline(feature_columns, estimator, profile)


def main(cache_dir=None, overrides=None):
    """전체 자료의 탐색 분석을 수행하고 수치 결과만 출력합니다.

    Args:
        cache_dir: 선택 UCI 캐시.
        overrides: 명시적 레시피 재정의.
    Returns:
        SkeletonOutputDTO: embedding/군집/거리와 기술 지표.
    Raises:
        ValueError/OSError/ImportError: 기법 조건·다운로드·라이브러리 오류.
    """
    data = load_data(cache_dir)
    X = data.drop(columns="target")
    profile = load_recipe(overrides)
    pipeline = build_pipeline(X.columns, profile=profile)
    pipeline.pipeline.fit(X, data.target if False else None)
    result = pipeline.get_model().result_
    metrics = {}
    for field in (
        "silhouette",
        "inertia",
        "stress",
        "distance_error",
        "kl_divergence",
        "reconstruction_error",
        "quantization_error",
        "topographic_error",
    ):
        value = getattr(result, field, None)
        if value is not None:
            metrics[field] = float(value)
    coordinates = pipeline.pipeline.named_steps["preprocessor"].transform(X)
    evaluation = run_clustering_metrics(
        data.target.to_numpy(), result.labels, X=coordinates
    )
    metrics.update({k: float(v) for k, v in evaluation.value.items()})
    details = {
        "preprocessor": pipeline.feature_specification.describe(),
        "evaluation_scope": "전체 자료의 탐색/기술 분석; 보류 일반화 성능 아님",
    }
    details["evaluation"] = evaluation.to_dict(json_safe=True)
    prediction = getattr(
        result,
        "embedding",
        getattr(result, "labels", getattr(result, "distances", None)),
    )
    if prediction is not None:
        details["output_shape"] = list(np.shape(prediction))
    if hasattr(result, "supports_transform"):
        details["supports_new_transform"] = result.supports_transform
    if hasattr(result, "n_clusters"):
        details["n_clusters"] = result.n_clusters
        details["n_noise"] = int(result.noise_mask.sum())
    return report(
        SkeletonOutputDTO(
            "HDBSCAN",
            "UCI Iris",
            "exploratory-analysis",
            len(data),
            X.shape[1],
            metrics,
            details,
            pipeline,
            predictions=prediction,
        )
    )


if __name__ == "__main__":
    main()
