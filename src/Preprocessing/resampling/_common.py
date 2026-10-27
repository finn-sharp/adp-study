"""
================================================================================
_common.py - 불균형 처리의 자료형 검증·네이티브 sampler 구성
================================================================================
설명:
    - 불균형 처리의 자료형 검증·네이티브 sampler 구성
분석 가정:
    - 입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

import importlib
import numpy as np
import pandas as pd
from src.Recipes import resolve_profile

SPECS = {
    "random_over": {
        "module": "over_sampling",
        "estimator": "RandomOverSampler",
        "kind": "mixed",
        "cls": "RandomOver",
    },
    "smote": {
        "module": "over_sampling",
        "estimator": "SMOTE",
        "kind": "numeric",
        "cls": "SMOTE",
    },
    "adasyn": {
        "module": "over_sampling",
        "estimator": "ADASYN",
        "kind": "numeric",
        "cls": "ADASYN",
    },
    "borderline_smote": {
        "module": "over_sampling",
        "estimator": "BorderlineSMOTE",
        "kind": "numeric",
        "cls": "BorderlineSMOTE",
    },
    "svm_smote": {
        "module": "over_sampling",
        "estimator": "SVMSMOTE",
        "kind": "numeric",
        "cls": "SVMSMOTE",
    },
    "smotenc": {
        "module": "over_sampling",
        "estimator": "SMOTENC",
        "kind": "mixed_declared",
        "cls": "SMOTENC",
    },
    "smoten": {
        "module": "over_sampling",
        "estimator": "SMOTEN",
        "kind": "categorical",
        "cls": "SMOTEN",
    },
    "random_under": {
        "module": "under_sampling",
        "estimator": "RandomUnderSampler",
        "kind": "mixed",
        "cls": "RandomUnder",
    },
    "tomek_links": {
        "module": "under_sampling",
        "estimator": "TomekLinks",
        "kind": "numeric",
        "cls": "TomekLinks",
    },
    "edited_nearest_neighbours": {
        "module": "under_sampling",
        "estimator": "EditedNearestNeighbours",
        "kind": "numeric",
        "cls": "EditedNearestNeighbours",
    },
    "near_miss": {
        "module": "under_sampling",
        "estimator": "NearMiss",
        "kind": "numeric",
        "cls": "NearMiss",
    },
    "cluster_centroids": {
        "module": "under_sampling",
        "estimator": "ClusterCentroids",
        "kind": "numeric",
        "cls": "ClusterCentroids",
    },
}


def build(method, p):
    """Args: method: 등록된 기법; p: 전용 Profile.
    Returns: Any: clone 가능한 imblearn 네이티브 sampler. CV Pipeline에 삽입합니다.
    Raises: ImportError: imbalanced-learn 누락. TypeError/ValueError: 설정 오류.
    """
    from dataclasses import asdict

    info = SPECS[method]
    values = asdict(p)
    for name in (
        "random_state",
        "k_neighbors",
        "m_neighbors",
        "n_neighbors",
        "version",
    ):
        if name in values:
            value = values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "random_state" else 1)
            ):
                raise ValueError(name + " 정수 범위 오류")
    if "version" in values and values["version"] not in (1, 2, 3):
        raise ValueError("NearMiss version은 1/2/3입니다.")
    estimator = getattr(
        importlib.import_module("imblearn." + info["module"]), info["estimator"]
    )
    return estimator(**values)


def validate(X, y, p, profile_type, kind):
    """Args: X/y: 학습 입력; p/profile_type: 설정·타입; kind: 지원 자료형.
    Returns: None.
    Raises: TypeError/ValueError: shape·결측·자료형·범주 인덱스 오류.
    """
    if not isinstance(p, profile_type):
        raise TypeError(profile_type.__name__ + "이 필요합니다.")
    a, b = np.asarray(X), np.asarray(y)
    if (
        a.ndim != 2
        or not a.shape[0]
        or not a.shape[1]
        or b.ndim != 1
        or len(a) != len(b)
    ):
        raise ValueError("X/y shape 오류")
    if pd.isna(a).any() or pd.isna(b).any():
        raise ValueError("결측은 학습 폴드에서 먼저 처리하세요.")
    if any(isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in a.flat):
        raise ValueError("비유한 특성 오류")
    from sklearn.utils.multiclass import type_of_target

    if type_of_target(b) not in ("binary", "multiclass") or len(np.unique(b)) < 2:
        raise ValueError("두 클래스 이상의 분류 타깃이 필요합니다.")
    if kind == "numeric":
        if a.dtype.kind not in "iuf" or not np.isfinite(a).all():
            raise ValueError("수치 거리 기반 기법에는 유한 수치형만 허용합니다.")
    if kind == "mixed_declared":
        cols = p.categorical_features
        if (
            not isinstance(cols, list)
            or any(isinstance(v, bool) or not isinstance(v, int) for v in cols)
            or len(set(cols)) != len(cols)
            or not 0 < len(cols) < a.shape[1]
            or min(cols) < 0
            or max(cols) >= a.shape[1]
        ):
            raise ValueError("SMOTENC는 일부 범주 열의 고유 위치 인덱스가 필요합니다.")
        numeric = [i for i in range(a.shape[1]) if i not in cols]
        try:
            numbers = a[:, numeric].astype(float)
        except (ValueError, TypeError):
            raise ValueError("비범주 열은 수치여야 합니다.")
        if not np.isfinite(numbers).all():
            raise ValueError("비범주 열 유한값 오류")
    # SMOTEN은 모든 입력 열을 범주로 해석합니다. 숫자 코드도 연속량이 아닙니다.


def resample(X, y, p, method, dto):
    """Args: X/y: 검증된 학습 자료; p: Profile; method: 기법; dto: 전용 출력 타입.
    Returns: DTO: 보완된 학습 자료와 전후 클래스 수·원행 대응.
    Raises: imblearn 예외: 표본 부족·이웃 조건·보완 불가능. 오류를 숨기지 않습니다.
    """

    def counts(values):
        labels, n = np.unique(np.asarray(values), return_counts=True)
        return dict(zip(labels.tolist(), n.astype(int).tolist()))

    sampler = build(method, p)
    if SPECS[method]["kind"] == "mixed_declared":
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for i in range(X.shape[1]):
            if i not in p.categorical_features:
                X[X.columns[i]] = pd.to_numeric(X.iloc[:, i])
    xr, yr = sampler.fit_resample(X, y)
    indices = getattr(sampler, "sample_indices_", None)
    return dto(
        method,
        xr,
        yr,
        counts(y),
        counts(yr),
        indices,
        sampler,
        [
            "학습 폴드에서만 실행하세요. 평가 자료의 분포는 유지합니다.",
            "합성·중심점 표본은 단일 원행에 대응하지 않으므로 sample_indices=None입니다.",
        ],
    )


def load_resampler(name, profile=None, recipe_path=None, overrides=None):
    """CV Pipeline용 factory. 분석 결과를 반환하는 run_*와 구분합니다.

    Args:
        name: random_over/smote/adasyn/borderline_smote/svm_smote/smotenc/smoten/
            random_under/tomek_links/edited_nearest_neighbours/near_miss/cluster_centroids.
        profile: 전용 Profile. JSON 옵션과 혼용 불가.
        recipe_path: 선택 사용자 레시피.
        overrides: 일회성 설정.
    Returns: Any: 아직 적합하지 않은 clone 가능한 imblearn sampler.
    Raises: ValueError/TypeError/OSError: 이름·설정·JSON 오류. ImportError: 의존성 누락.
    Notes: X 자료형·표본 조건은 fit_resample 시점에도 확인해야 합니다.
    """
    if name not in SPECS:
        raise ValueError("지원하지 않는 resampler")
    key = "resampling-" + name.replace("_", "-")
    p = resolve_profile(key, profile, recipe_path=recipe_path, overrides=overrides)
    module = importlib.import_module("src.Preprocessing.resampling." + name)
    if not isinstance(p, getattr(module, SPECS[name]["cls"] + "Profile")):
        raise TypeError("기법별 Profile 타입 불일치")
    return build(name, p)
