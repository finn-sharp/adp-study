"""
================================================================================
feature_pipeline.py - Feature 기반 전처리 순서와 컬럼 조합
================================================================================
설명:
    - FeatureSpec별 대상 열·처리 순서를 sklearn Pipeline/ColumnTransformer로 연결합니다.
분석 가정:
    - 행은 표본이고 열은 Feature입니다. RoPE는 이 표 계약이 아니라 임베딩 단계에 적용합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    Feature별 순서·학습 전용 전처리 추가
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.MachineLearning.base.feature import FeatureSpec
from src.MachineLearning.analysis_common import finite_number
from src.Preprocessing.positional.rope import RotaryPositionTransformer


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class FeaturePreprocessingProfile:
    """Feature 조합 설정; 각 기법 설정은 각 Feature의 변환기가 보존합니다.

    Attributes:
        remainder (str): 지정하지 않은 열 drop/passthrough; 기본 drop.
        sparse_threshold (float): ColumnTransformer sparse 결합 임계값 [0,1]; 기본 0.3.
        allow_overlap (bool): 서로 다른 그룹의 열 중복 허용; 기본 False.
        target_columns (Optional[List[str]]): Feature에서 제외할 타깃 이름; 기본 None.
    """

    remainder: str = "drop"
    sparse_threshold: float = 0.3
    allow_overlap: bool = False
    target_columns: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class FeaturePreprocessingOutputDTO(OutputDTO):
    """Feature 조합 전처리 결과.

    Attributes:
        preprocessed_feature (Any): 학습 Feature 변환 행렬.
        new_preprocessed_feature (Any): 새 Feature 변환 또는 None.
        input_features (List[str]): 원 열 이름과 순서.
        output_features (List[str]): 그룹/변환 출력 열 이름과 순서.
        feature_specs (List[Dict[str, Any]]): 그룹별 원 컬럼과 처리 단계 설명.
        parameters (Dict[str, Any]): Feature 조합 Profile.
        transformer (Any): 학습 자료에 적합한 ColumnTransformer; JSON 제외.
        n_samples (int): 학습 행 수.
    """

    preprocessed_feature: Any
    new_preprocessed_feature: Any
    input_features: List[str]
    output_features: List[str]
    feature_specs: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    transformer: Any
    n_samples: int


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate(features, profile):
    """Feature 이름·열 중복·타깃·위치 임베딩의 역할을 확인합니다.

    Args:
        features: FeatureSpec 목록.
        profile: 전용 Profile.
    Returns:
        list: 활성 FeatureSpec 목록.
    Raises:
        ValueError/TypeError: 스키마·설정·타깃/임베딩 혼용 오류.
    """
    if not isinstance(profile, FeaturePreprocessingProfile):
        raise TypeError("전용 Profile이 필요합니다.")
    if profile.remainder not in ("drop", "passthrough") or not isinstance(
        profile.allow_overlap, bool
    ):
        raise ValueError("Feature 조합 설정 오류")
    finite_number(profile.sparse_threshold, "sparse_threshold")
    if not 0 <= profile.sparse_threshold <= 1:
        raise ValueError("sparse_threshold 오류")
    if profile.target_columns is not None and (
        not isinstance(profile.target_columns, list)
        or any(not isinstance(c, str) for c in profile.target_columns)
    ):
        raise ValueError("target_columns는 문자열 목록입니다.")
    if (
        not isinstance(features, (list, tuple))
        or not features
        or any(not isinstance(f, FeatureSpec) for f in features)
    ):
        raise TypeError("FeatureSpec 목록이 필요합니다.")
    active = [f for f in features if f.enabled]
    if not active or len({f.name for f in active}) != len(active):
        raise ValueError("활성 Feature와 고유 이름이 필요합니다.")
    assigned = []
    targets = set(profile.target_columns or [])
    if targets and profile.remainder == "passthrough":
        raise ValueError(
            "명시한 타깃이 있으면 remainder=drop으로 Feature만 선택하세요."
        )
    for feature in active:
        assigned.extend(feature.columns)
        if targets.intersection(feature.columns):
            raise ValueError("타깃은 Feature 전처리에 포함하지 않습니다.")
        steps = (
            feature.transformer.named_steps.values()
            if isinstance(feature.transformer, Pipeline)
            else [feature.transformer]
        )
        if any(isinstance(step, RotaryPositionTransformer) for step in steps):
            raise ValueError(
                "RoPE는 표 Feature가 아니라 별도 시퀀스 임베딩 단계에 적용하세요."
            )
    if not profile.allow_overlap and len(set(assigned)) != len(assigned):
        raise ValueError("서로 다른 Feature 그룹에 같은 열이 중복되었습니다.")
    return active


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def make_feature_spec(name, columns, steps):
    """기법 정의와 별개로 Feature에 적용할 순서를 선언합니다.

    Args:
        name: Feature 그룹 이름.
        columns: 대상 원 열 목록.
        steps: (단계 이름, 미적합 sklearn 변환기) 순서 목록.
    Returns:
        FeatureSpec: 전처리 Pipeline을 가진 선언.
    Raises:
        ValueError/TypeError: 빈 단계·이름·변환기 오류.
    """
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ValueError("처리 단계 목록이 필요합니다.")
    if any(not isinstance(step, (list, tuple)) or len(step) != 2 for step in steps):
        raise ValueError("단계는 (이름,변환기)입니다.")
    names = [step[0] for step in steps]
    if any(
        not isinstance(name, str) or not name or "__" in name for name in names
    ) or len(set(names)) != len(names):
        raise ValueError("단계 이름 오류")
    if any(
        not hasattr(step[1], "fit") or not hasattr(step[1], "transform")
        for step in steps
    ):
        raise TypeError("각 단계에 fit/transform 변환기가 필요합니다.")
    return FeatureSpec(name, columns, Pipeline(list(steps)))


def build_feature_preprocessor(
    features, profile=None, *, recipe_path=None, overrides=None
):
    """모델 Pipeline/CV에 연결할 미적합 Feature 조합을 만듭니다.

    Args:
        features: 순서를 가진 FeatureSpec 목록.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        ColumnTransformer: sklearn clone/CV 호환 전처리.
    Raises:
        ValueError/TypeError/OSError: 스키마·설정·레시피 오류.
    """
    profile = resolve_profile(
        "feature-preprocessing", profile, recipe_path=recipe_path, overrides=overrides
    )
    active = _validate(features, profile)
    return ColumnTransformer(
        [(f.name, f.transformer, f.columns) for f in active],
        remainder=profile.remainder,
        sparse_threshold=profile.sparse_threshold,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_feature_preprocessing(
    X_train, features, X_new=None, profile=None, *, recipe_path=None, overrides=None
):
    """학습 구간에서만 Feature 처리를 적합하고 재사용합니다.

    Args:
        X_train: 이름을 가진 학습 Feature DataFrame.
        features: 대상 열/단계 순서의 FeatureSpec 목록.
        X_new: 같은 열 이름/순서의 새 DataFrame 또는 None.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        FeaturePreprocessingOutputDTO: 전처리 Feature와 명시적 단계 설명.
    Raises:
        ValueError/TypeError/OSError: 자료·스키마·타깃·레시피 오류.
    """
    profile = resolve_profile(
        "feature-preprocessing", profile, recipe_path=recipe_path, overrides=overrides
    )
    active = _validate(features, profile)
    if (
        not isinstance(X_train, pd.DataFrame)
        or X_train.empty
        or not X_train.columns.is_unique
        or any(not isinstance(c, str) for c in X_train.columns)
    ):
        raise ValueError("고유 문자열 열의 학습 DataFrame이 필요합니다.")
    if profile.remainder == "passthrough" and set(
        profile.target_columns or []
    ).intersection(X_train.columns):
        raise ValueError("passthrough로 타깃이 전달될 수 없습니다.")
    if X_new is not None and (
        not isinstance(X_new, pd.DataFrame)
        or X_new.columns.tolist() != X_train.columns.tolist()
    ):
        raise ValueError("새 Feature 열 이름/순서가 다릅니다.")
    transformer = build_feature_preprocessor(features, profile).fit(X_train)
    descriptions = []
    for feature in active:
        item = feature.describe()
        item["steps"] = (
            [
                dict(name=name, transformer=type(step).__name__)
                for name, step in feature.transformer.steps
            ]
            if isinstance(feature.transformer, Pipeline)
            else []
        )
        descriptions.append(item)
    return FeaturePreprocessingOutputDTO(
        transformer.transform(X_train),
        None if X_new is None else transformer.transform(X_new),
        X_train.columns.tolist(),
        transformer.get_feature_names_out().tolist(),
        descriptions,
        asdict(profile),
        transformer,
        len(X_train),
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.Preprocessing.numeric.scaling import build_numeric_scaling_transformer

    feature = make_feature_spec(
        "amount", ["amount"], [("scale", build_numeric_scaling_transformer())]
    )
    print(
        run_feature_preprocessing(
            pd.DataFrame({"amount": [1.0, 2.0, 3.0]}), [feature]
        ).to_json(indent=2)
    )
