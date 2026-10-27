"""
================================================================================
vif_selection.py - 공선성의 train-only 순차 VIF Feature 선택
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 학습 적합과 선택은 학습/CV 안에서 수행하고 추론 가정을 확인합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.stats.regression._common import validate_profile, numeric_data, vif_values
from src.Preprocessing._common import ProfileTransformer, execute_preprocessing


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class VIFSelectionProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        threshold (Optional[float]): VIF 제거 기준 1 초과; None은 상수만 제거. 기본 10.0.
        drop_constant (bool): 상수 Feature 제거 여부. 기본 True.
        max_steps (int): 순차 제거 단계 한도. 기본 100.
        min_features (int): 최소 남길 Feature 수. 기본 1.
        preserve_columns (Optional[List[str]]): 삭제 금지 원/변환 Feature 이름. 기본 None.
    """

    threshold: Optional[float] = 10.0
    drop_constant: bool = True
    max_steps: int = 100
    min_features: int = 1
    preserve_columns: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class VIFSelectionOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        preprocessed_feature (Any): 학습 Feature/타깃 변환.
        new_preprocessed_feature (Any): 새 자료 transform 또는 None.
        input_features (List[str]): 원 열 순서.
        output_features (List[str]): 생성/선택한 열 순서.
        parameters (Dict[str,Any]): 적용 설정.
        transformer (Any): 학습 자료에만 fit한 재사용 변환기; JSON 제외.
        n_samples (int): 학습 행 수.
        warnings (List[str]): 추론·선택 해석 정보.
        selection_history (pd.DataFrame): 순차 제거 순서와 VIF/상수 이유.
    """

    preprocessed_feature: Any
    new_preprocessed_feature: Any
    input_features: List[str]
    output_features: List[str]
    parameters: Dict[str, Any]
    transformer: Any
    n_samples: int
    warnings: List[str]
    selection_history: pd.DataFrame


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate(profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
    Returns:
        object: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    validate_profile(profile, VIFSelectionProfile)
    if profile.threshold is not None and (
        isinstance(profile.threshold, bool)
        or not isinstance(profile.threshold, (int, float))
        or not np.isfinite(profile.threshold)
        or profile.threshold <= 1
    ):
        raise ValueError("VIF 기준은 1 초과입니다.")
    if profile.max_steps < 1 or profile.min_features < 1:
        raise ValueError("선택 한도 오류")
    if profile.preserve_columns is not None and (
        not isinstance(profile.preserve_columns, list)
        or any(not isinstance(v, str) for v in profile.preserve_columns)
    ):
        raise ValueError("보호 열 목록 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def build_vif_selector(profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
        recipe_path: 전용 명세의 recipe_path 입력.
        overrides: 전용 명세의 overrides 입력.
    Returns:
        VIFSelector: 학습 Feature만 선택하는 clone 호환 변환기.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-vif-selection",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    return VIFSelector(profile)


class VIFSelector(TransformerMixin, BaseEstimator):
    """학습 구간에서 순차 VIF 제거를 하고 동일 열만 새 입력에 적용합니다."""

    def __init__(self, profile=None):
        """설정 보존.

        Args:
            profile: VIFSelectionProfile 또는 None.
        Returns:
            None: 미적합 선택기.
        Raises:
            None: fit에서 검증.
        """
        self.profile = profile

    def fit(self, X, y=None):
        """학습 Feature의 상수/최대 VIF를 순차 제거합니다.

        Args:
            X: 학습 수치 Feature.
            y: 사용하지 않는 타깃.
        Returns:
            VIFSelector: 선택 열과 이력을 가진 자신.
        Raises:
            ValueError: 보호 열·제거 한도·남은 Feature 오류.
        """
        raw, _ = numeric_data(X)
        profile = self.profile or VIFSelectionProfile()
        _validate(profile)
        self.feature_names_in_ = np.asarray(raw.columns, dtype=object)
        self.n_features_in_ = raw.shape[1]
        preserve = set(profile.preserve_columns or [])
        if not preserve.issubset(set(raw.columns)):
            raise ValueError("보호 Feature가 입력에 없습니다.")
        selected = list(range(raw.shape[1]))
        history = []
        for i in list(selected):
            if np.ptp(raw.values[:, i]) == 0:
                if not profile.drop_constant or raw.columns[i] in preserve:
                    raise ValueError("상수 Feature 처리/보호 규칙을 검토하세요.")
                selected.remove(i)
                history.append(
                    dict(
                        step=len(history),
                        feature=raw.columns[i],
                        vif=np.inf,
                        reason="constant",
                    )
                )
        if len(selected) < profile.min_features:
            raise ValueError("남은 Feature 수 부족")
        if profile.threshold is not None:
            while True:
                vif = vif_values(raw.values[:, selected])
                if np.max(vif) <= profile.threshold:
                    break
                eligible = [
                    j
                    for j, i in enumerate(selected)
                    if raw.columns[i] not in preserve and vif[j] > profile.threshold
                ]
                if (
                    not eligible
                    or len(selected) <= profile.min_features
                    or len(history) >= profile.max_steps
                ):
                    raise ValueError(
                        "VIF 기준을 만족하려면 보호/최소 Feature/단계 조건을 검토하세요."
                    )
                j = max(eligible, key=lambda j: vif[j])
                i = selected.pop(j)
                history.append(
                    dict(
                        step=len(history),
                        feature=raw.columns[i],
                        vif=float(vif[j]),
                        reason="vif",
                    )
                )
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.history_ = pd.DataFrame(
            history, columns=["step", "feature", "vif", "reason"]
        )
        self.final_vif_ = vif_values(raw.values[:, selected])
        return self

    def transform(self, X):
        """적합한 원 Feature 순서에서 선택 열을 적용합니다.

        Args:
            X: 같은 원 열의 새 자료.
        Returns:
            ndarray: 선택 Feature.
        Raises:
            ValueError/NotFittedError: 미적합·열 순서 오류.
        """
        check_is_fitted(self, "selected_indices_")
        raw, _ = numeric_data(X)
        if raw.shape[1] != self.n_features_in_ or (
            isinstance(X, pd.DataFrame)
            and raw.columns.tolist() != self.feature_names_in_.tolist()
        ):
            raise ValueError("학습 Feature 이름/순서와 다릅니다.")
        return raw.values[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        """선택한 Feature 이름을 전달합니다.

        Args:
            input_features: 상위 Pipeline 열 이름 또는 None.
        Returns:
            ndarray: 선택 열 순서.
        Raises:
            ValueError/NotFittedError: 이름 수·미적합 오류.
        """
        check_is_fitted(self, "selected_indices_")
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )
        if len(names) != self.n_features_in_:
            raise ValueError("원 열 수 불일치")
        return names[self.selected_indices_]


def _run(X_train, X_new, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X_train: 전용 명세의 X_train 입력.
        X_new: 전용 명세의 X_new 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        VIFSelectionOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    transformer = build_vif_selector(profile).fit(X_train)
    return VIFSelectionOutputDTO(
        transformer.transform(X_train),
        None if X_new is None else transformer.transform(X_new),
        transformer.feature_names_in_.tolist(),
        transformer.get_feature_names_out().tolist(),
        asdict(profile),
        transformer,
        len(X_train),
        [
            "VIF 제거는 표본/설계에 의존합니다. 목표 변수의 유효성 검정이 아니며 학습/CV 내부에서만 수행하세요."
        ],
        transformer.history_,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_vif_selection(
    X_train, X_new=None, profile=None, *, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X_train: 전용 명세의 X_train 입력.
        X_new: 전용 명세의 X_new 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        VIFSelectionOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-vif-selection",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate(profile)
    return _run(X_train, X_new, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
