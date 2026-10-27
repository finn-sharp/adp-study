"""
================================================================================
rope.py - Rotary Position Embedding 전용 변환
================================================================================
설명:
    - 시퀀스 임베딩의 짝수 차원 회전, 부분 차원, 역회전과 sklearn 연결을 제공합니다.
분석 가정:
    - (...,sequence,embedding) 구조입니다. 범주 인코딩이나 수치 스케일링이 아닙니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    RoPE 전용 모듈·레시피·명세
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
from typing import Any, Dict, Optional, List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.MachineLearning.analysis_common import positive_int, finite_number


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class RoPEProfile:
    """명시적인 회전 위치 변환 설정.

    Attributes:
        rotary_dim (Optional[int]): 회전할 선두 짝수 차원; None이면 전체 embedding.
        base (float): 주파수 밑, 유한한 1 초과; 기본 10000.
        layout (str): interleaved 또는 split_half; 기본 interleaved.
        offset (int): 기본 위치 시작값, 0 이상; 기본 0.
        position_scale (float): 위치를 나눌 양수 척도; 기본 1.
        inverse (bool): 음의 각도로 역회전할지 여부; 기본 False.
    """

    rotary_dim: Optional[int] = None
    base: float = 10000.0
    layout: str = "interleaved"
    offset: int = 0
    position_scale: float = 1.0
    inverse: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class RoPEOutputDTO(OutputDTO):
    """RoPE 결과와 실제 위치/주파수.

    Attributes:
        preprocessed_feature (np.ndarray): 회전한 임베딩; 원 shape 유지.
        positions (np.ndarray): 적용한 1차원 위치; 배치/헤드 간 공유.
        inverse_frequencies (np.ndarray): 회전 쌍별 base^(-2i/rotary_dim).
        parameters (Dict[str, Any]): 적용 Profile.
        transformer (Any): 재사용 sklearn 변환기; JSON 제외.
        warnings (List[str]): 임베딩 해석 정보.
    """

    preprocessed_feature: np.ndarray
    positions: np.ndarray
    inverse_frequencies: np.ndarray
    parameters: Dict[str, Any]
    transformer: Any
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate(X, positions, profile):
    """임베딩·짝수 부분 차원·위치·회전 설정을 검증합니다.

    Args:
        X: (...,sequence,embedding) 실수 자료.
        positions: 공유 sequence 위치 또는 None.
        profile: RoPEProfile.
    Returns:
        tuple: 실수 입력, 위치, 실제 회전 차원.
    Raises:
        ValueError/TypeError: 자료·차원·위치·설정 오류.
    """
    if not isinstance(profile, RoPEProfile):
        raise TypeError("RoPEProfile이 필요합니다.")
    if np.iscomplexobj(X):
        raise ValueError("실수 임베딩이 필요합니다.")
    a = np.asarray(X)
    if (
        a.dtype.kind not in "fiu"
        or a.ndim < 2
        or min(a.shape) < 1
        or not np.isfinite(a).all()
    ):
        raise ValueError("유한한 시퀀스 임베딩이 필요합니다.")
    positive_int(profile.offset, "offset", 0)
    finite_number(profile.base, "base")
    finite_number(profile.position_scale, "position_scale", True)
    if (
        profile.base <= 1
        or profile.layout not in ("interleaved", "split_half")
        or not isinstance(profile.inverse, bool)
    ):
        raise ValueError("RoPE 설정 오류")
    dim = a.shape[-1] if profile.rotary_dim is None else profile.rotary_dim
    positive_int(dim, "rotary_dim")
    if dim % 2 or dim > a.shape[-1]:
        raise ValueError("rotary_dim은 embedding 이하 짝수입니다.")
    pos = (
        np.arange(a.shape[-2], dtype=float) + profile.offset
        if positions is None
        else np.asarray(positions)
    )
    if (
        pos.ndim != 1
        or len(pos) != a.shape[-2]
        or pos.dtype.kind not in "fiu"
        or not np.isfinite(pos).all()
        or (pos < 0).any()
    ):
        raise ValueError("위치는 길이가 같은 유한 비음수 1차원 자료입니다.")
    return a, pos.astype(float), dim


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _rotate(X, positions, profile):
    """짝별 회전 행렬을 임베딩에 적용합니다.

    Args:
        X: 시퀀스 임베딩.
        positions: 명시적 위치 또는 None.
        profile: 회전 설정.
    Returns:
        tuple: 회전 임베딩, 위치, 역주파수.
    Raises:
        ValueError/TypeError: 입력/각도 비유한 값.
    """
    a, pos, dim = _validate(X, positions, profile)
    frequencies = profile.base ** (-np.arange(0, dim, 2, dtype=float) / dim)
    angles = np.outer(pos / profile.position_scale, frequencies) * (
        -1 if profile.inverse else 1
    )
    if not np.isfinite(angles).all():
        raise ValueError("회전 각도 overflow")
    cosine, sine = np.cos(angles), np.sin(angles)
    cosine = cosine.reshape((1,) * (a.ndim - 2) + cosine.shape)
    sine = sine.reshape((1,) * (a.ndim - 2) + sine.shape)
    result = a.astype(float, copy=True)
    part = result[..., :dim]
    if profile.layout == "interleaved":
        first, second = part[..., ::2].copy(), part[..., 1::2].copy()
        part[..., ::2] = first * cosine - second * sine
        part[..., 1::2] = first * sine + second * cosine
    else:
        first, second = part[..., : dim // 2].copy(), part[..., dim // 2 :].copy()
        part[..., : dim // 2] = first * cosine - second * sine
        part[..., dim // 2 :] = first * sine + second * cosine
    if a.dtype.kind == "f":
        result = result.astype(a.dtype)
    if not np.isfinite(result).all():
        raise ValueError("회전 결과가 출력 dtype 범위를 초과했습니다.")
    return result, pos, frequencies


class RotaryPositionTransformer(TransformerMixin, BaseEstimator):
    """학습 통계가 없는 clone 호환 RoPE 변환기.

    Attributes:
        profile (RoPEProfile): 회전 설정.
        embedding_dim_ (int): fit 입력의 마지막 차원.
    """

    def __init__(self, profile=None):
        """회전 설정을 보존합니다.

        Args:
            profile: RoPEProfile 또는 None.
        Returns:
            None: 미적합 변환기.
        Raises:
            None: 입력은 fit/transform에서 검증.
        """
        self.profile = profile

    def fit(self, X, y=None):
        """임베딩 차원만 확인합니다. 값/타깃에서 학습하지 않습니다.

        Args:
            X: 임베딩.
            y: 사용하지 않는 타깃.
        Returns:
            RotaryPositionTransformer: 차원을 확인한 자신.
        Raises:
            ValueError/TypeError: 정의역 오류.
        """
        _validate(X, None, self.profile or RoPEProfile())
        self.embedding_dim_ = np.shape(X)[-1]
        return self

    def transform(self, X, positions=None):
        """같은 embedding 차원에 위치 회전을 적용합니다.

        Args:
            X: 임베딩; sequence 길이는 바뀔 수 있음.
            positions: 선택 공유 위치.
        Returns:
            np.ndarray: 회전 임베딩.
        Raises:
            ValueError/NotFittedError: 미적합/차원 오류.
        """
        check_is_fitted(self, "embedding_dim_")
        if np.shape(X)[-1] != self.embedding_dim_:
            raise ValueError("학습 embedding 차원과 다릅니다.")
        return _rotate(X, positions, self.profile or RoPEProfile())[0]


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def build_rope_transformer(profile=None, *, recipe_path=None, overrides=None):
    """sklearn 파이프라인용 미적합 RoPE 변환기를 생성합니다.

    Args:
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        RotaryPositionTransformer: 미적합 변환기.
    Raises:
        ValueError/TypeError/OSError: 레시피 오류.
    """
    profile = resolve_profile(
        "rope", profile, recipe_path=recipe_path, overrides=overrides
    )
    return RotaryPositionTransformer(profile)


def run_rope(X, positions=None, profile=None, *, recipe_path=None, overrides=None):
    """RoPE 전처리 결과·위치·주파수를 반환합니다.

    Args:
        X: (...,sequence,embedding) 자료.
        positions: 배치/헤드에서 공유하는 sequence 위치 또는 None.
        profile: RoPEProfile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: Profile 생략 시 재정의.
    Returns:
        RoPEOutputDTO: 원 shape의 위치 회전 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·레시피 오류.
    """
    profile = resolve_profile(
        "rope", profile, recipe_path=recipe_path, overrides=overrides
    )
    transformed, pos, frequencies = _rotate(X, positions, profile)
    return RoPEOutputDTO(
        transformed,
        pos,
        frequencies,
        asdict(profile),
        RotaryPositionTransformer(profile).fit(X),
        [],
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(run_rope(np.ones((2, 3, 4))).to_json(indent=2))
