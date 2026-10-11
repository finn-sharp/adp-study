"""ANOVA・MANOVA・사후검정의 유한한 입력 및 범주 라벨 검증."""
from typing import Any, List
import numpy as np
import pandas as pd


def validate_alpha(alpha: float) -> None:
    """유의수준을 검증합니다.

    Args:
        alpha: bool 이외 유한한 수, 0<값<1.
    Returns:
        None: 성공 시 반환.
    Raises:
        ValueError: 잘못된 타입·유한성·범위.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha는 0과 1 사이의 유한한 수여야 합니다.")


def validate_groups(groups: Any) -> List[np.ndarray]:
    """독립 집단의 수치형 관측 목록을 검증합니다.

    Args:
        groups: 최소 두 집단의 1차원 관측 목록. 각 집단 최소 2행.
    Returns:
        List[np.ndarray]: 유한한 float 배열 목록.
    Raises:
        ValueError: 비수치·결측·무한대·차원·집단/관측 수 오류.
    """
    try:
        arrays = [np.asarray(group, dtype=float) for group in groups]
    except (TypeError, ValueError) as exc:
        raise ValueError("groups는 수치형 1차원 집단 배열의 목록이어야 합니다.") from exc
    if len(arrays) < 2 or any(g.ndim != 1 or len(g) < 2 or not np.isfinite(g).all() for g in arrays):
        raise ValueError("각 집단은 최소 2행의 유한한 1차원 수치 자료이며 최소 두 집단이 필요합니다.")
    return arrays


def validate_labels(labels: Any, n: int) -> np.ndarray:
    """결측 없는 범주형 라벨을 충돌 없이 문자열로 정규화합니다.

    Args:
        labels: 길이 n의 1차원 범주 라벨.
        n: 반응변수의 행 수.
    Returns:
        np.ndarray: 문자열 라벨 (n,). 원래 관측 순서를 보존.
    Raises:
        ValueError: 결측·무한대·차원·길이·문자열 변환 충돌 또는 단일 수준.
    """
    arr = np.asarray(labels, dtype=object)
    if arr.ndim != 1 or len(arr) != n or pd.isna(arr).any():
        raise ValueError("요인 라벨은 결측 없이 반응변수와 길이가 같은 1차원 배열이어야 합니다.")
    if any(isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in arr):
        raise ValueError("요인 라벨에 무한대를 포함할 수 없습니다.")
    normalized = np.array([str(v) for v in arr])
    if len(pd.unique(arr)) != len(np.unique(normalized)):
        raise ValueError("요인 라벨의 문자열 변환이 충돌합니다.")
    if len(np.unique(normalized)) < 2:
        raise ValueError("요인에는 최소 두 수준이 필요합니다.")
    return normalized
