"""
================================================================================
numerical.py - 최소제곱법의 정규방정식/SVD/TSVD와 릿지 폐형식 풀이
================================================================================
설명: 02/05장의 수치 풀이를 비교할 수 있는 교육용 실행기입니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 수치 풀이와 특이값 진단 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from src.MachineLearning.common import validate_features


@dataclass
class NumericalProfile:
    """수치 풀이 조건.

    Attributes:
        method: normal, svd, tsvd, ridge.
        fit_intercept: 절편 열을 추가할지 여부. 릿지에서는 절편을 규제하지 않음.
        n_components: TSVD에서 유지할 특이값 수. 1 이상 min(행, 열) 이하.
        rcond: 유효 특이값 판정 기준 (최대 특이값의 비율, 양수).
        alpha: 릿지 L2 규제 강도 (양수).
    """
    method: str = "svd"
    fit_intercept: bool = True
    n_components: int = 2
    rcond: float = 1e-12
    alpha: float = 1.0


@dataclass
class NumericalOutputDTO(OutputDTO):
    """교육용 수치 회귀 결과.

    Attributes:
        coefficients: 입력 특성 순서의 계수.
        intercept: 절편. fit_intercept=False이면 0.
        predictions: X_predict를 전달하면 해당 행의 예측, 생략하면 학습 적합값.
        singular_values: 절편 포함 설계행렬의 특이값.
        rank: rcond에 따른 설계행렬 유효 계수 (TSVD 유지 수와 다를 수 있음).
        condition_number: 설계행렬 조건수. 최소 특이값이 0이면 inf.
    """
    coefficients: np.ndarray
    intercept: float
    predictions: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
NumericalResult = NumericalOutputDTO


def run_numerical_regression(X: Any, y: Any, profile: Optional[NumericalProfile] = None,
                              X_predict: Any = None) -> NumericalOutputDTO:
    """교육용 회귀 풀이를 수행합니다. 자동 분할·일반화 평가를 하지 않습니다.

    Args:
        X: 적합용 수치형 2차원 특성.
        y: 길이가 같은 유한한 1차원 수치 타깃.
        profile: 풀이 조건.
        X_predict: 선택적 예측 특성. X와 같은 열 수·순서.
    Returns:
        NumericalOutputDTO: 계수, 예측 및 수치 안정성 진단.
    Raises:
        ValueError: 잘못된 method, 차원, rcond/alpha 또는 TSVD 유지 수.
        np.linalg.LinAlgError: 정규방정식에서 특이한 행렬을 만난 경우.

    Notes:
        정규방정식은 조건수가 큰 데이터에 불안정합니다. 기본은 SVD 풀이입니다.
    """
    p = profile if profile is not None else NumericalProfile()
    arr = validate_features(X)
    target = np.asarray(y, dtype=float)
    if target.ndim != 1 or len(target) != len(arr) or not np.isfinite(target).all():
        raise ValueError("y는 X와 길이가 같은 유한한 1차원 수치여야 합니다.")
    if not np.isfinite(p.rcond) or not 0 < p.rcond < 1:
        raise ValueError("rcond는 0과 1 사이여야 합니다.")
    design = np.column_stack([np.ones(len(arr)), arr]) if p.fit_intercept else arr
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    valid = s > p.rcond * s[0]
    if p.method == "normal":
        weights = np.linalg.solve(design.T @ design, design.T @ target)
    elif p.method == "ridge":
        if not np.isfinite(p.alpha) or p.alpha <= 0:
            raise ValueError("ridge alpha는 양수여야 합니다.")
        penalty = np.eye(design.shape[1]) * p.alpha
        if p.fit_intercept:
            penalty[0, 0] = 0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    elif p.method in ("svd", "tsvd"):
        keep = valid.copy()
        if p.method == "tsvd":
            if not isinstance(p.n_components, int) or not 1 <= p.n_components <= len(s):
                raise ValueError("TSVD 유지 수는 1 이상 특이값 수 이하여야 합니다.")
            keep[p.n_components:] = False
        inv = np.zeros_like(s)
        inv[keep] = 1 / s[keep]
        weights = vt.T @ (inv * (u.T @ target))
    else:
        raise ValueError("method는 normal, svd, tsvd, ridge 중 하나여야 합니다.")
    intercept = float(weights[0]) if p.fit_intercept else 0.0
    coefficients = weights[1:] if p.fit_intercept else weights
    pred_arr = arr if X_predict is None else validate_features(X_predict, min_samples=1)
    if pred_arr.shape[1] != arr.shape[1]:
        raise ValueError("예측 입력은 학습 입력과 같은 열 수여야 합니다.")
    return NumericalOutputDTO(coefficients, intercept, pred_arr @ coefficients + intercept, s,
                           int(valid.sum()), float(s[0] / s[-1]) if s[-1] > 0 else float("inf"))


if __name__ == "__main__":
    X = np.arange(12).reshape(6, 2)
    print(run_numerical_regression(X, np.arange(6)).coefficients)
