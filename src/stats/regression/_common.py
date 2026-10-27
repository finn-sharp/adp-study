"""
================================================================================
_common.py - OLS 추론·행 대응·설계 rank·반환 검정표의 공통 계약.
================================================================================
설명:
    - OLS 추론·행 대응·설계 rank·반환 검정표의 공통 계약.
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

from dataclasses import fields
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.Preprocessing._common import feature_array
from src.MachineLearning.analysis_common import finite_number


def validate_profile(profile, cls, check_alpha=True):
    """전용 필드 기본 타입과 alpha를 확인합니다.

    Args:
        profile: 전용 설정.
        cls: 전용 클래스.
        check_alpha: alpha를 검정 유의수준 (0,1)로 확인할지 여부.
    Returns:
        None: 유효 설정.
    Raises:
        ValueError/TypeError: 타입·유의수준 오류.
    """
    if not isinstance(profile, cls):
        raise TypeError("전용 Profile이 필요합니다.")
    for field in fields(profile):
        value = getattr(profile, field.name)
        if field.type is bool and not isinstance(value, bool):
            raise TypeError(field.name + " bool 오류")
        if field.type is int and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise TypeError(field.name + " int 오류")
        if field.type is str and not isinstance(value, str):
            raise TypeError(field.name + " str 오류")
        if field.type is float:
            finite_number(value, field.name)
    if check_alpha and hasattr(profile, "alpha") and not 0 < profile.alpha < 1:
        raise ValueError("검정 alpha는 (0,1)입니다.")


def numeric_data(X, y=None):
    """유한 원 Feature·타깃과 대응 인덱스를 보존합니다.

    Args:
        X: 유한 수치 DataFrame/2차원 배열.
        y: 대응 타깃 또는 None.
    Returns:
        tuple: 수치 DataFrame, 선택 1차원 타깃.
    Raises:
        ValueError/TypeError: 자료·행 대응 오류.
    """
    a, names = feature_array(X, True, False)
    raw = pd.DataFrame(
        a, index=X.index if isinstance(X, pd.DataFrame) else None, columns=names
    )
    if y is None:
        return raw, None
    if (
        isinstance(X, pd.DataFrame)
        and isinstance(y, pd.Series)
        and not X.index.equals(y.index)
    ):
        raise ValueError("Feature와 타깃 인덱스가 다릅니다.")
    if np.iscomplexobj(y):
        raise ValueError("실수 타깃이 필요합니다.")
    target = (
        y.to_numpy(dtype=float, na_value=np.nan)
        if isinstance(y, pd.Series)
        else np.asarray(y, dtype=float)
    )
    if (
        target.ndim != 1
        or len(target) != len(raw)
        or not np.isfinite(target).all()
        or np.ptp(target) == 0
    ):
        raise ValueError("대응 비상수 유한 타깃이 필요합니다.")
    return raw, target


def fit_ols(X, y, profile):
    """명시 학습 자료의 full-rank OLS를 적합합니다.

    Args:
        X: 유한 Feature.
        y: 대응 비상수 타깃.
        profile: fit_intercept/cov_type/hac_maxlags를 가진 설정.
    Returns:
        tuple: statsmodels OLS 결과, 원 Feature DataFrame, 타깃.
    Raises:
        ValueError/TypeError: 상수·rank·잔차 자유도·공분산 오류.
    """
    raw, target = numeric_data(X, y)
    if profile.fit_intercept:
        if "Intercept" in raw.columns or (np.ptp(raw.values, axis=0) == 0).any():
            raise ValueError("절편과 중복되는 상수 Feature를 제거하세요.")
        design = raw.copy()
        design.insert(0, "Intercept", 1.0)
    else:
        design = raw
    if (
        len(design) <= design.shape[1]
        or np.linalg.matrix_rank(design.values) != design.shape[1]
    ):
        raise ValueError("full-rank 설계와 양의 잔차 자유도가 필요합니다.")
    options = {}
    if profile.cov_type == "HAC":
        options = {"maxlags": profile.hac_maxlags}
    result = sm.OLS(
        target, design, hasconst=profile.fit_intercept, missing="raise"
    ).fit(cov_type=profile.cov_type, cov_kwds=options, use_t=True)
    return result, raw, target


def ols_result(model):
    """전용 DTO 또는 full-rank 적합 OLS 결과를 확인합니다.

    Args:
        model: OLS Output DTO 또는 statsmodels OLS 결과.
    Returns:
        object: 적합 OLS 결과.
    Raises:
        ValueError/TypeError: 미적합·다른 모델·rank/자유도 오류.
    """
    result = (
        model.model
        if not hasattr(model, "resid") and hasattr(model, "model")
        else model
    )
    if not hasattr(result, "resid") or not isinstance(
        getattr(result, "model", None), sm.OLS
    ):
        raise TypeError("적합 statsmodels OLS 또는 전용 DTO가 필요합니다.")
    if (
        result.df_resid <= 0
        or np.linalg.matrix_rank(result.model.exog) != result.model.exog.shape[1]
    ):
        raise ValueError("유효 잔차 자유도와 full-rank OLS가 필요합니다.")
    return result


def test_row(name, statistic, p_value, alpha, df_num=None, df_den=None):
    """검정의 유의성·자유도와 p 미제공 상태를 구분합니다.

    Args:
        name: 검정 이름.
        statistic: 수치 통계량.
        p_value: p 또는 None.
        alpha: 유의수준.
        df_num: 분자 자유도 또는 None.
        df_den: 분모 자유도 또는 None.
    Returns:
        dict: 검정표 한 행.
    Raises:
        ValueError: 잘못된 수치 결과.
    """
    statistic = float(np.asarray(statistic).squeeze())
    if not np.isfinite(statistic):
        raise ValueError("검정 통계량이 유한하지 않습니다.")
    if p_value is not None:
        p_value = float(np.asarray(p_value).squeeze())
        if not np.isfinite(p_value) or not 0 <= p_value <= 1:
            raise ValueError("유효 p 값이 아닙니다.")
    return dict(
        test=name,
        statistic=statistic,
        p_value=p_value,
        reject_h0=None if p_value is None else bool(p_value < alpha),
        df_num=df_num,
        df_den=df_den,
    )


def vif_values(X):
    """절편을 둔 보조회귀의 Feature별 VIF를 계산합니다.

    Args:
        X: 유한 2차원 수치 Feature 배열.
    Returns:
        ndarray: 상수/완전 공선성은 inf; 한 비상수 Feature는 1.
    Raises:
        ValueError: 비어 있거나 비유한 자료.
    """
    a = np.asarray(X, dtype=float)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("VIF 수치 행렬 오류")
    centered = a - a.mean(axis=0)
    norm = np.sqrt(np.sum(centered**2, axis=0))
    active = norm > 0
    if active.any():
        normalized = centered[:, active] / norm[active]
        correlation = normalized.T @ normalized
        # Well-conditioned full-rank designs: diag(inv(correlation)) equals auxiliary-regression VIF.
        if np.linalg.eigvalsh(correlation).min() > 1e-10:
            result = np.full(a.shape[1], np.inf)
            result[active] = np.maximum(1.0, np.diag(np.linalg.inv(correlation)))
            return result
    output = []
    for i in range(a.shape[1]):
        target = a[:, i]
        centered = target - target.mean()
        total = float(centered @ centered)
        if total == 0:
            output.append(np.inf)
            continue
        other = np.column_stack([np.ones(len(a)), np.delete(a, i, axis=1)])
        beta = np.linalg.lstsq(other, target, rcond=None)[0]
        residual = target - other @ beta
        ratio = float(residual @ residual / total)
        output.append(
            np.inf if ratio <= np.finfo(float).eps * 10 else max(1.0, 1.0 / ratio)
        )
    return np.asarray(output)
