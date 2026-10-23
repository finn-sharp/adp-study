"""
================================================================================
weighted_regression.py - 학습 분산 추정을 포함하는 실행가능 가중 회귀
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 학습 적합과 선택은 학습/CV 안에서 수행하고 추론 가정을 확인합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-23
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-23  김재현(Finn)     v1.0.0    응용 분석 모듈·모의10회 실행 연결
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
from src.stats.regression._common import validate_profile
from src.MachineLearning.LinearModels._weighted_estimator import FeasibleWLSRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class WeightedRegressionProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        iterations (int): 학습 log(잔차²)의 분산 모형 재추정 반복, 1~5 기본 2.
        variance_floor (float): 양수 잔차² floor와 예측 분산 floor 기본 0.0001.
        weight_ratio (float): 평균정규화 전 가중치 최대/최소 비율 상한, 1 이상 기본 100.0.
        cov_type (str): 추정 가중치에 조건부 nonrobust/HC3 공분산 기본 'HC3'.
    """

    iterations: int = 2
    variance_floor: float = 0.0001
    weight_ratio: float = 100.0
    cov_type: str = "HC3"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class WeightedRegressionOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        model (Any): CV clone 가능한 적합 FeasibleWLSRegressor; JSON 제외.
        coefficients (pd.DataFrame): 조건부 WLS coefficient·se·t·p·CI.
        weights (np.ndarray): 학습 행 순서의 추정 역분산 가중치.
        weighted_residuals (np.ndarray): sqrt(weight)×원단위 잔차; 추정 분산 모형에 조건부.
        fit_metrics (Dict[str,Any]): 원 학습 y 단위 R²/RMSE 및 조건부 WLS AIC/BIC·weighted_R².
        predictions (Optional[np.ndarray]): X_new의 원 단위 평균 예측.
        parameters (Dict[str,Any]): 설정 전체.
        warnings (List[str]): 가중치 추정·추론·정보량 비교의 한계.
    """

    model: Any
    coefficients: pd.DataFrame
    weights: np.ndarray
    weighted_residuals: np.ndarray
    fit_metrics: Dict[str, Any]
    predictions: Optional[np.ndarray]
    parameters: Dict[str, Any]
    warnings: List[str]


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
    validate_profile(profile, WeightedRegressionProfile)
    if (
        not 1 <= profile.iterations <= 5
        or profile.variance_floor <= 0
        or profile.weight_ratio < 1
        or profile.cov_type not in ("nonrobust", "HC3")
    ):
        raise ValueError("분산 모형 설정 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def build_weighted_regressor(profile=None, *, recipe_path=None, overrides=None):
    """구성만 수행하는 미적합 CV 가중 회귀 factory.

    Args:
        profile: WeightedRegressionProfile 또는 None.
        recipe_path: 사용자 JSON 또는 None.
        overrides: Profile 생략 시 일회성 재정의.
    Returns:
        FeasibleWLSRegressor: 전처리·분산 모형을 CV 학습 안에서 적합할 추정기.
    Raises:
        ValueError/TypeError/OSError: 레시피·설정 오류.
    """
    p = resolve_profile(
        "regression-weighted", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(p)
    return FeasibleWLSRegressor(**asdict(p))


def _run(X, y, X_new, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        X_new: 전용 명세의 X_new 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        WeightedRegressionOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    model = FeasibleWLSRegressor(**asdict(profile)).fit(X, y)
    native = model.model_
    ci = np.asarray(native.conf_int())
    table = pd.DataFrame(
        dict(
            term=["Intercept"] + model.feature_names_in_.tolist(),
            coefficient=np.asarray(native.params),
            standard_error=np.asarray(native.bse),
            t_statistic=np.asarray(native.tvalues),
            p_value=np.asarray(native.pvalues),
            lower=ci[:, 0],
            upper=ci[:, 1],
        )
    )
    return WeightedRegressionOutputDTO(
        model,
        table,
        model.weights_,
        np.asarray(native.wresid),
        dict(
            r2=float(r2_score(y, model.predict(X))),
            rmse=float(np.sqrt(mean_squared_error(y, model.predict(X)))),
            weighted_r2=float(native.rsquared),
            conditional_aic=float(native.aic),
            conditional_bic=float(native.bic),
        ),
        None if X_new is None else model.predict(X_new),
        asdict(profile),
        [
            "계수는 학습 대치·표준화 Feature 좌표계입니다. 학습 폴드에서만 분산 모형을 추정합니다. 가중치는 알려진 진실이 아니며 조건부 t/p/CI는 가중치 추정 불확실성을 반영하지 않습니다.",
            "weighted R²와 원 단위 예측 R²를 구별합니다. WLS 정보량을 OLS 또는 다른 가중치의 AIC/BIC와 직접 비교하지 않습니다.",
        ],
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_weighted_regression(
    X, y, X_new=None, profile=None, *, recipe_path=None, overrides=None
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        X_new: 전용 명세의 X_new 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        WeightedRegressionOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-weighted", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(X, y, X_new, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
