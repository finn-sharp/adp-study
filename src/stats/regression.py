"""
================================================================================
regression.py - OLS/Logit 추론, 회귀 진단 및 정보기준 변수 선택
================================================================================
설명: 통계분석 PDF의 회귀계수 추론, VIF, 잔차 진단, 전진/후진/단계적 선택을 재구성합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 회귀 추론과 변수 선택 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.MachineLearning.common import validate_features
from src.stats.analysis import numeric_sample


@dataclass
class RegressionProfile:
    """통계적 회귀 조건.

    Attributes:
        method: ols 또는 logit (타깃은 0/1).
        alpha: 계수 신뢰구간의 유의수준 (0, 1).
        fit_intercept: 절편 포함 여부.
        polynomial_degree: 다항/교호 특성 차수 (1 이상). 기본 1.
        max_iter: Logit 최대 반복 수.
    """
    method: str = "ols"
    alpha: float = 0.05
    fit_intercept: bool = True
    polynomial_degree: int = 1
    max_iter: int = 100


@dataclass
class RegressionOutputDTO(OutputDTO):
    """전체 입력의 회귀 추론 결과. 보류 테스트 성능이 아닙니다.

    Attributes:
        model: statsmodels 적합 결과 객체. 예측 시 같은 설계행렬 구성이 필요합니다.
        design: 적합에 사용한 DataFrame (절편 및 선택적 다항 특성 포함).
        transformer: 다항 변환기. degree=1이면 None.
        coefficient_table: 계수, 표준오차, p-value, 신뢰구간.
        diagnostics: OLS의 Durbin-Watson, 잔차 정규성, Breusch-Pagan, VIF;
            Logit은 AIC/BIC/LLR p-value와 pseudo_r2 (OLS R²와 다른 지표).
        residuals: 입력 순서의 잔차.
    """
    model: Any
    design: pd.DataFrame
    transformer: Any
    coefficient_table: pd.DataFrame
    diagnostics: Dict[str, Any]
    residuals: np.ndarray


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
RegressionResult = RegressionOutputDTO


def run_regression_inference(X: Any, y: Any,
                              profile: Optional[RegressionProfile] = None) -> RegressionOutputDTO:
    """회귀계수 추론과 잔차 진단을 수행합니다.

    Args:
        X: 유한한 수치 특성. DataFrame이면 고유한 열 이름을 사용합니다.
        y: 1차원 수치 타깃. Logit은 두 클래스 0/1.
        profile: 회귀 추론 설정.
    Returns:
        RegressionOutputDTO: 적합 모델, 계수 표 및 진단.
    Raises:
        ValueError: 잘못된 입력, 순위 부족, 잔차 자유도 부족 또는 Logit 타깃 오류.
        ImportError: statsmodels가 설치되지 않은 경우.

    Notes:
        OLS 계수 추론은 모형·오차 가정에 의존합니다. 잔차 검정만으로 가정을 보장하지 않습니다.
        변수 선택 후 같은 데이터의 p-value는 선택 과정의 불확실성을 반영하지 않습니다.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from scipy import stats
    from sklearn.preprocessing import PolynomialFeatures
    p = profile if profile is not None else RegressionProfile()
    if p.method not in ("ols", "logit") or not 0 < p.alpha < 1:
        raise ValueError("method는 ols/logit, alpha는 (0, 1)이어야 합니다.")
    if not isinstance(p.polynomial_degree, int) or p.polynomial_degree < 1:
        raise ValueError("polynomial_degree는 양의 정수여야 합니다.")
    arr, target = validate_features(X), numeric_sample(y)
    if len(arr) != len(target):
        raise ValueError("X와 y 길이는 같아야 합니다.")
    names = [str(v) for v in X.columns] if hasattr(X, "columns") else ["x{}".format(i) for i in range(arr.shape[1])]
    if len(set(names)) != len(names) or "const" in names:
        raise ValueError("특성 이름은 중복 없이 const 이외의 이름이어야 합니다.")
    transformer = None
    if p.polynomial_degree > 1:
        transformer = PolynomialFeatures(degree=p.polynomial_degree, include_bias=False)
        arr = transformer.fit_transform(arr)
        name_method = getattr(transformer, "get_feature_names_out", None) or transformer.get_feature_names
        names = list(name_method(names))
    design = pd.DataFrame(arr, columns=names)
    if p.fit_intercept:
        design.insert(0, "const", 1.0)
    if len(design) <= design.shape[1] or np.linalg.matrix_rank(design.values) < design.shape[1]:
        raise ValueError("추론에는 완전 열 순위와 양의 잔차 자유도가 필요합니다.")
    if p.method == "ols":
        model = sm.OLS(target, design).fit()
        residuals = np.asarray(model.resid)
        diagnostics = {"aic": float(model.aic), "bic": float(model.bic), "r2": float(model.rsquared),
                       "durbin_watson": float(durbin_watson(residuals)),
                       "vif": pd.Series([variance_inflation_factor(design.values, i)
                                         for i in range(design.shape[1])], index=design.columns)}
        if len(residuals) >= 8 and np.ptp(residuals):
            diagnostics["residual_normality_p"] = float(stats.normaltest(residuals).pvalue)
        if p.fit_intercept and design.shape[1] > 1:
            diagnostics["breusch_pagan_p"] = float(het_breuschpagan(residuals, design)[1])
    else:
        if not np.array_equal(np.unique(target), [0, 1]):
            raise ValueError("Logit 타깃은 0과 1의 두 클래스여야 합니다.")
        model = sm.Logit(target, design).fit(disp=False, maxiter=p.max_iter)
        residuals = np.asarray(model.resid_response)
        diagnostics = {"aic": float(model.aic), "bic": float(model.bic), "pseudo_r2": float(model.prsquared),
                       "llr_p": float(model.llr_pvalue), "converged": bool(model.mle_retvals.get("converged", False))}
    interval = np.asarray(model.conf_int(alpha=p.alpha))
    table = pd.DataFrame({"coefficient": np.asarray(model.params), "std_error": np.asarray(model.bse),
                          "p_value": np.asarray(model.pvalues), "ci_lower": interval[:, 0],
                          "ci_upper": interval[:, 1]}, index=design.columns)
    return RegressionOutputDTO(model, design, transformer, table, diagnostics, residuals)


@dataclass
class SelectionProfile:
    """OLS 정보기준 변수 선택 조건.

    Attributes:
        method: forward, backward, stepwise.
        criterion: aic 또는 bic (작을수록 좋음).
        max_steps: 반복 상한. 완료 여부는 result.converged에 반환합니다.
        min_improvement: 최소 정보기준 개선량 (양수).
    """
    method: str = "stepwise"
    criterion: str = "aic"
    max_steps: int = 100
    min_improvement: float = 1e-8


@dataclass
class SelectionOutputDTO(OutputDTO):
    """변수 선택 결과.

    Attributes:
        selected_features: 선택한 입력 열 이름.
        model: 절편 포함 statsmodels OLS 적합 객체.
        history: 단계별 선택 특성과 정보기준 기록.
        converged: 상한 도달 전에 개선 후보가 없어 종료했는지 여부.
    """
    selected_features: List[str]
    model: Any
    history: List[Dict[str, Any]]
    converged: bool


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
SelectionResult = SelectionOutputDTO


def run_variable_selection(X: Any, y: Any, profile: Optional[SelectionProfile] = None) -> SelectionOutputDTO:
    """전진·후진·단계적 OLS 변수 선택을 수행합니다.

    Args:
        X: 학습용 수치 특성. DataFrame의 열 이름은 고유해야 합니다.
        y: 학습용 수치 타깃.
        profile: 선택 알고리즘 및 정보기준.
    Returns:
        SelectionOutputDTO: 최종 특성, 적합 객체, 이력 및 종료 상태.
    Raises:
        ValueError: 잘못된 조건, 입력 또는 초기 모형의 순위·자유도 부족.

    Notes:
        이 실행기는 탐색적 변수 선택입니다. 일반화 평가는 별도 보류 데이터에서 수행하세요.
    """
    import statsmodels.api as sm
    p = profile if profile is not None else SelectionProfile()
    if p.method not in ("forward", "backward", "stepwise") or p.criterion not in ("aic", "bic"):
        raise ValueError("method 또는 criterion을 확인하세요.")
    if not isinstance(p.max_steps, int) or p.max_steps < 1 or not np.isfinite(p.min_improvement) or p.min_improvement <= 0:
        raise ValueError("max_steps와 min_improvement는 양수여야 합니다.")
    arr, target = validate_features(X), numeric_sample(y)
    if len(arr) != len(target):
        raise ValueError("X와 y 길이는 같아야 합니다.")
    names = [str(c) for c in X.columns] if hasattr(X, "columns") else ["x{}".format(i) for i in range(arr.shape[1])]
    if len(set(names)) != len(names) or "const" in names:
        raise ValueError("고유한 특성 이름이 필요하며 const는 예약된 이름입니다.")
    def fit(cols):
        design = np.column_stack([np.ones(len(arr)), arr[:, cols]])
        if len(arr) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
            return None
        return sm.OLS(target, pd.DataFrame(design, columns=["const"] + [names[i] for i in cols])).fit()
    selected = list(range(arr.shape[1])) if p.method == "backward" else []
    model = fit(selected)
    if model is None:
        raise ValueError("초기 모형의 순위 또는 잔차 자유도가 부족합니다.")
    history = [{"features": [names[i] for i in selected], p.criterion: float(getattr(model, p.criterion))}]
    for _ in range(p.max_steps):
        candidates = []
        if p.method != "backward":
            candidates.extend([selected + [i] for i in range(arr.shape[1]) if i not in selected])
        if p.method != "forward":
            candidates.extend([[j for j in selected if j != i] for i in selected])
        best, best_cols = model, selected
        for cols in candidates:
            candidate = fit(cols)
            if candidate is not None and np.isfinite(getattr(candidate, p.criterion)) and getattr(candidate, p.criterion) < getattr(best, p.criterion) - p.min_improvement:
                best, best_cols = candidate, cols
        if best_cols == selected:
            return SelectionOutputDTO([names[i] for i in selected], model, history, True)
        model, selected = best, best_cols
        history.append({"features": [names[i] for i in selected], p.criterion: float(getattr(model, p.criterion))})
    return SelectionOutputDTO([names[i] for i in selected], model, history, False)
