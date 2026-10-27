"""
================================================================================
influence.py - Cook·leverage·studentized 잔차·DFFITS·DFBETAS
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
from scipy import stats
from src.stats.regression._common import (
    validate_profile,
    numeric_data,
    fit_ols,
    ols_result,
    test_row,
    vif_values,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class InfluenceProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
        cook_threshold (Optional[float]): None이면 4/n 검토 기준. 기본 None.
        leverage_multiplier (float): 평균 leverage p/n에 곱할 양수 배율. 기본 2.0.
        studentized_threshold (float): 외적 studentized 잔차의 양수 절댓값 기준. 기본 3.0.
        p_adjust (str): outlier p의 bonferroni/holm/fdr_bh/none. 기본 'bonferroni'.
        max_samples (int): leave-one-out 영향 계산 표본 한도. 기본 3000.
    """

    alpha: float = 0.05
    cook_threshold: Optional[float] = None
    leverage_multiplier: float = 2.0
    studentized_threshold: float = 3.0
    p_adjust: str = "bonferroni"
    max_samples: int = 3000


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class InfluenceOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        table (pd.DataFrame): position/row_label/residual/studentized_internal/studentized_external/leverage/cooks_distance/dffits/outlier_p/adjusted_outlier_p 및 flag.
        dfbetas (pd.DataFrame): 원 행×설계 항의 계수 변화 표준화 지표.
        thresholds (Dict[str,float]): Cook·leverage·studentized·DFFITS 검토 기준.
        parameters (Dict[str,Any]): 적용 설정.
        warnings (List[str]): 영향·오류/결측의 구별과 선택 편향.
    """

    table: pd.DataFrame
    dfbetas: pd.DataFrame
    thresholds: Dict[str, float]
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
    validate_profile(profile, InfluenceProfile)
    if (
        profile.leverage_multiplier <= 0
        or profile.studentized_threshold <= 0
        or profile.max_samples < 4
        or profile.p_adjust not in ("bonferroni", "holm", "fdr_bh", "none")
    ):
        raise ValueError("영향도 설정 오류")
    if profile.cook_threshold is not None and (
        isinstance(profile.cook_threshold, bool)
        or not isinstance(profile.cook_threshold, (int, float))
        or not np.isfinite(profile.cook_threshold)
        or profile.cook_threshold <= 0
    ):
        raise ValueError("Cook 기준은 양수입니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(model, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        InfluenceOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    from statsmodels.stats.outliers_influence import OLSInfluence
    from src.stats.correlation._common import adjusted_pvalues

    result = ols_result(model)
    n = int(result.nobs)
    p = result.model.exog.shape[1]
    if (
        n > profile.max_samples
        or result.df_resid <= 1
        or result.ssr <= np.finfo(float).eps * np.sum(result.model.endog**2)
    ):
        raise ValueError("외적 영향도 자유도/잔차분산/표본 한도 오류")
    influence = OLSInfluence(result)
    leverage = np.asarray(influence.hat_matrix_diag)
    if (leverage >= 1 - 1e-12).any():
        raise ValueError("leverage≈1 관측은 leave-one-out 진단이 퇴화합니다.")
    external = np.asarray(influence.resid_studentized_external)
    cook = np.asarray(influence.cooks_distance[0])
    internal = np.asarray(influence.resid_studentized_internal)
    dffits = np.asarray(influence.dffits[0])
    outlier_p = 2 * stats.t.sf(np.abs(external), result.df_resid - 1)
    adjusted = adjusted_pvalues(outlier_p, profile.p_adjust)
    thresholds = {
        "cook": 4 / n if profile.cook_threshold is None else profile.cook_threshold,
        "leverage": profile.leverage_multiplier * p / n,
        "studentized": profile.studentized_threshold,
        "dffits": 2 * np.sqrt(p / n),
    }
    labels = result.model.data.row_labels
    if labels is None:
        labels = np.arange(n)
    table = pd.DataFrame(
        {
            "position": np.arange(n),
            "row_label": list(labels),
            "residual": np.asarray(result.resid),
            "studentized_internal": internal,
            "studentized_external": external,
            "leverage": leverage,
            "cooks_distance": cook,
            "dffits": dffits,
            "outlier_p": outlier_p,
            "adjusted_outlier_p": adjusted,
            "cook_flag": cook > thresholds["cook"],
            "leverage_flag": leverage > thresholds["leverage"],
            "studentized_flag": np.abs(external) > thresholds["studentized"],
            "outlier_reject": adjusted < profile.alpha,
        }
    )
    return InfluenceOutputDTO(
        table,
        pd.DataFrame(influence.dfbetas, index=labels, columns=result.model.exog_names),
        thresholds,
        asdict(profile),
        [
            "Cook/leverage 기준은 검토용 휴리스틱이며 데이터 오류·결측 판정이 아닙니다. 삭제 후 추론과 R²에는 선택 편향이 생기므로 근거와 민감도 비교를 남기세요."
        ],
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_influence(model, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        InfluenceOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-influence", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(model, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
