"""
================================================================================
independence.py - 순서 자료의 BG·Ljung–Box 및 DW 기술 지표
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
class IndependenceProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        alpha (float): 개별 검정 유의수준 (0,1). 기본 0.05.
        ordered (bool): True는 시간 순·동일 간격 자료임을 호출자가 명시. 기본 False.
        method (str): breusch_godfrey/ljung_box/durbin_watson. 기본 'breusch_godfrey'.
        nlags (int): 양의 잔차 지연 수. 기본 1.
    """

    alpha: float = 0.05
    ordered: bool = False
    method: str = "breusch_godfrey"
    nlags: int = 1


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class IndependenceOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        tests (pd.DataFrame): test/statistic/p_value/reject_h0/df_num/df_den; p 미제공은 None.
        details (Dict[str,Any]): 표본 수·귀무가설·수치 해석 근거.
        parameters (Dict[str,Any]): 적용 Profile.
        warnings (List[str]): 검정 가정·선택 후 추론 정보.
    """

    tests: pd.DataFrame
    details: Dict[str, Any]
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
    validate_profile(profile, IndependenceProfile)
    if (
        profile.method not in ("breusch_godfrey", "ljung_box", "durbin_watson")
        or profile.nlags < 1
    ):
        raise ValueError("독립성 설정 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(model, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        IndependenceOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox

    result = ols_result(model)
    residual = np.asarray(result.resid)
    rows = [
        test_row(
            "Durbin-Watson descriptive", durbin_watson(residual), None, profile.alpha
        )
    ]
    notes = [
        "Durbin-Watson에는 임의 p 값/2 근처 자동 판정을 부여하지 않습니다. 무상관은 독립성의 증명이 아닙니다."
    ]
    if not profile.ordered:
        notes.append(
            "시간 순·동일 간격이 명시되지 않아 자기상관 추론 검정은 실행하지 않았습니다."
        )
    elif profile.method != "durbin_watson":
        if len(residual) <= profile.nlags + result.model.exog.shape[1] + 2:
            raise ValueError("지연 대비 표본 수 부족")
        if profile.method == "breusch_godfrey":
            lm, p, f, fp = acorr_breusch_godfrey(result, nlags=profile.nlags)
            rows.extend(
                [
                    test_row("Breusch-Godfrey LM", lm, p, profile.alpha),
                    test_row("Breusch-Godfrey F", f, fp, profile.alpha),
                ]
            )
        else:
            table = acorr_ljungbox(residual, lags=[profile.nlags], return_df=True)
            rows.append(
                test_row(
                    "Ljung-Box",
                    table.lb_stat.iloc[0],
                    table.lb_pvalue.iloc[0],
                    profile.alpha,
                    profile.nlags,
                )
            )
    return IndependenceOutputDTO(
        pd.DataFrame(rows),
        {
            "ordered": profile.ordered,
            "n_samples": len(residual),
            "hypothesis": "지정 지연에서 자기상관 없음",
        },
        asdict(profile),
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_independence(model, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        model: 전용 명세의 model 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        IndependenceOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "regression-independence", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(model, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
