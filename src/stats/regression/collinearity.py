"""
================================================================================
collinearity.py - VIF·tolerance·표준화 조건수·설계 rank
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
__author__='김재현(Finn)'
__email__='penguin.klg@gmail.com'
__version__='1.0.0'
__status__='Development'

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass,asdict
from typing import Any,Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from scipy import stats
from src.stats.regression._common import validate_profile,numeric_data,fit_ols,ols_result,test_row,vif_values


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class CollinearityProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        vif_threshold (float): 1 초과 VIF 검토 기준; 자동 삭제 아님. 기본 10.0.
    """
    vif_threshold: float = 10.0

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CollinearityOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        table (pd.DataFrame): feature/vif/tolerance/flag; VIF는 검정 p가 아님.
        condition_number (float): 중심화·표준화 Feature 설계의 조건수; 상수 제외.
        rank (int): 절편 포함 원 설계 rank.
        parameters (Dict[str,Any]): 적용 설정.
        warnings (List[str]): 공선성·단위·구조적 rank 정보.
    """
    table: pd.DataFrame
    condition_number: float
    rank: int
    parameters: Dict[str,Any]
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
    validate_profile(profile,CollinearityProfile)
    if profile.vif_threshold<=1:raise ValueError('VIF 기준은 1 초과입니다.')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run(X, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        CollinearityOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    raw,_=numeric_data(X);vif=vif_values(raw.values);table=pd.DataFrame({'feature':raw.columns,'vif':vif,'tolerance':1/vif,'flag':vif>profile.vif_threshold})
    nonconstant=np.ptp(raw.values,axis=0)>0;Z=raw.values[:,nonconstant]
    if Z.shape[1]:Z=(Z-Z.mean(axis=0))/Z.std(axis=0);condition=float(np.linalg.cond(np.column_stack([np.ones(len(Z)),Z])))
    else:condition=np.inf
    rank=int(np.linalg.matrix_rank(np.column_stack([np.ones(len(raw)),raw.values])))
    return CollinearityOutputDTO(table,condition,rank,asdict(profile),['VIF는 Feature 간 선형 중복 진단입니다. Cook 영향도와 다른 문제이며 도메인상 필수 변수·더미 기준을 먼저 확인하세요.'])

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_collinearity(X, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        CollinearityOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('regression-collinearity',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(X,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
