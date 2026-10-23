"""
================================================================================
influence_filter.py - Cook 관측 검토·명시적 학습 행 처리; Pipeline 행 변환과 분리
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
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.stats.regression._common import validate_profile,numeric_data,vif_values
from src.Preprocessing._common import ProfileTransformer,execute_preprocessing


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class InfluenceFilterProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        action (str): flag/drop; 기본은 원 자료 유지. 기본 'flag'.
        cook_threshold (Optional[float]): None이면 4/n; 양수 검토 기준. 기본 None.
        max_remove_fraction (float): drop 시 최대 제거 비율 [0,1). 기본 0.1.
    """
    action: str = 'flag'
    cook_threshold: Optional[float] = None
    max_remove_fraction: float = 0.1

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class InfluenceFilterOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        cleaned_features (pd.DataFrame): 동일 Feature 순서의 유지 학습 자료.
        cleaned_target (np.ndarray): 유지한 대응 학습 타깃.
        kept_positions (np.ndarray): 원 학습 입력 내 유지 행 위치.
        removed_positions (np.ndarray): 명시적 drop으로 제거한 학습 행 위치.
        review_table (pd.DataFrame): position/row_label/cooks_distance/flag/removed.
        parameters (Dict[str,Any]): 적용 설정.
        warnings (List[str]): 관측 삭제의 근거·선택 편향·결측과의 구별.
    """
    cleaned_features: pd.DataFrame
    cleaned_target: np.ndarray
    kept_positions: np.ndarray
    removed_positions: np.ndarray
    review_table: pd.DataFrame
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
    validate_profile(profile,InfluenceFilterProfile)
    if profile.action not in ('flag','drop') or not 0<=profile.max_remove_fraction<1:raise ValueError('영향 관측 처리 설정 오류')
    if profile.cook_threshold is not None and (isinstance(profile.cook_threshold,bool) or not isinstance(profile.cook_threshold,(int,float)) or not np.isfinite(profile.cook_threshold) or profile.cook_threshold<=0):raise ValueError('Cook 기준은 양수입니다.')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run(X_train, y_train, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X_train: 전용 명세의 X_train 입력.
        y_train: 전용 명세의 y_train 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        InfluenceFilterOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    from src.stats.regression.ols import OLSProfile
    from src.stats.regression._common import fit_ols
    from statsmodels.stats.outliers_influence import OLSInfluence
    result,raw,target=fit_ols(X_train,y_train,OLSProfile());n=len(raw)
    if result.ssr<=np.finfo(float).eps*np.sum(target**2):raise ValueError('완전/거의 완전 적합의 Cook 기준은 퇴화합니다.')
    cook=np.asarray(OLSInfluence(result).cooks_distance[0]);threshold=4/n if profile.cook_threshold is None else profile.cook_threshold
    if not np.isfinite(cook).all():raise ValueError('Cook 값이 퇴화했습니다.')
    flags=cook>threshold;removed=flags if profile.action=='drop' else np.zeros(n,dtype=bool)
    if removed.mean()>profile.max_remove_fraction:raise ValueError('제거 비율 한도 초과; 검토 기준을 조정하세요.')
    kept=np.flatnonzero(~removed)
    if len(kept)<=raw.shape[1]+1 or np.linalg.matrix_rank(np.column_stack([np.ones(len(kept)),raw.values[kept]]))!=raw.shape[1]+1:raise ValueError('삭제 후 rank/잔차 자유도가 부족합니다.')
    table=pd.DataFrame({'position':np.arange(n),'row_label':list(raw.index),'cooks_distance':cook,'flag':flags,'removed':removed})
    return InfluenceFilterOutputDTO(raw.iloc[kept].copy(),target[kept],kept,np.flatnonzero(removed),table,asdict(profile),['Cook로 결측/오류를 판정하거나 자동 삭제하지 않습니다. drop을 명시한 경우에만 학습 행을 제거하며, 삭제 전후 민감도와 선택 후 추론 한계를 기록하세요.'])

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_influence_filter(X_train, y_train, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X_train: 전용 명세의 X_train 입력.
        y_train: 전용 명세의 y_train 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        InfluenceFilterOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('regression-influence-filter',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(X_train, y_train,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
