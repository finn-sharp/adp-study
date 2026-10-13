"""
================================================================================
cv_summary.py - 폴드별 평가값의 평균·표준편차·표본 수
================================================================================
설명:
    - 이미 계산한 CV 지표의 수치 분포를 요약합니다. 학습/CV 분할은 수행하지 않습니다.
분석 가정:
    - 폴드별 점수는 같은 지표 정의이며 표준편차는 신뢰구간이 아닙니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-13
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-13  김재현(Finn)      v1.0.0    CV 요약 메트릭
================================================================================
"""
__author__ = '김재현(Finn)'
__email__ = 'penguin.klg@gmail.com'
__version__ = '1.0.0'
__status__ = 'Development'

# 1. Imports
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes.loader import resolve_profile

# 2. Input Profile
@dataclass
class CVSummaryProfile:
    """CV 요약 설정.

    Attributes:
        ddof (int): 표준편차 자유도 보정; 0 또는 1.
        nan_policy (str): raise/omit; 정의 불가 폴드 명시 처리.
    """
    ddof: int = 1
    nan_policy: str = 'raise'

# 3. Output DTO
@dataclass
class CVSummaryOutputDTO(OutputDTO):
    """CV 요약 결과.

    Attributes:
        table (DataFrame): metric/count/mean/std/min/max/omitted.
        parameters (dict): 실제 요약 설정.
        warnings (list): 정의 불가 표준편차 및 해석 제한.
    """
    table: Any
    parameters: dict
    warnings: list

# 4. Input Validation
def validate_inputs(scores,p):
    """지표별 1D 폴드 점수를 검증합니다.

    Args:
        scores: 지표명→폴드 점수 사전.
        p: 전용 요약 Profile.
    Returns:
        None: 정상 입력.
    Raises:
        ValueError: 사전/설정 오류.
    """
    if not isinstance(p,CVSummaryProfile) or not isinstance(p.ddof,int) or isinstance(p.ddof,bool) or p.ddof not in (0,1) or p.nan_policy not in ('raise','omit'):
        raise ValueError('ddof=0/1, nan_policy=raise/omit')
    if not isinstance(scores,dict) or not scores or not all(isinstance(k,str) for k in scores):
        raise ValueError('지표명→폴드 점수 사전 필요')

# 5. Individual Model Runners
def _summary(scores,p):
    """유효 폴드만 명시적으로 요약합니다.

    Args:
        scores: 폴드 점수 사전.
        p: 요약 설정.
    Returns:
        CVSummaryOutputDTO: 요약표.
    Raises:
        ValueError: 1D/빈 배열/Infinity/NaN 오류.
    """
    rows=[];warnings=['폴드 표준편차는 일반화 오차의 신뢰구간이 아닙니다.']
    for name,values in scores.items():
        a=np.asarray(values,dtype=float)
        if a.ndim!=1 or not len(a) or np.any(np.isinf(a)): raise ValueError('점수는 비어 있지 않은 1D 배열; Infinity 불가')
        omitted=int(np.isnan(a).sum())
        if omitted and p.nan_policy=='raise': raise ValueError('NaN 폴드 점수')
        a=a[~np.isnan(a)]
        if not len(a): raise ValueError('유효 폴드가 없습니다.')
        std=float(a.std(ddof=p.ddof)) if len(a)>p.ddof else float('nan')
        if len(a)<=p.ddof: warnings.append(name+': 표준편차 정의 불가')
        rows.append((name,len(a),float(a.mean()),std,float(a.min()),float(a.max()),omitted))
    return CVSummaryOutputDTO(pd.DataFrame(rows,columns=['metric','count','mean','std','min','max','omitted']),{'ddof':p.ddof,'nan_policy':p.nan_policy},warnings)

# 6. Public Runner
def run_cv_summary(scores,profile=None,*,recipe_path=None,overrides=None):
    """지표별 폴드 점수를 요약합니다.

    Args:
        scores: 지표명→폴드 점수 사전.
        profile: CVSummaryProfile 또는 None.
        recipe_path: 사용자 JSON.
        overrides: JSON 재정의.
    Returns:
        CVSummaryOutputDTO: 수치 요약표.
    Raises:
        ValueError/TypeError/OSError: 입력/설정/레시피 오류.
    """
    p=resolve_profile('metric-cv-summary',profile,recipe_path,overrides=overrides)
    validate_inputs(scores,p)
    return _summary(scores,p)

# 7. Example Execution
if __name__=='__main__':
    print(run_cv_summary({'f1':[.7,.8,.9]}).to_json(indent=2))
