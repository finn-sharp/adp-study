"""
================================================================================
information_selection.py - 동일 반응 OLS AIC/BIC 후진 변수 선택
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
from src.stats.regression._common import validate_profile,numeric_data,fit_ols
from src.stats.regression.ols import OLSProfile

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class InformationSelectionProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        criterion (str): 동일 반응·동일 행 OLS의 aic/bic 기본 'bic'.
        min_features (int): 남길 최소 Feature 수, 1 이상 기본 1.
        max_steps (int): 최대 후진 제거 단계, 0 이상 기본 30.
        improvement (float): 정보량 감소 허용값, 비음수 기본 1e-08.
    """
    criterion: str = 'bic'
    min_features: int = 1
    max_steps: int = 30
    improvement: float = 1e-08

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class InformationSelectionOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        selector (Any): 적합 InformationSelector; JSON 제외.
        model (Any): 최종 조건부 statsmodels OLS; JSON 제외.
        selected_features (List[str]): 원 열 순서의 선택 Feature.
        history (pd.DataFrame): 단계·제거 열·aic/bic·채택·후보 기록.
        parameters (Dict[str,Any]): 적용 설정.
        warnings (List[str]): 탐욕 선택·선택 후 추론·척도 비교 한계.
    """
    selector: Any
    model: Any
    selected_features: List[str]
    history: pd.DataFrame
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
    validate_profile(profile,InformationSelectionProfile)
    if profile.criterion not in ('aic','bic') or profile.min_features<1 or profile.max_steps<0 or profile.improvement<0:raise ValueError('선택 기준/단계 오류')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
class InformationSelector(TransformerMixin,BaseEstimator):
    """동일 학습 행의 OLS 정보량 기준으로 후진 선택하는 CV 변환기."""
    def __init__(self,criterion='bic',min_features=1,max_steps=30,improvement=1e-8):
        """학습/분석 단계의 계산.
        
        Args:
        self: 명세의 단계 입력.
        criterion: 명세의 단계 입력.
        min_features: 명세의 단계 입력.
        max_steps: 명세의 단계 입력.
        improvement: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        self.criterion=criterion;self.min_features=min_features;self.max_steps=max_steps;self.improvement=improvement
    def fit(self,X,y):
        """학습/분석 단계의 계산.
        
        Args:
        self: 명세의 단계 입력.
        X: 명세의 단계 입력.
        y: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        raw,target=numeric_data(X,y)
        profile=InformationSelectionProfile(self.criterion,self.min_features,self.max_steps,self.improvement);_validate(profile)
        if raw.shape[1]<self.min_features:raise ValueError('최소 Feature 수가 입력보다 큽니다.')
        self.feature_names_in_=np.asarray(raw.columns,dtype=object);self.n_features_in_=raw.shape[1];cols=raw.columns.tolist();rows=[]
        def fit(cols):
            """학습/분석 단계의 계산.
            
            Args:
            cols: 명세의 단계 입력.
            Returns:
                object: 계산 결과.
            Raises:
                ValueError/TypeError: 자료·설정·적합 오류.
            """
            return fit_ols(raw[cols],target,OLSProfile())[0]
        result=fit(cols);rows.append(dict(step=0,removed=None,aic=float(result.aic),bic=float(result.bic),accepted=True))
        for step in range(1,self.max_steps+1):
            if len(cols)<=self.min_features:break
            candidates=[]
            for col in cols:
                candidate=[c for c in cols if c!=col];model=fit(candidate);score=float(getattr(model,self.criterion));candidates.append((score,col,model,candidate))
                rows.append(dict(step=step,removed=col,aic=float(model.aic),bic=float(model.bic),accepted=False))
            score,col,model,candidate=min(candidates,key=lambda a:a[0])
            if score>=float(getattr(result,self.criterion))-self.improvement:break
            cols=candidate;result=model
            next(r for r in rows if r['step']==step and r['removed']==col)['accepted']=True
        self.selected_features_=cols;self.model_=result;self.history_=pd.DataFrame(rows);return self
    def transform(self,X):
        """학습/분석 단계의 계산.
        
        Args:
        self: 명세의 단계 입력.
        X: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        check_is_fitted(self,'model_');raw,_=numeric_data(X)
        if raw.columns.tolist()!=self.feature_names_in_.tolist():raise ValueError('Feature 이름/순서 불일치')
        return raw[self.selected_features_].to_numpy()
    def get_feature_names_out(self,input_features=None):
        """학습/분석 단계의 계산.
        
        Args:
        self: 명세의 단계 입력.
        input_features: 명세의 단계 입력.
        Returns:
            object: 계산 결과.
        Raises:
            ValueError/TypeError: 자료·설정·적합 오류.
        """
        check_is_fitted(self,'model_');return np.asarray(self.selected_features_,dtype=object)

def _run(X, y, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        InformationSelectionOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    selector=InformationSelector(**asdict(profile)).fit(X,y)
    return InformationSelectionOutputDTO(selector,selector.model_,selector.selected_features_,selector.history_,asdict(profile),['동일 행·동일 반응 척도의 OLS 내에서만 AIC/BIC를 비교합니다. 변환된 y 모형·다른 행의 정보량과 직접 비교하지 않습니다.','탐욕적 후진 선택이며 전체 부분집합 최적성·선택 후 p값 보정을 보장하지 않습니다.'])

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_information_selection(X, y, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        X: 전용 명세의 X 입력.
        y: 전용 명세의 y 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        InformationSelectionOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('regression-information-selection',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(X, y,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
