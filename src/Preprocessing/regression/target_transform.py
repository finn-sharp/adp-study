"""
================================================================================
target_transform.py - 회귀 반응변수의 train-only 변환·역변환
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
class TargetTransformProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        method (str): none/log/log1p/yeo-johnson/box-cox 반응변수 변환. 기본 'none'.
    """
    method: str = 'none'

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class TargetTransformOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        transformed_target (np.ndarray): 학습 타깃의 모델 좌표계.
        new_transformed_target (Optional[np.ndarray]): 새 타깃 transform 또는 None.
        transformer (Any): 학습 전용 변환기; JSON 제외.
        parameters (Dict[str,Any]): 적용 설정.
        warnings (List[str]): inverse 예측 평균과 중앙값·추론 좌표계 구별.
    """
    transformed_target: np.ndarray
    new_transformed_target: Optional[np.ndarray]
    transformer: Any
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
    validate_profile(profile,TargetTransformProfile)
    if profile.method not in ('none','log','log1p','yeo-johnson','box-cox'):raise ValueError('타깃 변환 오류')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def build_target_transformer(profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
        recipe_path: 전용 명세의 recipe_path 입력.
        overrides: 전용 명세의 overrides 입력.
    Returns:
        RegressionTargetTransformer: 미적합 타깃 변환기.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('regression-target-transform',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return RegressionTargetTransformer(profile)

class RegressionTargetTransformer(TransformerMixin,BaseEstimator):
    """1차원 y의 학습 전용 변환; 행을 제거하지 않습니다."""
    def __init__(self,profile=None):
        """설정 보존.

        Args:
            profile: TargetTransformProfile 또는 None.
        Returns:
            None: 미적합 설정.
        Raises:
            None: fit에서 검증.
        """
        self.profile=profile

    def _array(self,y):
        """타깃 변환 정의역을 확인합니다.

        Args:
            y: 유한 1차원 타깃.
        Returns:
            ndarray: float 벡터.
        Raises:
            ValueError: shape·결측·log 정의역 오류.
        """
        if np.iscomplexobj(y):raise ValueError('실수 타깃 오류')
        a=np.asarray(y,dtype=float);profile=self.profile or TargetTransformProfile();_validate(profile)
        if a.ndim!=1 or len(a)<1 or not np.isfinite(a).all():raise ValueError('유한 1차원 타깃이 필요합니다.')
        if profile.method in ('log','box-cox') and (a<=0).any():raise ValueError('log/Box-Cox 타깃은 양수입니다.')
        if profile.method=='log1p' and (a<=-1).any():raise ValueError('log1p 타깃은 -1 초과입니다.')
        return a

    def fit(self,X,y=None):
        """학습 타깃에서만 lambda 등을 적합합니다.

        Args:
            X: 1차원 학습 타깃.
            y: 미사용 sklearn 호환 입력.
        Returns:
            RegressionTargetTransformer: 자신.
        Raises:
            ValueError: 정의역·상수 오류.
        """
        a=self._array(X);self.method_=(self.profile or TargetTransformProfile()).method
        if self.method_ in ('yeo-johnson','box-cox'):
            if np.ptp(a)==0:raise ValueError('상수 타깃 power 변환 오류')
            from sklearn.preprocessing import PowerTransformer
            self.transformer_=PowerTransformer(method=self.method_,standardize=False).fit(a[:,None])
        self.fitted_=True
        return self

    def transform(self,X):
        """타깃을 모델 좌표계로 변환합니다.

        Args:
            X: 새/학습 1차원 타깃.
        Returns:
            ndarray: 같은 길이의 변환 타깃.
        Raises:
            ValueError/NotFittedError: 미적합·정의역 오류.
        """
        check_is_fitted(self,'fitted_');a=self._array(X)
        if self.method_=='log':return np.log(a)
        if self.method_=='log1p':return np.log1p(a)
        if self.method_ in ('yeo-johnson','box-cox'):return self.transformer_.transform(a[:,None]).ravel()
        return a.copy()

    def inverse_transform(self,X):
        """모델 좌표계 예측을 원 좌표계로 복원합니다.

        Args:
            X: 유한 변환 좌표계 벡터.
        Returns:
            ndarray: 원 타깃 단위; 평균 불편성은 자동 보장하지 않음.
        Raises:
            ValueError/NotFittedError: 미적합·역변환 정의역/overflow 오류.
        """
        check_is_fitted(self,'fitted_');a=np.asarray(X,dtype=float)
        if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('역변환 벡터 오류')
        if self.method_=='log':result=np.exp(a)
        elif self.method_=='log1p':result=np.expm1(a)
        elif self.method_ in ('yeo-johnson','box-cox'):result=self.transformer_.inverse_transform(a[:,None]).ravel()
        else:result=a.copy()
        if not np.isfinite(result).all():raise ValueError('역변환 정의역/overflow 오류')
        return result

def _run(y_train, y_new, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        y_train: 전용 명세의 y_train 입력.
        y_new: 전용 명세의 y_new 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        TargetTransformOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    transformer=build_target_transformer(profile).fit(y_train)
    return TargetTransformOutputDTO(transformer.transform(y_train),None if y_new is None else transformer.transform(y_new),transformer,asdict(profile),['반응변수 변환은 정규 y를 만들기 위한 필수 조건이 아닙니다. 로그 inverse는 일반적으로 조건부 평균을 직접 추정하지 않습니다. 계수/검정은 변환 y 좌표계입니다.'])

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_target_transform(y_train, y_new=None, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        y_train: 전용 명세의 y_train 입력.
        y_new: 전용 명세의 y_new 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        TargetTransformOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('regression-target-transform',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(y_train, y_new,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
