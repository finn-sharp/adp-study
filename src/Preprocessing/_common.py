"""전처리의 학습 전용 적합·열 스키마·배열/범주 정의역 공통 계약."""
from dataclasses import asdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.utils.validation import check_is_fitted


def feature_array(X,numeric,allow_nan):
    """자료를 2차원으로 검증하고 범주의 결측 표현을 통일합니다.

    Args:
        X: 열 이름이 고유한 DataFrame 또는 2차원 배열.
        numeric: 수치형 여부.
        allow_nan: 결측 허용 여부.
    Returns:
        tuple: 배열과 원 열 이름.
    Raises:
        ValueError/TypeError: shape·열·타입·결측·무한대 오류.
    """
    if np.iscomplexobj(X):raise ValueError('복소수 입력은 지원하지 않습니다.')
    if isinstance(X,pd.DataFrame):
        names=[str(c) for c in X.columns]
        if len(set(names))!=len(names):raise ValueError('열 이름이 중복됩니다.')
    else:names=None
    if numeric and isinstance(X,pd.DataFrame):
        a=X.to_numpy(dtype=float,na_value=np.nan)
    elif numeric:
        raw=np.asarray(X,dtype=object);raw=raw.copy();raw[pd.isna(raw)]=np.nan;a=np.asarray(raw,dtype=float)
    else:a=np.asarray(X,dtype=object)
    if a.ndim!=2 or min(a.shape)<1:raise ValueError('비어 있지 않은 2차원 자료가 필요합니다.')
    names=names or ['feature_'+str(i) for i in range(a.shape[1])]
    if numeric:
        if np.isinf(a).any() or (not allow_nan and np.isnan(a).any()):raise ValueError('무한대/결측 오류')
    else:
        a=a.copy();a[pd.isna(a)]=np.nan
        if not allow_nan and pd.isna(a).any():raise ValueError('범주 결측은 먼저 대치해야 합니다.')
        for column in a.T:
            valid=column[~pd.isna(column)]
            if any(isinstance(v,(list,dict,set,tuple)) for v in valid):raise ValueError('범주는 scalar 값이어야 합니다.')
            if any(isinstance(v,(float,np.floating)) and not np.isfinite(v) for v in valid):raise ValueError('범주의 비유한 수치 오류')
            # sklearn versions differ on mixed numbers/strings. Normalize explicitly, without collisions.
            mapping={}
            for value in valid:
                key=str(value)
                if key in mapping and (type(value)!=type(mapping[key]) or value!=mapping[key]):raise ValueError('문자열 변환으로 서로 다른 범주가 충돌합니다.')
                mapping[key]=value
        mask=~pd.isna(a);a[mask]=[str(v) for v in a[mask]]
    return a,names


class ProfileTransformer(TransformerMixin,BaseEstimator):
    """FeatureSpec/Pipeline에 재사용 가능한 clone 호환 변환기.

    Attributes:
        estimator (object): 미적합 sklearn 추정기.
        numeric (bool): 수치형 여부.
        allow_nan (bool): 결측 허용 여부.
        reject_empty (bool): 학습 열 전체 결측을 거부할지 여부.
        estimator_ (object): 학습 입력에만 적합한 내부 추정기.
        feature_names_in_ (np.ndarray): 학습 열 이름과 순서.
    """
    def __init__(self,estimator,numeric=True,allow_nan=False,reject_empty=False,reject_constant=False,require_positive=False):
        """미적합 설정을 보존합니다.

        Args:
            estimator: sklearn 추정기.
            numeric: 수치형 여부.
            allow_nan: 결측 허용 여부.
            reject_empty: 전체 결측 열 거부.
            reject_constant: 상수 학습 열 거부.
            require_positive: 양수 자료만 허용.
        Returns:
            None: 변환기를 생성합니다.
        Raises:
            None: 실제 검증은 fit/transform에서 수행.
        """
        self.estimator=estimator;self.numeric=numeric;self.allow_nan=allow_nan;self.reject_empty=reject_empty
        self.reject_constant=reject_constant;self.require_positive=require_positive

    def fit(self,X,y=None):
        """제공한 학습 구간에만 내부 변환기를 적합합니다.

        Args:
            X: 학습 자료.
            y: sklearn 호환 선택 타깃.
        Returns:
            ProfileTransformer: 적합한 자신.
        Raises:
            ValueError/TypeError: 자료·열 정의역 오류.
        """
        a,names=feature_array(X,self.numeric,self.allow_nan)
        if self.reject_empty and pd.isna(a).all(axis=0).any():raise ValueError('학습 열 전체 결측은 허용하지 않습니다.')
        if self.reject_constant and (np.ptp(a,axis=0)==0).any():raise ValueError('상수 학습 열을 허용하지 않습니다.')
        if self.require_positive and (a<=0).any():raise ValueError('양수 자료만 지원합니다.')
        self.feature_names_in_=np.asarray(names,dtype=object);self.n_features_in_=a.shape[1]
        self.estimator_=clone(self.estimator).fit(a,y)
        return self

    def transform(self,X):
        """학습 상태를 갱신하지 않고 새 입력을 변환합니다.

        Args:
            X: 같은 열 이름/순서의 새 자료.
        Returns:
            object: 수치 배열 또는 sparse 행렬.
        Raises:
            ValueError/NotFittedError: 열·자료·미적합 오류.
        """
        check_is_fitted(self,'estimator_');a,names=feature_array(X,self.numeric,self.allow_nan)
        if a.shape[1]!=self.n_features_in_ or (isinstance(X,pd.DataFrame) and names!=self.feature_names_in_.tolist()):raise ValueError('학습 열 이름/순서와 다릅니다.')
        if self.require_positive and (a<=0).any():raise ValueError('양수 자료만 지원합니다.')
        return self.estimator_.transform(a)

    def get_feature_names_out(self,input_features=None):
        """출력 열 순서를 조회합니다.

        Args:
            input_features: sklearn ColumnTransformer가 전달한 원 열 이름 또는 None.
        Returns:
            np.ndarray: 변환한 열 이름.
        Raises:
            ValueError/NotFittedError: 미적합 또는 이름 불일치.
        """
        check_is_fitted(self,'estimator_');names=self.feature_names_in_ if input_features is None else np.asarray(input_features,dtype=object)
        if len(names)!=self.n_features_in_:raise ValueError('입력 열 수 불일치')
        if hasattr(self.estimator_,'get_feature_names_out'):return self.estimator_.get_feature_names_out(names)
        if hasattr(self.estimator_,'get_feature_names'):return self.estimator_.get_feature_names(names)
        return names.copy()

    def inverse_transform(self,X):
        """내부 변환기가 지원하는 경우 역변환합니다.

        Args:
            X: 변환한 수치 자료.
        Returns:
            object: 복원한 자료.
        Raises:
            ValueError/NotFittedError: 역변환 미지원 또는 미적합.
        """
        check_is_fitted(self,'estimator_')
        if not hasattr(self.estimator_,'inverse_transform'):raise ValueError('이 변환은 역변환을 지원하지 않습니다.')
        return self.estimator_.inverse_transform(X)


def execute_preprocessing(X_train,X_new,profile,transformer,output_class):
    """학습만 적합하고 원/새 전처리 결과 및 열 계약을 구성합니다.

    Args:
        X_train: 학습 자료.
        X_new: 선택 새 자료; 재적합하지 않음.
        profile: 전용 설정.
        transformer: 미적합 변환기.
        output_class: 전용 Output DTO.
    Returns:
        object: 전처리 결과 DTO.
    Raises:
        ValueError/TypeError: 학습·변환·열 오류.
    """
    trained=transformer.fit(X_train)
    return output_class(trained.transform(X_train),None if X_new is None else trained.transform(X_new),
                        trained.feature_names_in_.tolist(),trained.get_feature_names_out().tolist(),
                        asdict(profile),trained,len(X_train),[])
