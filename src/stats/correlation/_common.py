"""상관분석의 대응 결측·행렬·다중비교 보정 공통 계약."""
import numpy as np
import pandas as pd
from src.MachineLearning.analysis_common import finite_number,positive_int


def validate_profile(profile):
    """공통 상관분석 설정을 확인합니다.

    Args:
        profile: 전용 Profile.
    Returns:
        None: 유효한 설정.
    Raises:
        ValueError/TypeError: 설정 범위 오류.
    """
    finite_number(profile.alpha,'alpha')
    if not 0<profile.alpha<1 or profile.nan_policy not in ('raise','omit'):raise ValueError('alpha/nan_policy 오류')
    if hasattr(profile,'alternative') and profile.alternative not in ('two-sided','less','greater'):raise ValueError('alternative 오류')
    if hasattr(profile,'p_adjust') and profile.p_adjust not in ('none','bonferroni','holm','fdr_bh'):raise ValueError('p_adjust 오류')
    if hasattr(profile,'min_samples'):positive_int(profile.min_samples,'min_samples',3)
    if hasattr(profile,'max_pairs'):positive_int(profile.max_pairs,'max_pairs')


def numeric_pair(x,y,profile):
    """동일 위치의 쌍을 함께 제거하고 상수/무한대를 거부합니다.

    Args:
        x: 1차원 수치.
        y: 같은 길이의 1차원 수치.
        profile: 결측·표본수 설정.
    Returns:
        tuple: 유효 x,y 및 제거한 쌍 수.
    Raises:
        ValueError/TypeError: 길이·인덱스·상수·자료 오류.
    """
    if isinstance(x,pd.Series) and isinstance(y,pd.Series) and not x.index.equals(y.index):raise ValueError('Series의 대응 인덱스가 다릅니다.')
    if np.iscomplexobj(x) or np.iscomplexobj(y):raise ValueError('실수 자료가 필요합니다.')
    a=np.asarray(x,dtype=float);b=np.asarray(y,dtype=float)
    if a.ndim!=1 or b.shape!=a.shape or np.isinf(a).any() or np.isinf(b).any():raise ValueError('대응 수치 벡터/무한대 오류')
    mask=~(np.isnan(a)|np.isnan(b));removed=int((~mask).sum())
    if removed and profile.nan_policy=='raise':raise ValueError('대응 자료에 결측이 있습니다.')
    a=a[mask];b=b[mask]
    if len(a)<profile.min_samples or np.ptp(a)==0 or np.ptp(b)==0:raise ValueError('유효 표본 수 부족 또는 상수 자료')
    return a,b,removed


def adjusted_pvalues(values,method):
    """중복 없는 상삼각 검정들만 다중비교 보정합니다.

    Args:
        values: 유한한 p 값 벡터.
        method: none/bonferroni/holm/fdr_bh.
    Returns:
        np.ndarray: 원 순서의 보정 p 값.
    Raises:
        ValueError: 방법 오류.
    """
    p=np.asarray(values,dtype=float);n=len(p)
    if method=='none' or n==0:return p.copy()
    if method=='bonferroni':return np.minimum(1,p*n)
    order=np.argsort(p,kind='stable');sorted_p=p[order]
    if method=='holm':q=np.maximum.accumulate(sorted_p*(n-np.arange(n)))
    elif method=='fdr_bh':q=np.minimum.accumulate((sorted_p*n/(1+np.arange(n)))[::-1])[::-1]
    else:raise ValueError('p_adjust 오류')
    result=np.empty(n);result[order]=np.clip(q,0,1);return result


def correlation_matrix(X,profile,pair_runner,output_class,method):
    """각 쌍의 유효 표본 수와 보정 p 값을 함께 반환합니다.

    Args:
        X: 2개 이상 열의 수치 행렬/DataFrame.
        profile: 쌍별 결측/검정 설정.
        pair_runner: 동일 Profile을 사용하는 전용 pair 실행 함수.
        output_class: 전용 행렬 DTO.
        method: 기법 이름.
    Returns:
        object: 계수/p/보정 p/유효 표본 수/유의성 DataFrame.
    Raises:
        ValueError/TypeError: shape·열·무한대·검정 한도 오류.
    """
    if np.iscomplexobj(X):raise ValueError('실수 행렬이 필요합니다.')
    a=np.asarray(X,dtype=float)
    if a.ndim!=2 or a.shape[1]<2 or len(a)<profile.min_samples or np.isinf(a).any():raise ValueError('수치 행렬 shape/무한대 오류')
    if profile.nan_policy=='raise' and np.isnan(a).any():raise ValueError('행렬 결측 오류')
    names=[str(c) for c in X.columns] if isinstance(X,pd.DataFrame) else ['feature_'+str(i) for i in range(a.shape[1])]
    if len(set(names))!=len(names):raise ValueError('중복 열 이름')
    d=a.shape[1]
    if d*(d-1)//2>profile.max_pairs:raise ValueError('상관 쌍 한도 초과')
    coefficients=np.full((d,d),np.nan);pvalues=coefficients.copy();counts=np.zeros((d,d),dtype=int);warnings=[];valid=[]
    for i in range(d):
        column=a[:,i];column=column[~np.isnan(column)];counts[i,i]=len(column)
        if len(column)>=profile.min_samples and np.ptp(column)>0:coefficients[i,i]=1.
        for j in range(i+1,d):
            count=int((~(np.isnan(a[:,i])|np.isnan(a[:,j]))).sum());counts[i,j]=counts[j,i]=count
            try:result=pair_runner(a[:,i],a[:,j],profile=profile)
            except ValueError as error:
                warnings.append(names[i]+' / '+names[j]+': '+str(error));continue
            coefficients[i,j]=coefficients[j,i]=result.coefficient;pvalues[i,j]=pvalues[j,i]=result.p_value;valid.append((i,j))
            warnings.extend(result.warnings)
    adjusted=np.full((d,d),np.nan)
    for (i,j),value in zip(valid,adjusted_pvalues([pvalues[i,j] for i,j in valid],profile.p_adjust)):adjusted[i,j]=adjusted[j,i]=value
    significance=np.isfinite(adjusted)&(adjusted<profile.alpha)
    if profile.nan_policy=='omit' and np.isnan(a).any():warnings.append('쌍별 결측 제거로 표본 수가 다르며 계수 행렬의 PSD를 보장하지 않습니다.')
    frames=[pd.DataFrame(v,index=names,columns=names) for v in (coefficients,pvalues,adjusted,counts,significance)]
    return output_class(method,*frames,names,profile.alpha,list(dict.fromkeys(warnings)))


def categorical_pair(x,y,profile):
    """범주를 순서 숫자로 바꾸지 않고 동일 행의 결측을 함께 처리합니다.

    Args:
        x: 1차원 범주 자료.
        y: 대응 범주 자료.
        profile: 결측·유의수준 설정.
    Returns:
        tuple: 원 범주 x,y 및 제거한 쌍 수.
    Raises:
        ValueError/TypeError: 자료·결측·길이/인덱스 오류.
    """
    if isinstance(x,pd.Series) and isinstance(y,pd.Series) and not x.index.equals(y.index):raise ValueError('대응 인덱스 불일치')
    a=np.asarray(x,dtype=object);b=np.asarray(y,dtype=object)
    if a.ndim!=1 or b.shape!=a.shape:raise ValueError('대응 범주 벡터가 필요합니다.')
    valid=~(pd.isna(a)|pd.isna(b));removed=int((~valid).sum())
    if removed and profile.nan_policy=='raise':raise ValueError('대응 범주 결측')
    a=a[valid];b=b[valid]
    if len(a)<3:raise ValueError('3개 이상 유효 표본이 필요합니다.')
    for values in (a,b):
        for value in values:
            try:hash(value)
            except TypeError as error:raise ValueError('범주는 hashable scalar여야 합니다.') from error
            if isinstance(value,(float,np.floating)) and not np.isfinite(value):raise ValueError('비유한 범주')
    return a,b,removed
