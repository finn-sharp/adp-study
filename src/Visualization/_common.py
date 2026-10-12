"""단일 Axes 소유권·대응 행·유한 입력·직렬화 가능한 시각화 결과의 공통 계약."""
from dataclasses import fields, asdict
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from src.output import OutputDTO


class PlotOutputDTO(OutputDTO):
    """Axes/figure/artists는 runtime이고 data/diagnostics는 JSON 저장용입니다."""
    runtime_fields = OutputDTO.runtime_fields | frozenset(('artists',))


def validate_profile(profile, cls):
    """Profile 기본 타입·스타일을 검증합니다.

    Args:
        profile: 전용 dataclass.
        cls: 요구한 전용 Profile 클래스.
    Returns:
        None: 유효한 설정.
    Raises:
        TypeError/ValueError: 필드 타입·스타일 오류.
    """
    if not isinstance(profile, cls):
        raise TypeError('전용 Profile이 필요합니다.')
    for field in fields(profile):
        value = getattr(profile, field.name)
        if field.type is bool and not isinstance(value, bool):
            raise TypeError(field.name+'는 bool입니다.')
        if field.type is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError(field.name+'는 int입니다.')
        if field.type is str and not isinstance(value, str):
            raise TypeError(field.name+'는 str입니다.')
        if field.type is float and (isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value)):
            raise ValueError(field.name+'는 유한한 실수입니다.')
    if not is_color_like(profile.color) or not 0 <= profile.alpha <= 1:
        raise ValueError('color/alpha 오류')


def start(ax, cax=None):
    """그림을 생성하거나 다른 축을 고르지 않고 전달받은 축을 확인합니다.

    Args:
        ax: 기존 2D Matplotlib Axes.
        cax: 같은 figure의 선택 색상 막대 Axes.
    Returns:
        set: 그리기 전 artist 식별자.
    Raises:
        TypeError/ValueError: 축 타입·figure·축 중복 오류.
    """
    if not isinstance(ax, Axes) or ax.name != 'rectilinear':
        raise TypeError('기존 2D ax를 전달하세요.')
    if cax is not None and (not isinstance(cax, Axes) or cax is ax or cax.figure is not ax.figure):
        raise ValueError('cax는 같은 figure의 다른 Axes여야 합니다.')
    return {id(artist) for artist in ax.get_children()}


def finish(ax, before, data, diagnostics, profile, cls, title, warnings=None):
    """같은 축과 수치 근거를 전용 DTO로 묶습니다.

    Args:
        ax: 그린 원 축.
        before: 기존 artist 식별자.
        data: 표시한 값의 DataFrame.
        diagnostics: 판단 수치와 정의.
        profile: 적용 설정.
        cls: 전용 결과 클래스.
        title: 기본 제목.
        warnings: 해석 경고 목록 또는 None.
    Returns:
        PlotOutputDTO: figure/axes/artists와 저장 가능한 자료.
    Raises:
        TypeError: 잘못된 DTO 필드.
    """
    ax.set_title(profile.title or title)
    artists = [artist for artist in ax.get_children() if id(artist) not in before]
    return cls(ax.figure, ax, artists, data, diagnostics, asdict(profile), warnings or [])


def vector(values, name='value', min_size=1, allow_missing=True):
    """유한 수치 벡터를 검증하고 단변량 결측만 제거합니다.

    Args:
        values: 수치 벡터/Series.
        name: 오류 필드 이름.
        min_size: 결측 제거 후 최소 크기.
        allow_missing: NaN 제거 허용.
    Returns:
        tuple: 유효 배열, 제거 수.
    Raises:
        ValueError/TypeError: shape·복소·무한대·표본 수 오류.
    """
    if np.iscomplexobj(values):raise ValueError(name+'는 실수입니다.')
    a=values.to_numpy(dtype=float,na_value=np.nan) if isinstance(values,pd.Series) else np.asarray(values,dtype=float)
    if a.ndim!=1 or np.isinf(a).any():raise ValueError(name+' 벡터/무한대 오류')
    mask=np.isnan(a)
    if mask.any() and not allow_missing:raise ValueError(name+' 결측 오류')
    if (~mask).sum()<min_size:raise ValueError(name+' 유효 표본 수 부족')
    return a[~mask],int(mask.sum())


def frame(values, allow_missing=False):
    """행/Feature 순서를 보존한 수치 DataFrame을 검증합니다.

    Args:
        values: 2차원 DataFrame/배열.
        allow_missing: NaN 허용 여부.
    Returns:
        DataFrame: 유한 수치 자료와 원 이름.
    Raises:
        ValueError/TypeError: shape·열·타입·무한대 오류.
    """
    if np.iscomplexobj(values):raise ValueError('복소수 자료 오류')
    result=values.copy() if isinstance(values,pd.DataFrame) else pd.DataFrame(values)
    if result.empty or not result.columns.is_unique:raise ValueError('비어 있거나 중복 열입니다.')
    result.columns=[str(c) for c in result.columns]
    if not result.columns.is_unique:raise ValueError('문자열 열 이름 충돌')
    a=result.to_numpy(dtype=float,na_value=np.nan)
    if np.isinf(a).any() or (not allow_missing and np.isnan(a).any()):raise ValueError('수치 자료 결측/무한대 오류')
    return pd.DataFrame(a,index=result.index,columns=result.columns)


def paired(**values):
    """수치·범주·시간의 대응 행을 유지한 채 결측 쌍/행을 제거합니다.

    Args:
        values: 같은 길이 벡터의 이름별 입력; Series끼리는 인덱스 일치 필요.
    Returns:
        tuple: 유효 행 DataFrame, 제거 행 수.
    Raises:
        ValueError/TypeError: 길이·대응 인덱스·빈 자료 오류.
    """
    indices=[v.index for v in values.values() if isinstance(v,pd.Series)]
    if indices and any(not index.equals(indices[0]) for index in indices):raise ValueError('대응 Series 인덱스 불일치')
    if any(np.ndim(value)!=1 for value in values.values()):raise ValueError('1차원 대응 벡터가 필요합니다.')
    arrays={key:pd.Series(value).to_numpy() for key,value in values.items()}
    if any(value.ndim!=1 for value in arrays.values()) or len({len(v) for v in arrays.values()})!=1:raise ValueError('대응 벡터 길이 오류')
    result=pd.DataFrame(arrays);valid=result.notna().all(axis=1)
    if not valid.any():raise ValueError('유효 대응 행이 없습니다.')
    return result.loc[valid].reset_index(drop=True),int((~valid).sum())


def levels(values):
    """원 범주를 문자열 표시로 정규화하되 서로 다른 범주의 충돌을 거부합니다.

    Args:
        values: 결측 제거한 범주 Series.
    Returns:
        Series: 표시 가능한 문자열 범주.
    Raises:
        ValueError/TypeError: 비스칼라·비유한·문자열 충돌 오류.
    """
    mapping={}
    for value in values:
        if isinstance(value,(list,dict,set,tuple)) or (isinstance(value,(float,np.floating)) and not np.isfinite(value)):raise ValueError('유한 scalar 범주가 필요합니다.')
        key=str(value)
        if key in mapping and (type(value)!=type(mapping[key]) or value!=mapping[key]):raise ValueError('범주 문자열 충돌')
        mapping[key]=value
    return values.astype(str)


def heatmap(ax, values, row_names, col_names, cmap='viridis', cax=None, vmin=None, vmax=None, annotate=False):
    """지정 축에 행렬을 그리며 명시한 cax만 사용합니다.

    Args:
        ax: 대상 Axes.
        values: 2차원 수치/NaN 행렬.
        row_names: 행 이름.
        col_names: 열 이름.
        cmap: Matplotlib colormap.
        cax: 선택 색상 막대 축; None이면 추가하지 않음.
        vmin: 선택 최소 색 범위.
        vmax: 선택 최대 색 범위.
        annotate: 유한 값을 숫자로 표시.
    Returns:
        AxesImage: image artist; NaN은 마스킹.
    Raises:
        ValueError: colormap·축/행렬 오류.
    """
    image=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.set_xticks(np.arange(len(col_names)));ax.set_xticklabels(col_names,rotation=45,ha='right')
    ax.set_yticks(np.arange(len(row_names)));ax.set_yticklabels(row_names)
    if annotate:
        for i,j in np.ndindex(np.shape(values)):
            if np.isfinite(values[i,j]):
                rgba=image.cmap(image.norm(values[i,j]))
                luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,format(values[i,j],'.2g'),ha='center',va='center',fontsize=8,color='white' if luminance<.5 else 'black')
    if cax is not None:ax.figure.colorbar(image,cax=cax)
    return image


def binary_scores(y_true, scores, positive_label=1, probability=False):
    """대응 이진 타깃과 점수 정의역을 검증합니다.

    Args:
        y_true: 원 이진 라벨.
        scores: 큰 값이 positive인 점수.
        positive_label: 명시적인 양성 라벨.
        probability: 점수를 [0,1] 확률로 제한.
    Returns:
        tuple: 0/1 타깃, 점수, 제거 수.
    Raises:
        ValueError/TypeError: 클래스·점수·대응 오류.
    """
    data,removed=paired(target=y_true,score=scores)
    unique=pd.unique(data.target)
    if len(unique)!=2 or positive_label not in unique:raise ValueError('정확히 두 클래스와 관측 양성 라벨이 필요합니다.')
    a,_=vector(data.score,allow_missing=False)
    if probability and ((a<0)|(a>1)).any():raise ValueError('확률은 [0,1]입니다.')
    return (data.target==positive_label).to_numpy(dtype=int),a,removed
