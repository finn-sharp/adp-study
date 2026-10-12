"""
================================================================================
seasonal.py - 요일·월·시간별 반복 구조와 수집 불균형을 비교합니다.
================================================================================
설명:
    - Profile → JSON → 단일 ax 시각화 → 수치 근거·runtime DTO.
분석 가정:
    - 사용자가 전달한 2D Axes에만 그리며 figure 생성·show·save·예측 모델 재학습을 수행하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    Axes 조합 EDA·의사결정 시각화 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.Recipes import resolve_profile
from src.Visualization._common import PlotOutputDTO, validate_profile, start, finish, vector, frame, paired, levels, heatmap, binary_scores

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class TimeSeriesSeasonalProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        period (str): weekday/month/hour의 원 관측 분포. 기본 'weekday'.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8
    period: str = 'weekday'

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class TimeSeriesSeasonalOutputDTO(PlotOutputDTO):
    """그린 축과 의사결정 수치.

    Attributes:
        figure (Any): 원 ax의 Figure; runtime으로 JSON 제외.
        axes (Any): 전달한 동일 Axes; runtime으로 JSON 제외.
        artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
        data (pd.DataFrame): 표시/계산한 수치와 원 label 순서.
        diagnostics (Dict[str, Any]): 표본 수·분모·선택 기준·추가 근거.
        parameters (Dict[str, Any]): 적용 Profile 설정.
        warnings (List[str]): 해석 가정·미표시/표본 제한 정보.
    """
    figure: Any
    axes: Any
    artists: List[Any]
    data: pd.DataFrame
    diagnostics: Dict[str, Any]
    parameters: Dict[str, Any]
    warnings: List[str]

# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate(profile):
    """전용 Profile과 공통 스타일을 확인합니다.

    Args:
        profile: TimeSeriesSeasonalProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, TimeSeriesSeasonalProfile)

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(times, values, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        times: 원 대응 시간 벡터; timeline은 고유 오름차순 시간 필요.
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        TimeSeriesSeasonalOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    raw,removed=paired(time=times,value=values);raw['time']=pd.to_datetime(raw.time,errors='raise');raw['value']=vector(raw.value,allow_missing=False)[0]
    if raw.time.isna().any() or profile.period not in ('weekday','month','hour'):raise ValueError('계절 period/시간 오류')
    raw['period']=raw.time.dt.dayofweek if profile.period=='weekday' else raw.time.dt.month if profile.period=='month' else raw.time.dt.hour
    names=sorted(pd.unique(raw.period));groups=[raw.loc[raw.period==name,'value'].to_numpy() for name in names];ax.boxplot(groups,positions=np.arange(len(names)),patch_artist=True,boxprops={'facecolor':profile.color,'alpha':profile.alpha});ax.set_xticks(np.arange(len(names)));ax.set_xticklabels(names);ax.set_xlabel(profile.period+' (weekday: Mon=0)');ax.set_ylabel('Observed value');data=raw.groupby('period').value.agg(['count','mean','median','std']);diagnostics={'n':len(raw),'missing_removed':removed,'period':profile.period};notes=['집계 분포는 추세·표본 구성·노출량 차이를 보정하지 않습니다. 미관측 기간은 생성하지 않습니다.']
    return finish(ax, before, data, diagnostics, profile, TimeSeriesSeasonalOutputDTO, 'Time Series — Seasonal', notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_seasonal(times, values, *, ax, profile=None, recipe_path=None, overrides=None) -> TimeSeriesSeasonalOutputDTO:
    """요일·월·시간별 반복 구조와 수집 불균형을 비교합니다.

    Args:
        times: 원 대응 시간 벡터; timeline은 고유 오름차순 시간 필요.
        values: 원 수치 벡터; quality에서는 범주를 포함한 DataFrame, clustering에서는 명시 input_kind의 행렬.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        TimeSeriesSeasonalOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile=resolve_profile('plot-time-series-seasonal', profile, recipe_path=recipe_path, overrides=overrides)
    _validate(profile)
    before=start(ax)
    return _draw(times, values, ax, profile, before)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.time_series에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_seasonal(..., ax=axes[0, 0]); result.axes is axes[0, 0]
