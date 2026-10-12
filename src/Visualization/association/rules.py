"""
================================================================================
rules.py - 빈도·조건부 확률·독립 대비 lift를 함께 비교해 규칙 검토 우선순위를 정합니다.
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
class AssociationRulesProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        min_support (float): 표시할 최소 support [0,1]. 기본 0.0.
        min_confidence (float): 최소 confidence [0,1]. 기본 0.0.
        min_lift (float): 비음수 최소 lift. 기본 0.0.
        max_rules (int): lift 내림차순 표시 한도. 기본 200.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8
    min_support: float = 0.0
    min_confidence: float = 0.0
    min_lift: float = 0.0
    max_rules: int = 200

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class AssociationRulesOutputDTO(PlotOutputDTO):
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
        profile: AssociationRulesProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, AssociationRulesProfile)

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(rules, ax, profile, before, cax):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        rules: support/confidence/lift 열을 포함한 원 규칙 DataFrame.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
        cax: 선택 색상 막대 축.
    Returns:
        AssociationRulesOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    raw=rules.copy() if isinstance(rules,pd.DataFrame) else pd.DataFrame(rules)
    if raw.empty or not {'support','confidence','lift'}.issubset(raw.columns):raise ValueError('support/confidence/lift 열이 필요합니다.')
    for name in ('support','confidence','lift'):raw[name]=vector(raw[name],allow_missing=False)[0]
    if not 0<=profile.min_support<=1 or not 0<=profile.min_confidence<=1 or profile.min_lift<0 or profile.max_rules<1 or ((raw[['support','confidence']]<0)|(raw[['support','confidence']]>1)).any().any() or (raw.lift<0).any():raise ValueError('규칙/필터 정의역 오류')
    data=raw.loc[(raw.support>=profile.min_support)&(raw.confidence>=profile.min_confidence)&(raw.lift>=profile.min_lift)].sort_values('lift',ascending=False,kind='stable').reset_index(drop=True);shown=data.iloc[:profile.max_rules]
    artist=ax.scatter(shown.support,shown.confidence,c=shown.lift,s=25+50*shown.support,alpha=profile.alpha,cmap='viridis')
    if cax is not None and len(shown):ax.figure.colorbar(artist,cax=cax)
    ax.set_xlabel('Support');ax.set_ylabel('Confidence');ax.set_ylim(0,1.02);diagnostics={'input_rules':len(raw),'eligible_rules':len(data),'displayed_rules':len(shown),'color':'lift','size':'support'};notes=['높은 lift의 낮은 support 규칙은 희소성을 확인하세요. 규칙은 인과·수익성을 보장하지 않습니다.']
    return finish(ax, before, data, diagnostics, profile, AssociationRulesOutputDTO, 'Association — Rules', notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_rules(rules, *, ax, profile=None, recipe_path=None, overrides=None, cax=None) -> AssociationRulesOutputDTO:
    """빈도·조건부 확률·독립 대비 lift를 함께 비교해 규칙 검토 우선순위를 정합니다.

    Args:
        rules: support/confidence/lift 열을 포함한 원 규칙 DataFrame.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
        cax: 명시적 색상 막대 Axes; None이면 색상 막대 생략.
    Returns:
        AssociationRulesOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile=resolve_profile('plot-association-rules', profile, recipe_path=recipe_path, overrides=overrides)
    _validate(profile)
    before=start(ax, cax)
    return _draw(rules, ax, profile, before, cax)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.association에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_rules(..., ax=axes[0, 0]); result.axes is axes[0, 0]
