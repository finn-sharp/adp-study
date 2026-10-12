"""
================================================================================
precision_recall.py - 불균형 분류에서 재현율과 양성 예측 신뢰도를 비교합니다.
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
class ClassificationPrecisionRecallProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        positive_label (Any): 큰 score가 의미하는 양성 라벨. 기본 1.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8
    positive_label: Any = 1

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ClassificationPrecisionRecallOutputDTO(PlotOutputDTO):
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
        profile: ClassificationPrecisionRecallProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, ClassificationPrecisionRecallProfile)

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(y_true, scores, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        y_true: 대응 참조 분류 라벨; 이진 score 차트는 정확히 두 클래스.
        scores: 큰 값이 positive_label인 유한 점수; 반드시 확률일 필요는 없음.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        ClassificationPrecisionRecallOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    from sklearn.metrics import precision_recall_curve,average_precision_score
    y,s,removed=binary_scores(y_true,scores,profile.positive_label);precision,recall,thresholds=precision_recall_curve(y,s);ap=float(average_precision_score(y,s));ax.plot(recall,precision,color=profile.color,alpha=profile.alpha,label='AP='+format(ap,'.3f'));ax.axhline(y.mean(),color='gray',linestyle='--',label='Prevalence');ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.legend(fontsize=8)
    data=pd.DataFrame({'precision':precision,'recall':recall,'threshold':np.r_[thresholds,np.nan]});diagnostics={'average_precision':ap,'n':len(y),'positive_fraction':float(y.mean()),'missing_removed':removed};notes=['AP는 사다리꼴 PR 면적과 다릅니다. 불균형 표본의 양성 비율을 함께 확인하세요.']
    return finish(ax, before, data, diagnostics, profile, ClassificationPrecisionRecallOutputDTO, 'Classification — Precision Recall', notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_precision_recall(y_true, scores, *, ax, profile=None, recipe_path=None, overrides=None) -> ClassificationPrecisionRecallOutputDTO:
    """불균형 분류에서 재현율과 양성 예측 신뢰도를 비교합니다.

    Args:
        y_true: 대응 참조 분류 라벨; 이진 score 차트는 정확히 두 클래스.
        scores: 큰 값이 positive_label인 유한 점수; 반드시 확률일 필요는 없음.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        ClassificationPrecisionRecallOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile=resolve_profile('plot-classification-precision-recall', profile, recipe_path=recipe_path, overrides=overrides)
    _validate(profile)
    before=start(ax)
    return _draw(y_true, scores, ax, profile, before)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.classification에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_precision_recall(..., ax=axes[0, 0]); result.axes is axes[0, 0]
