"""
================================================================================
comparison.py - 동일 정답·행 순서의 여러 모델 평가표
================================================================================
설명:
    - 한 정답 배열에 모델별 예측/확률을 연결하여 같은 기준으로 평가합니다.
분석 가정:
    - 입력 예측은 동일 평가 자료의 동일 행 순서이며 최종 test로 모델 선택하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-13
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-13  김재현(Finn)      v1.0.0    독립 모델 비교 메트릭
================================================================================
"""
__author__ = '김재현(Finn)'
__email__ = 'penguin.klg@gmail.com'
__version__ = '1.0.0'
__status__ = 'Development'

# 1. Imports
from dataclasses import dataclass
from typing import Any
import pandas as pd
from src.output import OutputDTO
from src.Recipes.loader import resolve_profile
from src.Evaluation.adapters import evaluate_predictions
from src.Evaluation._common import LOWER_IS_BETTER

# 2. Input Profile
@dataclass
class ComparisonProfile:
    """동일 평가표 설정.

    Attributes:
        evaluation_task (str): classification/regression.
    """
    evaluation_task: str = 'classification'

# 3. Output DTO
@dataclass
class ComparisonOutputDTO(OutputDTO):
    """모델별 평가표.

    Attributes:
        table (DataFrame): 행=모델, 열=스칼라 지표; 미정의 NaN.
        directions (dict): 지표별 higher/lower 방향.
        n_samples (int): 동일 정답 행 수.
        parameters (dict): 실제 설정.
        warnings (list): 행 정렬 책임·통계적 유의성 제한.
    """
    table: Any
    directions: dict
    n_samples: int
    parameters: dict
    warnings: list

# 4. Input Validation
def validate_inputs(predictions, scores, classes, p):
    """모델명 대응과 목적을 검증합니다.

    Args:
        predictions: 모델명 → 예측 사전.
        scores: 모델명 → 확률 또는 None.
        classes: 모델명 → 확률 열 클래스 순서 또는 None.
        p: 전용 Profile.
    Returns:
        None: 정상 입력.
    Raises:
        ValueError/TypeError: 잘못된 사전/목적/대응.
    """
    if not isinstance(p,ComparisonProfile) or p.evaluation_task not in ('classification','regression'):
        raise ValueError('전용 Profile과 classification/regression 목적 필요')
    if not isinstance(predictions,dict) or not predictions or not all(isinstance(k,str) and k for k in predictions):
        raise ValueError('비어 있지 않은 문자열 모델명→예측 사전 필요')
    for mapping in (scores,classes):
        if mapping is not None and (not isinstance(mapping,dict) or set(mapping)-set(predictions)):
            raise ValueError('점수/클래스의 모델명이 예측 사전에 있어야 합니다.')
    if p.evaluation_task=='regression' and (scores or classes): raise ValueError('회귀에 확률/클래스 입력 불가')

# 5. Individual Model Runners
def _compare(y_true,predictions,scores,classes,p):
    """같은 정답으로 모든 모델을 독립 평가합니다.

    Args:
        y_true: 동일 평가 정답.
        predictions: 모델별 대응 예측.
        scores: 모델별 확률.
        classes: 모델별 확률 열 순서.
        p: 비교 설정.
    Returns:
        ComparisonOutputDTO: 평가표와 방향.
    Raises:
        ValueError: 입력 shape·정렬·확률 오류.
    """
    rows={k:evaluate_predictions(y_true,v,p.evaluation_task,(scores or {}).get(k),(classes or {}).get(k)) for k,v in predictions.items()}
    table=pd.DataFrame.from_dict(rows,orient='index')
    return ComparisonOutputDTO(table,{k:'lower' if k in LOWER_IS_BETTER else 'higher' for k in table.columns},len(y_true),{'evaluation_task':p.evaluation_task},['예측의 동일 행 순서는 호출자가 보장합니다.','같은 표의 점수 차이는 통계적 유의성 또는 최종 test 모델 선택 근거가 아닙니다.'])

# 6. Public Runner
def run_comparison(y_true,predictions,profile=None,*,scores=None,classes=None,recipe_path=None,overrides=None):
    """모델 비교를 실행합니다.

    Args:
        y_true: 같은 평가 자료 정답.
        predictions: 모델명 → 같은 행 순서 예측.
        profile: ComparisonProfile 또는 None.
        scores: 모델명 → 확률; 없는 모델은 라벨 메트릭만 계산.
        classes: 모델명 → 확률 열 순서.
        recipe_path: 사용자 JSON.
        overrides: JSON 재정의.
    Returns:
        ComparisonOutputDTO: 모델별 평가표.
    Raises:
        ValueError/TypeError/OSError: 설정/입력/레시피 오류.
    """
    p=resolve_profile('metric-comparison',profile,recipe_path,overrides=overrides)
    validate_inputs(predictions,scores,classes,p)
    return _compare(y_true,predictions,scores,classes,p)

# 7. Example Execution
if __name__=='__main__':
    print(run_comparison([0,0,1,1],{'baseline':[0,0,0,0],'model':[0,1,1,1]}).to_json(indent=2))
