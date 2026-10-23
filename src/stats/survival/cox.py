"""
================================================================================
cox.py - Cox 비례위험 회귀·생존예측·PH 진단
================================================================================
설명:
    - 기법별 Profile·JSON 레시피·공개 실행기·전용 Output DTO를 제공합니다.
분석 가정:
    - 독립 대상·조건부 비정보성 우측 중도절단·로그위험 선형·비례위험. 인과 추론이 아닙니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-20
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-20  김재현(Finn)     v1.0.0    도메인 공통화·전용 명세·실행 예제 추가
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
from typing import Any, Optional, Dict, List
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from .kaplan_meier import run_kaplan_meier, KaplanMeierProfile
from .logrank import _holm

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class CoxProfile:
    """Attributes:
        duration_col (str): 시간 열 이름, 기본 time.
        event_col (str): 사건 열 이름, 기본 event. 나머지 열은 모두 수치 공변량.
        alpha (float): 계수 구간·PH Holm 판정 유의수준, 기본 0.05.
        penalizer (float): L2 벌점, 기본 0, 비음수. 양수면 벌점 추정의 근사 추론.
        ph_diagnostics (bool): rank 변환 Schoenfeld 검정+Holm, 기본 True.
        prediction_times (Optional[List[float]]): 새 관측 생존예측 시각, 기본 None.
            새 관측과 함께 지정하며 0~최대 학습 관측시간의 증가하는 고유값.
    """
    duration_col: str = 'time'
    event_col: str = 'event'
    alpha: float = 0.05
    penalizer: float = 0.0
    ph_diagnostics: bool = True
    prediction_times: Optional[List[float]] = None

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class CoxOutputDTO(OutputDTO):
    """Attributes:
        method (str): Efron ties/Breslow baseline Cox.
        coefficients (pd.DataFrame): 공변량별 coef, SE, HR, CI, z, 원 p값.
            Wald 검정 H0: beta_j=0, H1: beta_j!=0, 각각 양측 탐색 검정.
        ph_diagnostics (Optional[pd.DataFrame]): H0: 계수 시간불변, H1: 시간의존.
            rank Schoenfeld 통계량·원/보정 p값·판정. 미요청 None.
        predictions (Optional[pd.DataFrame]): 행=시각, 열=새 입력 행 라벨의 생존확률.
        sample_size (int): 학습 대상 수.
        events (int): 사건 수.
        alpha (float): 적용 유의수준.
        model (Any): 적합 lifelines CoxPHFitter. 기본 JSON에서 제외.
        warnings (List[str]): 벌점·예측·추론 조건.
    """
    method: str
    coefficients: pd.DataFrame
    ph_diagnostics: Optional[pd.DataFrame]
    predictions: Optional[pd.DataFrame]
    sample_size: int
    events: int
    alpha: float
    model: Any
    warnings: List[str]

# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate_inputs(data, new_data, p):
    """Args: data: 학습 DataFrame; new_data: 선택 새 공변량; p: CoxProfile.
    Returns: list: 공변량 이름.
    Raises: TypeError/ValueError: 열·수치·관측·예측 범위·설정 오류.
    """
    if not isinstance(p, CoxProfile) or not isinstance(data, pd.DataFrame): raise TypeError('CoxProfile·DataFrame이 필요합니다.')
    if not data.columns.is_unique or not data.index.is_unique: raise ValueError('학습 열·행 라벨은 고유해야 합니다.')
    if p.duration_col == p.event_col or not {p.duration_col,p.event_col}.issubset(data): raise ValueError('시간·사건 열 오류')
    run_kaplan_meier(data[p.duration_col], data[p.event_col], KaplanMeierProfile(alpha=p.alpha))
    if isinstance(p.penalizer, bool) or not isinstance(p.penalizer,(int,float)) or not np.isfinite(p.penalizer) or p.penalizer < 0: raise ValueError('penalizer 오류')
    if not isinstance(p.ph_diagnostics, bool): raise TypeError('ph_diagnostics는 bool입니다.')
    cols = [c for c in data if c not in (p.duration_col, p.event_col)]
    if not cols or data[p.event_col].sum() <= len(cols): raise ValueError('공변량보다 많은 사건이 필요합니다.')
    for frame in [data[cols]] + ([] if new_data is None else [new_data]):
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != cols or not frame.index.is_unique or len(frame)==0:
            raise ValueError('공변량 열 순서·행 라벨·비어 있는 입력 오류')
        if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in cols) or not np.isfinite(frame.to_numpy(dtype=float)).all(): raise ValueError('공변량은 유한 수치형이어야 합니다.')
    if (new_data is None) != (p.prediction_times is None): raise ValueError('새 관측과 prediction_times를 함께 지정하세요.')
    if p.prediction_times is not None:
        t = np.asarray(p.prediction_times, float)
        if t.ndim!=1 or not t.size or not np.isfinite(t).all() or (t<0).any() or (t>data[p.duration_col].max()).any() or (np.diff(t)<=0).any(): raise ValueError('예측 시각 범위·순서 오류')
    return cols

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _fit_cox(data, p):
    """Args: data: 검증된 학습 자료; p: Profile.
    Returns: Any: 적합 CoxPHFitter. 동시 사건 Efron, 기준 누적위험 Breslow.
    Raises: ImportError: 선택 의존성 lifelines 누락. lifelines 예외: 수렴·식별 실패.
    """
    from lifelines import CoxPHFitter
    return CoxPHFitter(alpha=p.alpha, penalizer=p.penalizer).fit(data.copy(), duration_col=p.duration_col, event_col=p.event_col)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_cox(data, profile: Optional[CoxProfile] = None, recipe_path: Optional[str] = None,
            overrides: Optional[Dict[str, Any]] = None, new_data=None) -> CoxOutputDTO:
    """Args:
        data: 고유 인덱스의 학습 DataFrame. 시간·사건 외 열은 공변량.
        profile: 명시적 CoxProfile, JSON 옵션과 혼용 불가.
        recipe_path: 사용자 JSON, 기본 survival-cox.json.
        overrides: 일회성 설정.
        new_data: 선택 공변량 DataFrame. 학습과 같은 열 순서, 행 라벨 보존.
    Returns: CoxOutputDTO: 계수 추론·PH 진단·선택 예측·적합 객체.
    Raises: TypeError/ValueError/OSError: 입력·레시피 오류. ImportError: lifelines 누락.
        lifelines 예외: 식별·수렴 실패. 경고는 숨기지 않습니다.
    """
    p = resolve_profile('survival-cox', profile, recipe_path=recipe_path, overrides=overrides)
    _validate_inputs(data, new_data, p); model = _fit_cox(data, p)
    diagnostics = None
    if p.ph_diagnostics:
        from lifelines.statistics import proportional_hazard_test
        diagnostics = proportional_hazard_test(model, data, time_transform='rank').summary.copy()
        diagnostics['adjusted_p_value'] = _holm(diagnostics['p'].to_numpy())
        diagnostics['reject_h0'] = diagnostics.adjusted_p_value < p.alpha
    predictions = None if new_data is None else model.predict_survival_function(new_data, times=p.prediction_times)
    notes = ['HR는 순간 위험비이며 누적 고장확률비가 아닙니다.', '계수 p값은 개별 탐색 검정, PH 판정은 Holm 보정입니다.', 'PH 비기각은 가정의 증명이 아닙니다.']
    if p.penalizer: notes.append('벌점 추정의 표준오차·p값은 근사 추론입니다.')
    return CoxOutputDTO('Cox PH (Efron ties, Breslow baseline)', model.summary.copy(), diagnostics,
                        predictions, len(data), int(data[p.event_col].sum()), p.alpha, model, notes)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rng = np.random.RandomState(1234); x = rng.normal(size=100)
    t = rng.exponential(10*np.exp(-.6*x)); stop = rng.uniform(5, 20, 100)
    d = pd.DataFrame({'time': np.minimum(t, stop), 'event': (t<=stop).astype(int), 'x': x})
    print(run_cox(d).to_json(indent=2))
