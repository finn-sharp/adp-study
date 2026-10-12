"""
================================================================================
MANOVA.py - 일원배치 MANOVA
================================================================================
설명:
    - 일원배치 MANOVA을 수행하고 Profile·Output DTO로 입력과 결과를 정의합니다.

분석 가정:
    - 독립 관측·집단 내 다변량 정규성·동일한 집단 공분산을 가정합니다. 여러 종속변수의 공동 평균을 검정하며 잔차 공분산은 full rank여야 합니다.
    - 결측값·무한대는 사전에 처리하며 가정 검정은 자동 수행하지 않습니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.1.0    통계 명세·검증·Output DTO 통일 및 검정 구현
    2026-10-11  김재현(Finn)     v1.2.0    모델별 JSON 레시피 로딩과 공개 실행기 연결
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.2.0"
__status__ = "Development"




# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.output import OutputDTO
from src.Recipes import load_profile, resolve_profile
from src.stats.tests.parametric.Average.Multivariate._common import validate_alpha, validate_groups, validate_labels
from statsmodels.multivariate.manova import MANOVA


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------

@dataclass
class MANOVAProfile:
    """일원배치 MANOVA 입력 조건.

    Attributes:
        alpha (float): 유의수준. 기본 0.05, 유한한 0<값<1. 기각 기준은 p_value<alpha.
        criterion (str): 기각 판단 지표. pillai/wilks/hotelling/roy. 기본 pillai. 모든 지표를 함께 반환.
    """

    alpha: float = 0.05
    criterion: str = "pillai"


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------

@dataclass
class MANOVAOutputDTO(OutputDTO):
    """일원배치 MANOVA 검정 결과.

    Attributes:
        test_type (str): MANOVA 검정 이름.
        criterion (str): 선택한 기각 판단 지표.
        statistic (float): 선택한 다변량 통계량 원 값. F 값과 다름.
        f_value (float): 선택한 지표의 근사 F.
        p_value (float): 선택한 지표의 전체 집단 공동 평균 동일성 p-value.
        reject_h0 (bool): 선택한 지표의 p_value<alpha 여부.
        df_num (float): 선택한 F의 분자 자유도.
        df_den (float): 선택한 F의 분모 자유도.
        group_name (str): 요인 표시 이름.
        response_names (List[str]): 입력 종속변수 열 순서의 이름.
        group_labels (List[str]): 문자열 정렬 순서의 집단 라벨.
        group_ns (List[int]): group_labels 순서의 표본 수.
        group_means (pd.DataFrame): 행은 group_labels, 열은 response_names의 집단 평균.
        test_table (pd.DataFrame): Wilks/Pillai/Hotelling-Lawley/Roy 전체 지표. value·df_num·df_den·statistic(F)·p_value·reject_h0 열.
    """

    test_type: str
    criterion: str
    statistic: float
    f_value: float
    p_value: float
    reject_h0: bool
    df_num: float
    df_den: float
    group_name: str
    response_names: List[str]
    group_labels: List[str]
    group_ns: List[int]
    group_means: pd.DataFrame
    test_table: pd.DataFrame


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate_inputs(Y: Any, groups: Any, profile: MANOVAProfile,
                     response_names: Optional[List[str]], group_name: str) -> Any:
    """다변량 반응 및 일원 요인을 검증합니다.

    Args:
        Y: 최소 두 열의 유한한 수치형 2차원 반응 행렬 또는 DataFrame.
        groups: 행 수와 같은 길이의 집단 라벨. 각 집단 최소 2행.
        profile: MANOVAProfile.
        response_names: 선택적 고유한 반응 열 이름. None이면 DataFrame 이름 또는 Y0 등의 이름.
        group_name: 비어 있지 않은 요인 표시 이름.
    Returns:
        Any: (반응 배열, 문자열 라벨, 반응 이름 목록).
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 조건·차원·유한성·표본·이름 오류.
    """
    if not isinstance(profile,MANOVAProfile):
        raise TypeError("MANOVAProfile이 필요합니다.")
    validate_alpha(profile.alpha)
    if profile.criterion not in ('pillai','wilks','hotelling','roy'):
        raise ValueError("criterion은 pillai/wilks/hotelling/roy여야 합니다.")
    response = np.asarray(Y,dtype=float)
    if response.ndim!=2 or response.shape[1]<2 or not np.isfinite(response).all():
        raise ValueError("Y는 최소 두 열의 유한한 수치형 2차원 행렬이어야 합니다.")
    labels = validate_labels(groups,len(response))
    levels,counts = np.unique(labels,return_counts=True)
    if counts.min()<2 or len(response)-len(levels)<=response.shape[1]:
        raise ValueError("각 집단 최소 두 관측과 반응 열 수보다 큰 잔차 자유도가 필요합니다.")
    names = list(response_names) if response_names is not None else (
        [str(c) for c in Y.columns] if hasattr(Y,'columns') else ['Y'+str(i) for i in range(response.shape[1])])
    if len(names)!=response.shape[1] or any(not isinstance(s,str) or not s.strip() for s in names) or len(set(names))!=len(names):
        raise ValueError("response_names는 반응 열 수와 같은 고유한 비어 있지 않은 문자열 목록이어야 합니다.")
    if not isinstance(group_name,str) or not group_name.strip():
        raise ValueError("group_name은 비어 있지 않은 문자열이어야 합니다.")
    return response,labels,names


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run_manova(Y: np.ndarray, labels: np.ndarray, names: List[str],
                group_name: str, profile: MANOVAProfile) -> MANOVAOutputDTO:
    """한 요인의 모든 집단 대비를 공동 검정합니다.

    Args:
        Y: 검증된 다변량 반응 배열.
        labels: 검증된 문자열 집단 라벨.
        names: 반응 열 표시 이름.
        group_name: 요인 표시 이름.
        profile: 검정 조건.
    Returns:
        MANOVAOutputDTO: 네 지표의 검정표·선택 지표 판단·집단 요약.
    Raises:
        ValueError: 잔차 공분산 특이성 또는 비유한 검정 결과.
    Notes:
        절편을 제외한 모든 집단 더미 계수에 하나의 공동 대비를 적용합니다.
        다요인 MANOVA 및 MANCOVA는 이 실행기의 범위에 포함하지 않습니다.
    """
    levels,counts = np.unique(labels,return_counts=True)
    means = np.array([Y[labels==level].mean(axis=0) for level in levels])
    residual = Y.copy()
    for level,mean in zip(levels,means):
        residual[labels==level] -= mean
    if np.linalg.matrix_rank(residual)<Y.shape[1]:
        raise ValueError("잔차 공분산이 특이합니다. 중복·선형종속 반응변수를 제거하세요.")
    design = np.column_stack([np.ones(len(Y))]+[(labels==level).astype(float) for level in levels[1:]])
    contrast = np.eye(len(levels))[1:]
    table = MANOVA(Y,design).mv_test([('Group',contrast)]).results['Group']['stat'].copy()
    table = table.rename(columns={'Value':'value','Num DF':'df_num','Den DF':'df_den','F Value':'statistic','Pr > F':'p_value'}).astype(float)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("MANOVA 지표가 유한하지 않습니다. 표본 수와 공분산을 점검하세요.")
    table['reject_h0'] = table['p_value']<profile.alpha
    rows={'pillai':"Pillai's trace",'wilks':"Wilks' lambda",'hotelling':"Hotelling-Lawley trace",'roy':"Roy's greatest root"}
    selected=table.loc[rows[profile.criterion]]
    return MANOVAOutputDTO('One-way MANOVA',profile.criterion,float(selected['value']),
        float(selected['statistic']),float(selected['p_value']),bool(selected['reject_h0']),
        float(selected['df_num']),float(selected['df_den']),group_name,names,levels.tolist(),counts.tolist(),
        pd.DataFrame(means,index=levels,columns=names),table)


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_manova(Y: Any, groups: Any, profile: Optional[MANOVAProfile] = None,
               response_names: Optional[List[str]] = None, group_name: str = "Group",
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> MANOVAOutputDTO:
    """여러 반응변수의 공동 평균을 한 요인으로 검정합니다.

    Args:
        Y: 행은 관측, 열은 서로 다른 수치형 반응인 2차원 자료. 최소 두 열.
        groups: Y와 같은 행 순서의 일원 요인 집단 라벨.
        profile: None이면 manova.json에서 구성.
        response_names: 반응 열 표시 이름. None이면 입력 열 이름을 사용.
        group_name: 요인 표시 이름. 기본 Group.
        recipe_path: 사용자 JSON 경로. None이면 src/Recipes의 모델별 기본 파일.
        overrides: Profile이 없을 때 레시피 위에 적용할 명시적 설정 사전.
    Returns:
        MANOVAOutputDTO: 공동 검정 지표·집단 요약·JSON 변환 계약.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·조건·표본·잔차 공분산 또는 수치 계산 오류.
        FileNotFoundError/OSError: 레시피 파일 누락 또는 읽기 실패.
    Notes:
        네 지표를 개별 보고하지만 기본 기각 판단은 사전에 정한 한 지표(Pillai)를 사용합니다.
        종속변수별 사후검정을 여러 번 수행하면 추가 다중검정 보정이 필요합니다.
    """
    profile = resolve_profile("manova", profile, recipe_path=recipe_path, overrides=overrides)
    response,labels,names = _validate_inputs(Y,groups,profile,response_names,group_name)
    return _run_manova(response,labels,names,group_name,profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    Y = np.vstack([rng.normal(size=(20,2))+shift for shift in (0,1,2)])
    groups = ['A']*20+['B']*20+['C']*20
    result = run_manova(Y,groups,load_profile("manova"))
    print(result.test_table)
    print(result.to_json()[:200])
