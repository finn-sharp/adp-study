"""
================================================================================
manova.py - 다변량 반응의 요인별 MANOVA
================================================================================
설명:
    - 두 개 이상 수치 반응에 대해 Wilks/Pillai/Hotelling-Lawley/Roy 검정표를 제공합니다.
    - 요인 및 선택적 교호작용을 Sum 대비 수식으로 검정하며 rank 결손을 사전에 거부합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-11
    - 버전: 1.0.0
변경 이력: 2026-10-11 v1.0.0 Profile/Result·mv_test 및 rank 검증 추가
사용 예제: python -m examples.multivariate_workflow
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.1.0"
__status__ = "Development"

from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd


@dataclass
class MANOVAProfile:
    """MANOVA 조건.

    Attributes:
        responses: 두 개 이상의 서로 다른 수치 반응 열 이름.
        factors: 하나 이상의 범주 요인 열 이름. 각 요인 최소 두 수준.
        interaction: True이면 모든 요인의 교호작용을 포함, False이면 주효과만.
        alpha: 각 요인/통계량의 유의수준 (0,1). 요인 간 다중검정 보정은 하지 않음.
    """
    responses: List[str]
    factors: List[str]
    interaction: bool = False
    alpha: float = 0.05


@dataclass
class MANOVAOutputDTO(OutputDTO):
    """요인별 다변량 검정 결과.

    Attributes:
        model: statsmodels MANOVA 객체; fit() 대신 mv_test() 사용.
        multivariate_test: statsmodels MultivariateTestResults 원 객체.
        tables: 요인명/교호작용명 -> 네 통계량의 Value/Num DF/Den DF/F Value/Pr > F/reject_h0 표.
        responses: 입력 반응 열 순서.
        factors: 입력 요인 열 순서.
        residual_df: 설계행렬에 따른 잔차 자유도.
    """
    model: Any
    multivariate_test: Any
    tables: Dict[str, pd.DataFrame]
    responses: List[str]
    factors: List[str]
    residual_df: int


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
MANOVAResult = MANOVAOutputDTO


def run_manova(data: pd.DataFrame, profile: MANOVAProfile) -> MANOVAOutputDTO:
    """독립 관측의 여러 반응을 함께 검정합니다.

    Args:
        data: 반응과 범주 요인의 long-form DataFrame. 완전 관측만 허용.
        profile: 반응 목록·요인 목록·교호작용·유의수준.
    Returns:
        MANOVAOutputDTO: 요인/교호작용별 네 통계량의 검정표. 절편 검정은 tables에서 제외.
    Raises:
        ValueError: 열/수준/표본 부족, rank 결손 설계, 특이 잔차 공분산 또는 비유한 검정표.
        ImportError: statsmodels/patsy 미설치.
    Notes:
        독립 관측·집단 내 다변량 정규성과 공분산 동질성을 가정합니다. 반복측정 MANOVA나
        강건 MANOVA는 아닙니다. 네 통계량 중 무엇을 보고할지는 분석 전에 정합니다.
        개별 반응의 사후검정을 여러 번 실행할 때 반응 간 추가 다중성도 고려해야 합니다.
    """
    from statsmodels.multivariate.manova import MANOVA
    from patsy import dmatrices
    p = profile
    if p is None or len(p.responses) < 2 or not p.factors or not isinstance(p.interaction, bool) or not 0 < p.alpha < 1:
        raise ValueError("두 개 이상의 반응, 요인 및 유효한 설정이 필요합니다.")
    names = list(p.responses)+list(p.factors)
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique or len(set(names)) != len(names) or not set(names).issubset(data.columns) or data[names].isna().any().any():
        raise ValueError("중복 없는 열과 결측 없는 관측이 필요합니다.")
    Y = data[p.responses].to_numpy(dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError("반응은 유한한 수치여야 합니다.")
    frame = pd.DataFrame({"y{}".format(i):Y[:,i] for i in range(Y.shape[1])})
    term_names = {}
    for i, name in enumerate(p.factors):
        codes, levels = pd.factorize(data[name], sort=False)
        if len(levels) < 2:
            raise ValueError("각 요인에 최소 두 수준이 필요합니다.")
        frame["f{}".format(i)] = codes.astype(str)
        term_names["C(f{}, Sum)".format(i)] = name
    formula = " + ".join("y{}".format(i) for i in range(Y.shape[1]))+" ~ "+(" * " if p.interaction else " + ").join(term_names)
    endog, design = dmatrices(formula, frame, return_type="dataframe")
    rank = np.linalg.matrix_rank(design.values)
    residual_df = len(frame)-rank
    if rank != design.shape[1] or residual_df < Y.shape[1]:
        raise ValueError("설계행렬이 full rank이고 잔차 자유도가 반응 수 이상이어야 합니다.")
    residual = Y-design.values.dot(np.linalg.lstsq(design.values,Y,rcond=None)[0])
    if np.linalg.matrix_rank(residual) < Y.shape[1]:
        raise ValueError("잔차 공분산이 특이합니다. 중복/선형 종속 반응을 제거하세요.")
    try:
        model = MANOVA.from_formula(formula, frame)
        result = model.mv_test()
    except (np.linalg.LinAlgError, RuntimeError) as exc:
        raise ValueError("MANOVA 수치 계산에 실패했습니다: {}".format(exc)) from exc
    tables = {}
    for term, item in result.results.items():
        if term == "Intercept":
            continue
        table = item["stat"].copy().astype(float)
        if not np.isfinite(table[["Value","Num DF","Den DF","F Value","Pr > F"]].values.astype(float)).all():
            raise ValueError("유한한 MANOVA 통계량을 계산할 수 없습니다.")
        table["reject_h0"] = table["Pr > F"].astype(float) < p.alpha
        label = ":".join(term_names[part] for part in term.split(":"))
        tables[label] = table
    return MANOVAOutputDTO(model,result,tables,list(p.responses),list(p.factors),residual_df)
