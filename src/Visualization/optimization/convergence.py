"""
================================================================================
convergence.py - 유전 알고리즘 실행가능 최우수 목적값의 세대별 변화
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 전달한 ax에만 표시하며 해를 다시 탐색하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-23
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-23  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
================================================================================
"""
__author__='김재현(Finn)'
__email__='penguin.klg@gmail.com'
__version__='1.0.0'
__status__='Development'

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass,asdict
from typing import Any,Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.Visualization._common import PlotOutputDTO,start,finish,validate_profile as validate_plot_profile

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class OptimizationConvergenceProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        title (str): 사용자 제목, 빈 문자열이면 기본 제목 기본 ''.
        color (str): 곡선 색상 기본 'C0'.
        alpha (float): 투명도 [0,1] 기본 0.8.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class OptimizationConvergenceOutputDTO(PlotOutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        figure (Any): 원 축 Figure; runtime JSON 제외.
        axes (Any): 전달한 동일 Axes; runtime JSON 제외.
        artists (List[Any]): 이 호출 artist; runtime JSON 제외.
        data (pd.DataFrame): 세대별 실제 탐색 이력; infeasible 세대 목적값 None.
        diagnostics (Dict[str,Any]): 실행가능 여부·방향·최우수 목적값·중지 사유.
        parameters (Dict[str,Any]): Profile 전체.
        warnings (List[str]): 정체는 최적성 증명이 아님.
    """
    figure: Any
    axes: Any
    artists: List[Any]
    data: pd.DataFrame
    diagnostics: Dict[str,Any]
    parameters: Dict[str,Any]
    warnings: List[str]

# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

def _validate(profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
    Returns:
        object: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    validate_plot_profile(profile,OptimizationConvergenceProfile)

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run(result, ax, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        result: 전용 명세의 result 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        OptimizationConvergenceOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    if not hasattr(result,'history') or not hasattr(result,'feasible'):raise TypeError('GeneticAlgorithmOutputDTO가 필요합니다.')
    data=result.history.copy()
    if not {'generation','best_objective','best_violation','feasible_fraction'}.issubset(data.columns):raise ValueError('탐색 이력 열 누락')
    before=start(ax)
    ax.plot(data.generation,data.best_objective,color=profile.color,alpha=profile.alpha,label='Best feasible objective')
    ax.set(xlabel='Generation',ylabel='Objective (original units)',title=profile.title or 'Genetic algorithm search');ax.legend()
    return finish(ax,before,data,dict(feasible=result.feasible,maximize=result.parameters['maximize'],best_objective=result.best_objective,stop_reason=result.stop_reason),profile,OptimizationConvergenceOutputDTO,'Genetic algorithm search',['실행가능 후보가 없는 세대는 목적값 곡선을 그리지 않습니다. 곡선 정체는 전역 최적성 증명이 아닙니다.'])

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def plot_convergence(result, *, ax, profile=None, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        result: 전용 명세의 result 입력.
        ax: 전용 명세의 ax 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        OptimizationConvergenceOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('plot-optimization-convergence',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(result, ax,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
