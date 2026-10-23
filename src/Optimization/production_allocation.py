"""
================================================================================
production_allocation.py - 직렬 생산라인 정수 설비 배분과 독립 전수검증
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 동일 공정 병렬 처리·직렬 병목 최소능력·연간 가동분과 비용 단위를 고정합니다.
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
from itertools import product
from src.Optimization._common import validate_profile
from src.Optimization.genetic_algorithm import run_genetic_algorithm,GeneticAlgorithmProfile

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class ProductionAllocationProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        annual_minutes (float): 기계별 연간 가동분, 양수 기본 120000.0.
        target_production (float): 직렬 라인 연간 최소 생산량, 양수 기본 50000.0.
        min_machines (int): 공정별 최소 정수 기계 수, 1 이상 기본 1.
        max_machines (int): 공정별 최대 정수 기계 수, min 이상 기본 10.
        budget (Optional[float]): 연간 구매+운영 비용 상한; 입력 비용과 동일 단위 기본 None.
        population_size (int): GA 개체 수, 4 이상 기본 80.
        generations (int): GA 최대 진화 세대, 1 이상 기본 120.
        random_state (int): GA 독립 난수 시드 기본 1234.
        verify_exact (bool): 작은 유한 공간 전수열거로 독립 검증 기본 True.
        max_exact_candidates (int): 전수열거 후보 한도, 양수; 초과 시 명시적 오류 기본 1000000.
    """
    annual_minutes: float = 120000.0
    target_production: float = 50000.0
    min_machines: int = 1
    max_machines: int = 10
    budget: Optional[float] = None
    population_size: int = 80
    generations: int = 120
    random_state: int = 1234
    verify_exact: bool = True
    max_exact_candidates: int = 1000000

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class ProductionAllocationOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        ga_result (Any): 별도 GeneticAlgorithmOutputDTO; 최적성 보장 없음.
        process_table (pd.DataFrame): 선택 GA 해의 공정별 기계·능력·구매·운영·비용; 미발견 시 빈 표.
        annual_production (Optional[float]): GA 해의 병목 최소 처리량.
        total_cost (Optional[float]): GA 해의 연간 구매+운영 비용, 입력과 같은 단위.
        exact_solution (Optional[np.ndarray]): 전수열거 최적 정수 해, 미실행/불가능 시 None.
        exact_cost (Optional[float]): 독립 전수열거 최소 비용.
        exact_status (str): not_requested / optimal / infeasible.
        objective_gap (Optional[float]): GA 비용-전수 최소 비용; 비교 불가 시 None.
        minimum_required (np.ndarray): 처리능력만으로 계산한 공정별 필요 최소 정수 수; 경계 하한 반영.
        parameters (Dict[str,Any]): 적용 Profile 전체.
        warnings (List[str]): 결정적 용량·비용 가정과 추론 범위.
    """
    ga_result: Any
    process_table: pd.DataFrame
    annual_production: Optional[float]
    total_cost: Optional[float]
    exact_solution: Optional[np.ndarray]
    exact_cost: Optional[float]
    exact_status: str
    objective_gap: Optional[float]
    minimum_required: np.ndarray
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
    validate_profile(profile,ProductionAllocationProfile,check_alpha=False)
    if profile.annual_minutes<=0 or profile.target_production<=0 or profile.min_machines<1 or profile.max_machines<profile.min_machines or profile.max_exact_candidates<1:raise ValueError('가동시간/목표/기계 경계/검증한도 오류')
    if profile.budget is not None and (isinstance(profile.budget,bool) or not np.isfinite(profile.budget) or profile.budget<0):raise ValueError('budget은 유한한 비음수 비용입니다.')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------

def _run(process_times, purchase_costs, operating_costs, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        process_times: 전용 명세의 process_times 입력.
        purchase_costs: 전용 명세의 purchase_costs 입력.
        operating_costs: 전용 명세의 operating_costs 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        ProductionAllocationOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    times=np.asarray(process_times,dtype=float);buy=np.asarray(purchase_costs,dtype=float);operating=np.asarray(operating_costs,dtype=float)
    if times.ndim!=1 or len(times)<1 or buy.shape!=times.shape or operating.shape!=times.shape or not all(np.isfinite(a).all() for a in (times,buy,operating)) or np.any(times<=0) or np.any(buy<0) or np.any(operating<0):raise ValueError('공정별 양수 처리분·비음수 비용의 같은 길이 벡터가 필요합니다.')
    d=len(times);unit=buy+operating
    if profile.verify_exact and (profile.max_machines-profile.min_machines+1)**d>profile.max_exact_candidates:raise ValueError('전수열거 한도 초과: verify_exact=False 또는 작은 경계를 명시하세요.')
    def objective(x):
        """최적화 단계의 수치 계산.
        
        Args:
        x: 현재 후보 또는 계산 자료.
        Returns:
            object: 단계별 계산 결과.
        Raises:
            ValueError/TypeError: 목적 또는 제약 계산 오류.
        """
        return float(x@unit)
    def constraints(x):
        """최적화 단계의 수치 계산.
        
        Args:
        x: 현재 후보 또는 계산 자료.
        Returns:
            object: 단계별 계산 결과.
        Raises:
            ValueError/TypeError: 목적 또는 제약 계산 오류.
        """
        capacity=x*profile.annual_minutes/times
        g=list((profile.target_production-capacity)/profile.target_production)
        if profile.budget is not None:g.append((objective(x)-profile.budget)/max(profile.budget,1.))
        return np.asarray(g),np.array([])
    profile_ga=GeneticAlgorithmProfile(population_size=profile.population_size,generations=profile.generations,random_state=profile.random_state,constraint_tolerance=0.)
    ga=run_genetic_algorithm(objective,np.tile([profile.min_machines,profile.max_machines],(d,1)),integer_mask=np.ones(d,dtype=bool),constraints=constraints,profile=profile_ga)
    required=np.maximum(profile.min_machines,np.ceil(profile.target_production*times/profile.annual_minutes)).astype(int)
    exact=None;exact_cost=None;status='not_requested'
    if profile.verify_exact:
        status='infeasible'
        for raw in product(range(profile.min_machines,profile.max_machines+1),repeat=d):
            candidate=np.asarray(raw,dtype=int);g,_=constraints(candidate)
            if np.all(g<=0):
                cost=objective(candidate)
                if exact_cost is None or cost<exact_cost:exact=candidate;exact_cost=cost;status='optimal'
    table=pd.DataFrame();production=cost=None
    if ga.feasible:
        x=ga.best_solution.astype(int);capacity=x*profile.annual_minutes/times;production=float(capacity.min());cost=objective(x)
        table=pd.DataFrame(dict(process=np.arange(1,d+1),machines=x,minutes_per_unit=times,annual_capacity=capacity,purchase_cost=x*buy,operating_cost=x*operating,total_cost=x*unit,bottleneck=capacity==capacity.min()))
    gap=None if cost is None or exact_cost is None else cost-exact_cost
    notes=['직렬 공정, 동일 공정 내 병렬 설비, 연 1회 구매+운영 비용의 결정적 모형입니다. 가동률·고장·대기·재고·불량·감가상각은 포함하지 않습니다.','목표 생산량/설비 수/비용은 검정통계량이 아닙니다. 최적화에 p-value를 부여하지 않습니다.','비용 단위는 호출자가 통일합니다. 블로그 예제 단위는 만원입니다.']
    if gap is not None and gap>1e-8:notes.append('GA 해가 독립 전수 최적값보다 비쌉니다. exact 해를 GA가 찾았다고 보고하지 마세요.')
    return ProductionAllocationOutputDTO(ga,table,production,cost,exact,exact_cost,status,gap,required,asdict(profile),notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------

def run_production_allocation(process_times, purchase_costs, operating_costs, profile=None, *, recipe_path=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        process_times: 전용 명세의 process_times 입력.
        purchase_costs: 전용 명세의 purchase_costs 입력.
        operating_costs: 전용 명세의 operating_costs 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        ProductionAllocationOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile=resolve_profile('production-allocation',profile,recipe_path=recipe_path,overrides=overrides)
    _validate(profile)
    return _run(process_times, purchase_costs, operating_costs,profile)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
