"""
================================================================================
genetic_algorithm.py - 실수·정수 혼합 유전 알고리즘: 실행가능성 우선 선택
================================================================================
설명:
    - 전용 Profile → JSON → 검증·실행 → 전용 Output DTO.
분석 가정:
    - 유한 경계·결정적 scalar 목적·정규화 잔차, 확률적 탐색은 전역 최적성을 보장하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-23
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-23  김재현(Finn)     v1.0.0    별도 회귀/분포 모듈·명세·레시피 추가
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from src.Optimization._common import validate_profile


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class GeneticAlgorithmProfile:
    """전용 설정의 전체 필드 명세.

    Attributes:
        population_size (int): 개체 수, 4 이상 기본 80.
        generations (int): 최대 진화 세대 수, 1 이상 기본 120.
        elite_size (int): 보존할 상위 개체 수, [1,population_size) 기본 2.
        tournament_size (int): 복원추출 토너먼트 크기, [2,population_size] 기본 3.
        crossover_rate (float): 부모 쌍 교차확률 [0,1] 기본 0.8.
        mutation_rate (float): 유전자별 변이확률 [0,1] 기본 0.15.
        mutation_scale (float): 실수 변이 표준편차/변수 범위 비율, 양수 기본 0.1.
        maximize (bool): 목적함수 최대화 여부 기본 False.
        constraint_tolerance (float): 정규화 제약 위반 합 허용값, 비음수 기본 1e-08.
        stall_generations (int): 개선 없을 때 조기 종료 세대 수, 1 이상 기본 30.
        improvement_tolerance (float): 개선 판단 절대 허용값, 비음수 기본 1e-10.
        random_state (int): 독립 RandomState 시드 [0,2**32) 기본 1234.
    """

    population_size: int = 80
    generations: int = 120
    elite_size: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_scale: float = 0.1
    maximize: bool = False
    constraint_tolerance: float = 1e-08
    stall_generations: int = 30
    improvement_tolerance: float = 1e-10
    random_state: int = 1234


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class GeneticAlgorithmOutputDTO(OutputDTO):
    """전용 결과; 자료·검정 정의와 runtime을 구별합니다.

    Attributes:
        best_solution (Optional[np.ndarray]): 허용 제약을 만족한 최우수 해, 없으면 None.
        best_objective (Optional[float]): best_solution의 원 목적값, 없으면 None.
        best_candidate (np.ndarray): 제약 미충족일 때도 제공하는 최우수 후보.
        best_violation (float): 최우수 후보 정규화 양수 부등식·절대 등식 잔차 합.
        feasible (bool): 허용 제약 만족 여부. 전역 최적성 보장 아님.
        history (pd.DataFrame): generation(0=초기),best_objective,best_violation,feasible_fraction,feasible_mean_objective.
        population (np.ndarray): 마지막 세대 개체, 입력 변수 순서.
        objectives (np.ndarray): 마지막 세대 원 목적값.
        violations (np.ndarray): 마지막 세대 위반 합.
        n_evaluations (int): 실제 목적·제약 평가 횟수.
        n_generations (int): 실행 진화 세대 수.
        stop_reason (str): stall 또는 generations; 수렴·최적성 증명 아님.
        parameters (Dict[str,Any]): Profile 전체.
        warnings (List[str]): 확률 탐색·제약 및 전역 최적성 한계.
    """

    best_solution: Optional[np.ndarray]
    best_objective: Optional[float]
    best_candidate: np.ndarray
    best_violation: float
    feasible: bool
    history: pd.DataFrame
    population: np.ndarray
    objectives: np.ndarray
    violations: np.ndarray
    n_evaluations: int
    n_generations: int
    stop_reason: str
    parameters: Dict[str, Any]
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
    validate_profile(profile, GeneticAlgorithmProfile, check_alpha=False)
    if (
        profile.population_size < 4
        or not 1 <= profile.elite_size < profile.population_size
        or not 2 <= profile.tournament_size <= profile.population_size
    ):
        raise ValueError("개체/엘리트/토너먼트 크기 오류")
    if (
        profile.generations < 1
        or profile.stall_generations < 1
        or not 0 <= profile.random_state < 2**32
    ):
        raise ValueError("세대/시드 오류")
    if (
        any(not 0 <= v <= 1 for v in (profile.crossover_rate, profile.mutation_rate))
        or profile.mutation_scale <= 0
        or min(profile.constraint_tolerance, profile.improvement_tolerance) < 0
    ):
        raise ValueError("확률·허용값 오류")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run(objective, bounds, integer_mask, constraints, initial_population, profile):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        objective: 전용 명세의 objective 입력.
        bounds: 전용 명세의 bounds 입력.
        integer_mask: 전용 명세의 integer_mask 입력.
        constraints: 전용 명세의 constraints 입력.
        initial_population: 전용 명세의 initial_population 입력.
        profile: 전용 명세의 profile 입력.
    Returns:
        GeneticAlgorithmOutputDTO: 전용 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    if not callable(objective):
        raise TypeError("objective는 callable이어야 합니다.")
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.ndim != 2
        or bounds.shape[1] != 2
        or len(bounds) < 1
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("bounds는 유한한 (변수 수,2) 하한/상한입니다.")
    d = len(bounds)
    integer = (
        np.zeros(d, dtype=bool) if integer_mask is None else np.asarray(integer_mask)
    )
    if integer.shape != (d,) or integer.dtype.kind != "b":
        raise ValueError("integer_mask는 변수별 bool 벡터입니다.")
    lo = bounds[:, 0].copy()
    hi = bounds[:, 1].copy()
    lo[integer] = np.ceil(lo[integer])
    hi[integer] = np.floor(hi[integer])
    if np.any(lo > hi):
        raise ValueError("정수 경계 안에 가능한 정수가 없습니다.")
    if np.any(np.abs(bounds) > 1e12):
        raise ValueError(
            "안정적인 부동소수점 유전자 처리를 위해 |bounds|<=1e12가 필요합니다."
        )
    if constraints is not None and not callable(constraints):
        raise TypeError("constraints는 None 또는 callable입니다.")
    rng = np.random.RandomState(profile.random_state)

    def sample():
        """최적화 단계의 수치 계산.

        Args:
        Returns:
            object: 단계별 계산 결과.
        Raises:
            ValueError/TypeError: 목적 또는 제약 계산 오류.
        """
        row = rng.uniform(lo, hi)
        for j in np.flatnonzero(integer):
            row[j] = rng.randint(int(lo[j]), int(hi[j]) + 1)
        return row

    population = np.array([sample() for _ in range(profile.population_size)])
    if initial_population is not None:
        initial = np.asarray(initial_population, dtype=float)
        if (
            initial.ndim != 2
            or initial.shape[1] != d
            or not 1 <= len(initial) <= len(population)
            or not np.isfinite(initial).all()
            or np.any(initial < lo)
            or np.any(initial > hi)
            or np.any(initial[:, integer] != np.rint(initial[:, integer]))
        ):
            raise ValueError("초기 개체 shape/범위/정수 제약 오류")
        population[: len(initial)] = initial
    constraint_size = [None]
    evaluations = 0

    def evaluate(pop):
        """최적화 단계의 수치 계산.

        Args:
        pop: 현재 후보 또는 계산 자료.
        Returns:
            object: 단계별 계산 결과.
        Raises:
            ValueError/TypeError: 목적 또는 제약 계산 오류.
        """
        values = []
        violations = []
        for row in pop:
            value = np.asarray(objective(row.copy()))
            if (
                value.ndim != 0
                or value.dtype.kind not in "iuf"
                or not np.isfinite(value)
            ):
                raise ValueError("목적함수는 유한한 실수 scalar를 반환해야 합니다.")
            if constraints is None:
                g = h = np.array([], dtype=float)
            else:
                raw = constraints(row.copy())
                if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                    raise ValueError(
                        "constraints는 (g<=0,h=0) 벡터 쌍을 반환해야 합니다."
                    )
                g, h = (np.atleast_1d(np.asarray(v, dtype=float)) for v in raw)
                if any(v.ndim != 1 or not np.isfinite(v).all() for v in (g, h)):
                    raise ValueError("제약 잔차는 유한한 1차원 벡터여야 합니다.")
            size = (len(g), len(h))
            if constraint_size[0] is None:
                constraint_size[0] = size
            if constraint_size[0] != size:
                raise ValueError("제약 차원이 후보마다 변합니다.")
            violation = float(np.maximum(g, 0).sum() + np.abs(h).sum())
            if not np.isfinite(violation):
                raise ValueError("제약 위반 합이 유한하지 않습니다.")
            values.append(float(value))
            violations.append(violation)
        return np.asarray(values), np.asarray(violations)

    def rank_key(i):
        """최적화 단계의 수치 계산.

        Args:
        i: 현재 후보 또는 계산 자료.
        Returns:
            object: 단계별 계산 결과.
        Raises:
            ValueError/TypeError: 목적 또는 제약 계산 오류.
        """
        feasible = violations[i] <= profile.constraint_tolerance
        score = -values[i] if profile.maximize else values[i]
        return (0, score) if feasible else (1, violations[i], score)

    history = []
    stall = 0
    previous = None
    stop = "generations"
    for generation in range(profile.generations + 1):
        values, violations = evaluate(population)
        evaluations += len(population)
        order = sorted(range(len(population)), key=rank_key)
        best = order[0]
        feasible = violations <= profile.constraint_tolerance
        current = (bool(feasible[best]), float(values[best]), float(violations[best]))
        history.append(
            dict(
                generation=generation,
                best_objective=current[1] if current[0] else None,
                best_violation=current[2],
                feasible_fraction=float(feasible.mean()),
                feasible_mean_objective=(
                    float(values[feasible].mean()) if feasible.any() else None
                ),
            )
        )
        score = (
            (-current[1] if profile.maximize else current[1])
            if current[0]
            else current[2]
        )
        improved = (
            previous is None
            or (current[0] and not previous[0])
            or (
                current[0] == previous[0]
                and score < previous[1] - profile.improvement_tolerance
            )
        )
        if improved:
            previous = (current[0], score)
            stall = 0
        else:
            stall += 1
        if generation == profile.generations:
            break
        if stall >= profile.stall_generations:
            stop = "stall"
            break
        children = [population[i].copy() for i in order[: profile.elite_size]]

        def parent():
            """최적화 단계의 수치 계산.

            Args:
            Returns:
                object: 단계별 계산 결과.
            Raises:
                ValueError/TypeError: 목적 또는 제약 계산 오류.
            """
            choices = rng.randint(0, len(population), size=profile.tournament_size)
            return population[min(choices, key=rank_key)].copy()

        while len(children) < profile.population_size:
            a, b = parent(), parent()
            if rng.rand() < profile.crossover_rate:
                swap = rng.rand(d) < 0.5
                weight = rng.rand(d)
                child = np.where(
                    integer, np.where(swap, a, b), weight * a + (1 - weight) * b
                )
            else:
                child = a
            mutate = rng.rand(d) < profile.mutation_rate
            for j in np.flatnonzero(mutate):
                child[j] = (
                    rng.randint(int(lo[j]), int(hi[j]) + 1)
                    if integer[j]
                    else child[j]
                    + rng.normal(scale=profile.mutation_scale * (hi[j] - lo[j]))
                )
            child = np.clip(child, lo, hi)
            child[integer] = np.rint(child[integer])
            children.append(child)
        population = np.asarray(children)
    best = sorted(range(len(population)), key=rank_key)[0]
    ok = bool(violations[best] <= profile.constraint_tolerance)
    candidate = population[best].copy()
    notes = [
        "유전 알고리즘은 확률적 탐색이며 stall 종료나 실행가능 해 발견은 전역 최적성 증명이 아닙니다.",
        "constraints는 호출자가 단위를 정규화한 g<=0, h=0 잔차입니다. 위반 합 허용값을 확인하세요.",
        "목적/제약 함수는 결정적이고 부작용이 없어야 하며, 모든 개체를 다시 평가합니다.",
    ]
    if not ok:
        notes.append(
            "실행가능 해를 찾지 못했습니다. best_solution과 best_objective는 None입니다."
        )
    return GeneticAlgorithmOutputDTO(
        candidate.copy() if ok else None,
        float(values[best]) if ok else None,
        candidate,
        float(violations[best]),
        ok,
        pd.DataFrame(history),
        population.copy(),
        values,
        violations,
        evaluations,
        generation,
        stop,
        asdict(profile),
        notes,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_genetic_algorithm(
    objective,
    bounds,
    integer_mask=None,
    constraints=None,
    initial_population=None,
    profile=None,
    *,
    recipe_path=None,
    overrides=None,
):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        objective: 전용 명세의 objective 입력.
        bounds: 전용 명세의 bounds 입력.
        integer_mask: 전용 명세의 integer_mask 입력.
        constraints: 전용 명세의 constraints 입력.
        initial_population: 전용 명세의 initial_population 입력.
        profile: 전용 Profile 또는 None.
        recipe_path: Profile 생략 시 사용자 JSON 경로.
        overrides: Profile 생략 시 JSON 위에 적용할 재정의.
    Returns:
        GeneticAlgorithmOutputDTO: 입력 순서와 설정을 보존한 결과.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    profile = resolve_profile(
        "genetic-algorithm", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate(profile)
    return _run(
        objective, bounds, integer_mask, constraints, initial_population, profile
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실제 자료 연결은 분야별 skeleton과 전체 명세에서 제공합니다.
