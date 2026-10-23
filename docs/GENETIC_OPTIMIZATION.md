# 유전 알고리즘과 생산 설비 배분

일반 수치 최적화는 `src/Optimization`에 두어 ML 하이퍼파라미터 탐색과 구분합니다. genetic_algorithm·production_allocation·Visualization/optimization/convergence의 **3개 별도 모듈**은 기존 전체 명세·7섹션·Profile/JSON/DTO 계약을 따릅니다.

## 일반 GA

`run_genetic_algorithm(objective, bounds, integer_mask=None, constraints=None, initial_population=None, profile=None, *, recipe_path=None, overrides=None)`.

bounds는 유한 (d,2) 하한/상한, integer_mask는 d개 bool이며 정수 유효 경계는 ceil(lower)..floor(upper)입니다. 각 경계 절댓값은1e12 이하입니다. 같은 값의 고정 변수도 가능합니다. 초기 개체는 범위·정수성을 검증하며 남은 모집단은 독립 RandomState로 생성합니다. 목적함수는 유한한 실수 scalar, 제약 함수는 **(g<=0, h=0)**의 고정 길이 유한 벡터 쌍을 반환합니다. 목적과 제약은 부작용 없는 결정적 함수여야 합니다.

제약 위반은 sum(max(g,0))+sum(abs(h))입니다. 호출자가 잔차 단위를 정규화해야 하며 하나의 constraint_tolerance로 합을 판정합니다. 실행가능 개체는 위반 개체보다 먼저, 가능 개체끼리는 원 목적 방향, 불가능 개체끼리는 위반 합→목적 방향으로 정렬합니다. 임의 큰 페널티 상수 하나로 제약을 무시하지 않습니다. 토너먼트는 복원추출, 정수 유전자는 균등 부모 선택 교차·정수 재추출 변이, 실수는 convex 혼합 교차·범위 비율 Gaussian 변이 후 clip입니다. 상위 elite_size개는 보존합니다.

generation0은 초기 개체 평가입니다. 매 세대 모든 후보를 다시 평가하며 n_evaluations=(n_generations+1)×population_size입니다. 개선은 실행가능 진입 또는 동일 상태에서 방향을 반영한 목적/위반 감소입니다. stall은 중지 규칙일 뿐 수렴·전역 최적성 증명이 아닙니다. 실행가능 해 미발견 시 best_solution/best_objective=None, best_candidate와 위반 값은 진단으로만 반환합니다. 콜백은 JSON에 포함하지 않으며 의도치 않은 실행 코드를 레시피에서 읽지 않습니다.

## 생산 배분

`run_production_allocation(process_times, purchase_costs, operating_costs, profile=None, *, recipe_path=None, overrides=None)`은 동일 길이의 양수 분/개·비음수 비용 벡터를 받습니다. Nj는 정수, Qj=Nj×H/tj, 직렬 라인 Q=min(Qj), 목적 C=sum Nj×(purchase_j+operation_j), Q>=target입니다. 선택 budget은 같은 비용 단위의 상한입니다. 기본 연간 H=120,000분, target=50,000개, 공정별1..10대이며 불량/고장/대기/재고/감가상각은 모형 밖입니다.

기본 입력 t=[3,2,4], 구매=[5000,7000,4000], 운영=[1000,1200,800]은 사용자 제공 블로그 화면에서 확인한 교육용 상수입니다. UCI 실측 자료로 표시하지 않습니다. 단위는 만원이고 첫해 비용을 최소화합니다. 필요 최소 대수 ceil(target×t/H)=(2,1,2), 연간 능력=(80,000,60,000,60,000), 최저 비용29,800만원(구매25,000+운영4,800)입니다. 병목은 공정2·3 공동이며 목표 여유10,000개입니다.

verify_exact=True이면 지정 경계의 모든 후보를 독립 열거하고 exact_solution/exact_cost를 보고합니다. 최대 후보수 기본1,000,000을 넘으면 오류로 중단하며 실행하지 않은 검증을 완료로 표시하지 않습니다. GA 해는 exact 해로 교체하지 않고 별도 objective_gap으로 보고합니다. 전수 결과의 전역 최적성은 이 유한한 모형에 한정됩니다. 최적화에 p-value를 부여하지 않습니다.

```python
from src.Optimization.skeleton.production_allocation import main
r = main()
print(r.process_table, r.total_cost, r.exact_solution, r.objective_gap)
# caller가 만든 subplots의 ax를 재사용
from src.Visualization.optimization.convergence import plot_convergence
p = plot_convergence(r.ga_result, ax=ax)
```

일반 GA skeleton은 실제 UCI Iris의 특성 선택을 예시로 하며 train CV 오차+0.001×선택열수를 최소화합니다. 빈 부분집합을 거부하고 대치/표준화/모형 선택은 train CV 안에 둡니다. 최종 test는 선택된 열로 한 번 평가합니다. 생산 skeleton은 별도 블로그 조건을 사용합니다. import 시 다운로드·탐색·figure 생성은 없습니다.

모의고사5회에는 ML40점+생산 최적화40점+처리시간 표본 검정20점, 총14문항을 제공합니다. [모의고사 안내](../기출문제/모의고사/5회/README.md)를 참고하세요. 블로그 코드나 설명을 복사하지 않고 확인한 설계 상수로 Python 구현·지문·풀이를 새로 작성했습니다. 첫 번째 링크는 본문을 확인하지 못했으므로 해당 글의 세부 구현을 재현했다고 주장하지 않습니다.

참고 링크: [사용자 지정 생산 최적화 글](https://m.blog.naver.com/2theright/223883377283), [사용자 지정 GA 글](https://m.blog.naver.com/plasticcode/221517148322). 웹 접근 제한으로 두 번째 글은 사용자 제공 화면만 확인했습니다.

## 기법별 전체 입력·출력 명세

### genetic-algorithm

`src.Optimization.genetic_algorithm`; 기본 JSON `genetic-algorithm.json`.

**GeneticAlgorithmProfile**

```text
전용 설정의 전체 필드 명세.

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
```

**GeneticAlgorithmOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

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
```

```text
run_genetic_algorithm(objective, bounds, integer_mask=None, constraints=None, initial_population=None, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

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
```


### production-allocation

`src.Optimization.production_allocation`; 기본 JSON `production-allocation.json`.

**ProductionAllocationProfile**

```text
전용 설정의 전체 필드 명세.

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
```

**ProductionAllocationOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

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
```

```text
run_production_allocation(process_times, purchase_costs, operating_costs, profile=None, *, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

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
```


### plot-optimization-convergence

`src.Visualization.optimization.convergence`; 기본 JSON `plot-optimization-convergence.json`.

**OptimizationConvergenceProfile**

```text
전용 설정의 전체 필드 명세.

Attributes:
    title (str): 사용자 제목, 빈 문자열이면 기본 제목 기본 ''.
    color (str): 곡선 색상 기본 'C0'.
    alpha (float): 투명도 [0,1] 기본 0.8.
```

**OptimizationConvergenceOutputDTO**

```text
전용 결과; 자료·검정 정의와 runtime을 구별합니다.

Attributes:
    figure (Any): 원 축 Figure; runtime JSON 제외.
    axes (Any): 전달한 동일 Axes; runtime JSON 제외.
    artists (List[Any]): 이 호출 artist; runtime JSON 제외.
    data (pd.DataFrame): 세대별 실제 탐색 이력; infeasible 세대 목적값 None.
    diagnostics (Dict[str,Any]): 실행가능 여부·방향·최우수 목적값·중지 사유.
    parameters (Dict[str,Any]): Profile 전체.
    warnings (List[str]): 정체는 최적성 증명이 아님.
```

```text
plot_convergence(result, *, ax, profile=None, recipe_path=None, overrides=None)
전용 입력/설정을 검증하고 해당 계산을 수행합니다.

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
```

