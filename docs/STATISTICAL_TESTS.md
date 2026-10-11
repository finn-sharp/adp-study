# 통계 검정 선택과 결과 계약

기존 평균 검정·ANOVA/MANOVA/사후검정에 다음 검정을 추가합니다. `src/stats/tests`는 분석 코드, 저장소 최상위 `tests`는 이를 검증하는 테스트 코드입니다.

| 질문 | 경로 (`src/stats/tests` 아래) | 실행 API | 귀무가설 / 조건 |
| --- | --- | --- | --- |
| 관측 범주 빈도가 지정 분포와 같은가? | `nonParametric/chi_square.py` | `run_chi_square(observed, expected=None)` | 기대 분포와 같음; 독립 관측, 빈도 입력, 동일 총빈도 |
| 두 범주 변수가 독립인가? | `nonParametric/independence.py` | `run_independence(table)` | 독립; 2×2 이상 교차 빈도표, 양의 주변합 |
| 같은 사람의 전후 이진 비율이 같은가? | `nonParametric/mcnemar.py` | `run_mcnemar(table)` | 대응 주변 비율이 같음; 2×2 대응 빈도표 |
| 두 정규 모집단의 분산비가 지정 값인가? | `parametric/Variance/f_test.py` | `run_f_test(x, y)` | var(x)/var(y)=ratio_0; 독립·정규 모집단 |
| 표본이 정규 모집단에서 왔는가? | `diagnostics/normality.py` | `run_normality(x)` | 정규성; Shapiro / D’Agostino–Pearson / Jarque–Bera |
| 여러 독립 집단의 분산이 같은가? | `diagnostics/homogeneity.py` | `run_homogeneity(*groups)` | 같은 모집단 분산; Levene / Bartlett / Fligner–Killeen |

각 검정은 전용 Profile/Output DTO와 별도 JSON 레시피를 제공합니다. 전체 입력 필드 및 결과 필드는 [전체 명세](ANALYSIS_CATALOG.md)에 있습니다. 결측은 자동 제거하지 않으며 빈도 입력은 비음수 정수입니다. `expected`만 양의 실수 기대도수를 허용합니다. `alpha`는 (0,1)이고 기각 규칙은 `p_value < alpha`입니다.

```python
from src.stats.tests.nonParametric.chi_square import run_chi_square
from src.stats.tests.nonParametric.independence import run_independence
from src.stats.tests.nonParametric.mcnemar import run_mcnemar
from src.stats.tests.parametric.Variance.f_test import run_f_test
from src.stats.tests.diagnostics.normality import run_normality
from src.stats.tests.diagnostics.homogeneity import run_homogeneity

fit = run_chi_square([20, 30, 25, 25], expected=[25, 25, 25, 25])
ind = run_independence([[30, 20], [10, 40]], overrides={"correction": False})
paired = run_mcnemar([[20, 8], [2, 30]])  # exact=True 기본
ratio = run_f_test([1, 2, 5, 8, 9], [2, 3, 4, 6, 7],
                   overrides={"alternative": "greater", "ratio_0": 1.0})
normal = run_normality([1, 2, 3, 4, 5, 6, 8, 10])
equal = run_homogeneity([1, 2, 5, 8, 9], [2, 3, 4, 6, 7])
print(ind.statistic, ind.p_value, ind.reject_h0, ind.details, ind.warnings)
print(ind.to_json(indent=2))
```

## 해석 경계

- 카이제곱 적합도 자유도는 범주 수−1−ddof입니다. 자료로부터 모수를 추정했다면 호출자가 ddof를 지정합니다. 작은 기대도수는 경고 필드로 반환합니다.
- 독립성 검정은 자유도 1일 때만 Yates 보정이 적용됩니다. Cramér’s V는 보정 없는 Pearson 통계량으로 계산하며 편향 보정은 하지 않습니다. 2×2 소표본의 Fisher exact는 이 실행기에 포함하지 않습니다.
- McNemar의 exact 통계량은 min(b,c), p 값은 이항 정확 검정입니다. 근사는 χ²(1)이며 correction은 근사에만 적용합니다. 불일치쌍이 0이면 정확 검정 p=1, 근사 검정은 오류입니다. 독립 표본 교차표와 대응 표를 혼용하지 않습니다.
- F는 평균 차이를 검정하는 ANOVA의 F 통계량과 다른 **분산비 검정**입니다. 두 표본의 분산이 양수여야 합니다. two-sided는 두 꼬리 확률 중 작은 값의 2배이며 less/greater는 var(x)/var(y) 방향입니다.
- Shapiro는 n≥3, D’Agostino는 n≥8이며 n>5000 Shapiro p 값은 경고합니다. Jarque–Bera의 작은 표본 점근 한계를 경고합니다. 상수 표본은 오류입니다.
- median Levene는 Brown–Forsythe 변형입니다. Bartlett는 정규성에 민감합니다. trimmed 중심은 SciPy 기본 절사 비율(0.05)을 사용합니다. 중심 편차 분산이 퇴화해 통계량이 미정의이면 오류입니다.
- 정규성/등분산 검정에서 기각하지 못했다는 결과는 가정이 참이라는 증명이 아닙니다. 관측의 독립성이나 대응 설계는 호출자가 보장해야 합니다.

## 참조

새 코드는 원문 복제가 아닌 프로젝트 계약에 맞춘 구현입니다.

- [SciPy chisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
- [SciPy chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
- [statsmodels McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
- [SciPy 정규성·분산 검정](https://docs.scipy.org/doc/scipy/reference/stats.html)

## details 결과 키

| 검정 | `details` 필드 |
| --- | --- |
| 적합도 | df(int), observed/expected(ndarray) |
| 독립성 | df(int), observed/expected(ndarray), cramers_v(float, 보정 없는 Pearson 기준) |
| McNemar | discordant_pairs(int), exact(bool), correction_applied(bool), df(None 또는 1) |
| F | df_num/df_den(int), variance_ratio/ratio_0(float), alternative(str) |
| 정규성 | n(int), method(str) |
| 등분산 | group_sizes(list[int]), method(str), center(str 또는 Bartlett의 None) |
