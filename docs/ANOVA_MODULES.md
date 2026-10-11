# ANOVA·MANOVA·사후검정 모듈

네 파일 모두 기존 모델과 동일하게 파일 첫머리 명세·메타데이터와 7개 섹션을 사용합니다.
Profile·Output DTO의 모든 필드, 입력 검증 및 함수의 Args·Returns·Raises를 명시합니다.
원래 ANOVA의 공개 실행 함수와 결과 클래스 이름은 유지하며 Result 이름은 전용 DTO의 별칭입니다.

| 파일 | Profile / Output DTO | 공개 실행기 | 지원 |
| --- | --- | --- | --- |
| `OneWay.py` | `OneWayANOVAProfile` / `OneWayANOVAOutputDTO` | `run_oneway_anova` | 고전 일원배치·Welch ANOVA |
| `TwoWay.py` | `TwoWayANOVAProfile` / `TwoWayANOVAOutputDTO` | `run_twoway_anova` | 이원배치, 교호작용 선택, Type I·II·III |
| `MANOVA.py` | `MANOVAProfile` / `MANOVAOutputDTO` | `run_manova` | 일원 요인, 여러 종속변수의 공동 평균 검정 |
| `PostHoc.py` | `PostHocProfile` / `PostHocOutputDTO` | `run_posthoc` | Tukey HSD·Holm·Bonferroni |

파일 경로는 `src/stats/tests/parametric/Average/Multivariate/`입니다.
기존 폴더 이름 `Multivariate` 아래의 OneWay·TwoWay는 단일 반응변수 ANOVA이고,
여러 반응변수의 공동 검정은 MANOVA.py가 담당합니다.

## 사용 예제

저장소 루트에서 import합니다.

```python
from src.Recipes import load_profile
from src.stats.tests.parametric.Average.Multivariate.OneWay import run_oneway_anova
from src.stats.tests.parametric.Average.Multivariate.PostHoc import run_posthoc

groups = [[1, 2, 3, 4], [4, 5, 6, 8], [8, 9, 10, 12]]
anova = run_oneway_anova(groups)
print(anova.statistic, anova.p_value, anova.reject_h0)
print(anova.anova_table)

posthoc = run_posthoc(groups, load_profile("posthoc"), ["A", "B", "C"])
print(posthoc.comparisons)
print(posthoc.to_json(indent=2))

# 이분산: Welch ANOVA와 쌍별 Welch t + Holm 보정
welch = run_oneway_anova(groups, load_profile("oneway-anova", overrides={"equal_var": False}))
holm = run_posthoc(groups, load_profile("posthoc", overrides={"method": "holm", "equal_var": False}))
```

```python
import numpy as np
from src.Recipes import load_profile
from src.stats.tests.parametric.Average.Multivariate.MANOVA import run_manova

rng = np.random.RandomState(1234)
Y = np.vstack([rng.normal(size=(20, 2)) + shift for shift in (0, 1, 2)])
labels = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
result = run_manova(Y, labels, load_profile("manova"), ["Response1", "Response2"])
print(result.test_table)
print(result.p_value, result.reject_h0)
```

## 해석과 입력 계약

- 모든 검정은 독립 관측을 대상으로 합니다. 반복측정·대응 자료는 지원하지 않습니다.
- 정상적인 p-value 계산에 필요한 정규성·분산/공분산 가정은 각 파일 첫머리에 명시합니다.
  입력 검증은 가정 검정이나 자동 결측 대체를 수행하지 않습니다.
- 고전 ANOVA에는 양의 pooled 잔차 분산이 필요하고 Welch에는 모든 집단의 양의 분산이 필요합니다.
  Welch 표의 SS·MS는 정의하지 않으며 NaN으로 표시합니다.
- 이원배치의 Type I는 요인 1→요인 2 순차 검정입니다. Sum 코딩을 사용하며
  Type III의 절편 행을 주효과로 읽지 않도록 효과 이름으로 추출합니다.
  교호작용 모델은 모든 요인 조합에 최소 두 관측을 요구합니다.
  교호작용이 유의하면 단순효과를 별도로 확인해야 합니다.
- MANOVA는 일원 요인만 지원합니다. 다요인 MANOVA·MANCOVA는 포함하지 않습니다.
  잔차 자유도는 반응 열 수보다 커야 하며 잔차 공분산의 rank가 부족하면 오류를 반환합니다.
  Wilks·Pillai·Hotelling-Lawley·Roy를 모두 반환하고 기본 판단 지표는 Pillai입니다.
- 사후검정은 모든 집단 쌍을 하나의 family로 보정하며 평균 차이는 **집단 2 − 집단 1**입니다.
  현재 사후검정은 모든 집단에 양의 표본 분산을 요구합니다.
  Tukey HSD는 등분산일 때 사용합니다. 이분산 비교는 `equal_var=False`와 Holm·Bonferroni를 사용합니다.
- Tukey·Bonferroni의 신뢰구간은 family-wise 구간입니다. Holm 조정 신뢰구간은 제공하지 않아 NaN입니다.
  Tukey의 raw p-value·t 통계량·t 자유도도 NaN이며 조정 p-value와 구간을 반환합니다.
- 전체 검정의 유의성에 따라 사후검정을 자동 차단하지 않습니다. MANOVA 후 종속변수별 반복 비교에는
  쌍별 보정 외에 종속변수 사이의 추가 다중검정 보정을 고려해야 합니다.
- DTO의 `to_json()`은 DataFrame의 행·열 이름을 보존하고 NaN을 null로 변환합니다.

## 실행 및 검증

환경별 requirements 파일에 statsmodels·patsy를 추가했습니다.
Python 3.7.17은 statsmodels 0.13.2, Python 3.12.14는 0.14.6 조합입니다.
실제 시험 서버나 Python 3.7.4에서는 직접 검증하지 않았습니다.

```bash
python -m unittest discover -s tests -v
python -m src.stats.tests.parametric.Average.Multivariate.OneWay
python -m src.stats.tests.parametric.Average.Multivariate.TwoWay
python -m src.stats.tests.parametric.Average.Multivariate.MANOVA
python -m src.stats.tests.parametric.Average.Multivariate.PostHoc
```

통계 테스트는 SciPy·statsmodels 참조값과 고전/Welch F·p·자유도, 이원배치 6개 조건,
MANOVA 네 지표, Tukey 및 Student/Welch t의 Holm·Bonferroni 보정, 구간·비교 방향,
퇴화 설계·잘못된 입력·결과 직렬화·전체 명세를 검증합니다.

## 검정별 설정 관리

검정별 JSON의 기본값과 일회성 overrides는 [레시피 안내](RECIPES.md)를 따릅니다.
모든 실행 예제는 해당 JSON에서 Profile을 구성하며, `.py`의 전체 명세 형식은 유지합니다.
