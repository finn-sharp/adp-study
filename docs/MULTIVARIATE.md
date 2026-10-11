# ANOVA·MANOVA·사후검정 모듈 명세

계약의 기준은 `src/stats/anova.py`, `manova.py`, `posthoc.py` 내부의 파일 설명,
Profile/Result `Attributes`, 공개 함수 `Args/Returns/Raises/Notes`다.
Python 3.7에서는 별도 가상환경에 `requirements-py37.txt`를 설치한다.

## 제공 범위

| 목적 | API | 설정/결과 |
|---|---|---|
| 일원 ANOVA | `run_anova` | `method='one_way', equal_var=True` → ANOVAOutputDTO.analysis의 OneWayANOVAOutputDTO |
| Welch ANOVA | `run_anova` | `method='one_way', equal_var=False` → ANOVAOutputDTO.analysis의 Welch 결과 |
| 이원 ANOVA | `run_anova` | `method='two_way'`, 교호작용·SS Type I/II/III → ANOVAOutputDTO.analysis의 TwoWayANOVAOutputDTO |
| 균형 반복측정 | `run_anova` | `method='repeated'`, subject/within → ANOVAOutputDTO.analysis의 RepeatedANOVAOutputDTO |
| 여러 반응의 요인 효과 | `run_manova` | 반응 2개 이상, 요인 및 선택적 교호작용 → MANOVAOutputDTO |
| 등분산 독립 집단 사후검정 | `run_posthoc` | Tukey HSD, 불균형 표본 크기 허용 |
| 이분산 독립 집단 사후검정 | `run_posthoc` | Games–Howell, Welch 자유도 및 studentized range |
| 쌍별 독립 t 사후검정 | `run_posthoc` | pooled/Welch t + Holm 또는 Bonferroni |

ANOVA는 기존 OneWay/TwoWay/repeated 구현을 재사용한다. DataFrame의 공백이나 문장부호가
포함된 반응/요인 열 이름도 받으며 이원 계산에는 안전한 내부 이름을 사용한다.
MANOVA는 단변량 ANOVA를 여러 번 수행하는 것과 다르다. 요인별 네 통계량
(Wilks, Pillai, Hotelling–Lawley, Roy)을 제공하고 절편 검정은 반환 표에서 제외한다.

## ANOVA 사용

```python
from src.stats import ANOVAProfile, run_anova
result = run_anova(df, ANOVAProfile(response='score', factors=['group']))
print(result.statistic, result.p_value, result.reject_h0)
print(result.anova_table)
welch = run_anova(df, ANOVAProfile('score', ['group'], equal_var=False))
two_way = run_anova(df, ANOVAProfile(
    'score', ['group', 'treatment'], method='two_way', interaction=True, ss_type=3
))
repeated = run_anova(repeated_df, ANOVAProfile(
    'score', ['visit'], method='repeated', subject='participant'
))
```

입력은 결측 없는 long-form DataFrame이며 요인은 범주형 의미를 가진다.
일원 집단 순서는 최초 등장 순서이고 각 집단에 최소 두 관측이 필요하다.
이원 Type III는 Sum 대비를 사용한다. 반복측정은 대상·요인 조합당 한 관측과
균형 설계가 필요하며 집단 간 요인이나 구형성 보정은 제공하지 않는다.

## MANOVA 사용

```python
from src.stats import MANOVAProfile, run_manova
result = run_manova(df, MANOVAProfile(
    responses=['score', 'response_time'], factors=['group']
))
print(result.tables['group'])
# 각 표: Value, Num DF, Den DF, F Value, Pr > F, reject_h0
```

각 요인은 최소 두 수준, 설계행렬은 full rank, 잔차 자유도는 반응 수 이상이어야 한다.
반응이 선형 종속이면 잔차 공분산이 특이하므로 명시적으로 거부한다.
`interaction=True`는 요인들의 모든 교호작용을 포함하므로 설계 크기에 주의한다.
원 statsmodels 객체는 `result.model`과 `result.multivariate_test`로 접근한다.
`fit()` 대신 `mv_test()`를 사용하는 구현이다.

## 사후검정 사용

```python
from src.stats import PostHocProfile, run_posthoc
# 등분산
hsd = run_posthoc(df.score, df.group, PostHocProfile(method='tukey'))
# 이분산
games = run_posthoc(df.score, df.group, PostHocProfile(
    method='games_howell', equal_var=False
))
# Welch 쌍별 t + Holm 가족 단위 보정
holm = run_posthoc(df.score, df.group, PostHocProfile(
    method='pairwise_t', equal_var=False, correction='holm'
))
print(games.table)
```

| 반환 열 | 의미 |
|---|---|
| `group1`, `group2` | 원래 집단 라벨, 최초 등장 순서의 조합 |
| `mean_diff` | group2 평균 − group1 평균 |
| `statistic` | Games–Howell q / 쌍별 t의 부호 있는 t. Tukey는 NaN |
| `df` | Tukey는 N−K, 쌍별 t는 pooled/Welch, Games–Howell은 Welch 자유도 |
| `p_raw` | 쌍별 t의 미보정 p. 자체 가족 단위 보정 방법은 NaN |
| `p_adjusted` | 가족 단위 보정된 p |
| `ci_lower`, `ci_upper` | 평균차 구간. Holm은 별도 동시 구간을 계산하지 않으므로 NaN |
| `reject_h0` | p_adjusted < alpha |

Tukey와 Bonferroni 구간은 각 방법의 가족 단위 구간이며 Games–Howell은 근사 동시 구간이다.
Holm의 p 보정을 개별 t 구간 보정으로 표시하지 않는다.
기법 선택은 독립성·정규성·등분산 가정과 분석 계획에 맞춰 사용자가 정한다.
ANOVA 유의성으로 사후검정 실행 여부를 자동 결정하지 않는다. 이원 ANOVA의 단순효과나
반복측정 쌍 비교를 독립 일원 사후검정으로 대체하지 않는다.
MANOVA 뒤 여러 반응에 따로 비교를 적용하면 반응 간 추가 다중성도 고려해야 한다.
리샘플링으로 합성한 표본을 독립 관측으로 취급한 이 통계적 추론은 제공하지 않는다.

## 재현 예제 및 검증

- `python -m examples.multivariate_workflow`
- `codebook/11장. ANOVA·MANOVA·사후검정.ipynb`
- `tests/test_multivariate.py`: scipy/statsmodels 기준 수치 비교, 특이 입력 거부,
  Games–Howell 두 집단 결과와 Welch t 비교, 보정 p/구간·평균차 방향 확인.
- Python 3.7.12와 3.12에서 전체 51개 테스트 통과. 구체적 버전은 [검증 기록](VALIDATION.md).

공식 API: [MANOVA](https://www.statsmodels.org/v0.13.5/generated/statsmodels.multivariate.manova.MANOVA.html),
[Tukey HSD](https://www.statsmodels.org/v0.13.5/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html),
[studentized_range](https://docs.scipy.org/doc/scipy-1.7.1/reference/generated/scipy.stats.studentized_range.html).

모든 결과는 [Output DTO 공통 계약](OUTPUT_DTOS.md)을 따릅니다. ANOVA의 방법별 상세 결과는 `result.analysis`입니다.
