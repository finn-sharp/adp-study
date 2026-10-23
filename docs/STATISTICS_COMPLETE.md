# 통계 검정·확률분포 기능과 API 명세

평균·비율·분산·순위·범주·다변량 검정과 확률분포의 입력·가설·결과·사용 조건을 정리합니다. 기초 검정·ANOVA·사후검정 안내도 이 문서에 통합했습니다. 전체 등록 수는 [기능 카탈로그](ANALYSIS_CATALOG.md)를 따릅니다.

Statistics.pdf의 분포·검정·분류표·이미지 수식 대조와 오류 교정은 아래에 보존합니다. 기존 호환 경로는 구현 모듈로 중복 집계하지 않습니다.

## 도메인과 경로

폴더는 검정 계열·목적, 입력 계약은 표본 관계·척도를 담당합니다. 같은 Sign/Wilcoxon을 단일/대응 폴더에 중복 복사하지 않습니다.
Average/Multivariate의 단일 반응 ANOVA·사후검정은 Average/anova로, 복수 반응변수 MANOVA는 parametric/Multivariate로 이관했습니다.

| 영역 (src/stats/) | 경로 | 표본/목적 |
| --- | --- | --- |
| 분포 | distributions | PMF/PDF·CDF/SF·구간·분위수·모멘트·난수 |
| 평균 | tests/parametric/Average | 단일/독립2표본/대응 |
| ANOVA | tests/parametric/Average/anova | 독립 일원·이원, 반복측정, 독립 사후검정 |
| 모비율 | tests/parametric/Proportion | 성공수/전체수, exact/근사 |
| 분산 | tests/parametric/Variance | 단일분산χ²/두분산F |
| 다변량 | tests/parametric/Multivariate | MANOVA/Hotelling |
| 순위 | tests/nonParametric/Location | Sign/Wilcoxon/Mann–Whitney/Kruskal/Mood/Friedman |
| 범주 | tests/nonParametric/Categorical | χ²/Fisher/McNemar/Bowker/Cochran Q |
| 순서 | tests/nonParametric/Sequence | Runs |
| 가정 | tests/diagnostics | 정규성·분포·등분산·Box M |
| 연관 | tests/Association | OR/RR/CMH; src/Association/apriori는 그대로 유지 |
| 시계열 | tests/TimeSeries | ADF |

## 공통 실행 계약

Profile 생략 시 기법별 JSON을 읽으며 JSON → overrides 순서입니다. 명시적 Profile과 recipe_path/overrides는 동시에 지정하지 않습니다.
각 모듈은7개 섹션, 전체 Profile/전용 dataclass DTO 필드, Args/Returns/Raises, 실제 예제를 제공합니다.
기법별 검증·계산은 해당 파일에 있으며 공통 모듈에는 배열·수치 검증, 꼬리확률·구간과 도수표 보조 함수만 둡니다.
분포별 모수 검증·SciPy 객체 생성도 각 분포 파일에서 수행하고 조회·직렬화 계약을 공유합니다.
to_dict/to_json은 runtime 객체를 제외합니다. NaN/Infinity는 엄격 JSON에서 null입니다. 비유계 구간이나 미정의 모멘트는 원 객체와 moment_defined/기법 조건으로 구별합니다.
분포 DTO.distribution은 적합이 아닌 frozen SciPy분포이며 추가 cdf/pmf 조회에 사용합니다.

## 입력·귀무가설·정확/근사 경계

| 기법 | 입력·지원 조건 |
| --- | --- |
| one-sample-mean | x 1차원 n≥2; H0:mu=mu_0. sigma가 알려지면 Z, 아니면 t. 구간은 평균 자체, mean_difference는 평균−mu_0 |
| two-sample-mean | 독립 x/y 각n≥2; H0:mu1−mu2=diff_0. Student/Welch/Z, 구간과 mean_difference는 원 평균차 |
| paired-t | 같은 길이 x/y의 x−y; 차이 정규성, 양의 SE. H0:E(x−y)=diff_0 |
| one-sample-variance | 정규 모집단 x, n≥2, 양의 표본분산; H0:variance=variance_0. 구간은 모분산 |
| one-sample-proportion | 정수s,n, 0≤s≤n,n≥1, 0<p0<1. exact 이항/Clopper–Pearson 또는 귀무p0 분산 score Z/Wilson |
| two-sample-proportion | 독립 s1,n1,s2,n2. pooled귀무분산 Z, 각Wilson/독립Newcombe차이 구간. pooled0/1은 오류 |
| sign | 단일 기준 또는 대응 차이. 0차이 제외, 정확이항, 0이 아닌 차이 필요 |
| wilcoxon | one_sample/paired, 차이분포의 중심대칭성. auto: n≤50이고 동률/0없으면 exact, 아니면 asymptotic. 동률/0포함 exact는 오류. 모든0은 degenerate p=1과 경고 |
| mann-whitney | 독립표본 각n≥1. auto: 작은집단≤8이며 동률없으면 exact. 동률exact 거부, 나머지 asymptotic |
| kruskal-wallis | 독립k≥2 각n≥2, 동률 보정. 중앙값 해석에는 형태 가정 |
| mood-median | 독립k≥2 각n≥2, 공동중앙값 위/아래 도수표, ties above/below/ignore |
| friedman | 행=피험자/블록, 열=조건k≥3, 완전대응 순위 |
| repeated-measures-anova | 행=피험자, 열=조건k≥2. 균형 완전 일원 반복측정. none/GG epsilon 자유도 보정. 잔차변동0은 오류. 다요인·불균형·subject ID long form은 미지원 |
| runs | 원순서0/1 또는 cutoff이진화. cutoff같은 값 제외. 두상태·n≥3·양의분산, 정규근사 양측 |
| anderson-darling | n≥8, norm/expon/logistic/gumbel_l/gumbel_r. 표본모수적합, 임계값에서 지원하는alpha만. p_value=None, statistic>critical_value로 판단 |
| kolmogorov-smirnov | 연속x n≥4, norm/expon/uniform/logistic의 고정loc/scale. 표본추정은 norm/expon 양측Lilliefors; 고정모수와 동시금지. 추정expon은 loc0·비음수 |
| hotelling | 독립(n,p)두표본 각n≥2, n1+n2>p+1, pooled covariance SPD. 개별공분산은 특이해도 pooled가 SPD이면 가능 |
| box-m | 독립그룹(n,p), 각n>p, 각covariance SPD. 보정χ²근사, 다변량 정규성 |
| odds-ratio/risk-ratio | 독립2×2 행=노출/비노출, 열=사건/비사건. log-Wald p/CI, H0:OR/RR=1. 0셀 none오류/haldane명시. RR은 cohort위험설계 |
| fisher-exact | 2×2auto/exact는SciPy. RxCauto는 고정주변합 확률순서 양측enumeration. max_tables초과오류. Monte Carlo조건부표 생성, p=(hits+1)/(B+1), SE·seed·반복수 반환 |
| cmh | shape=(층,2,2), 층조건부 H0:common OR=1. 공통OR/CI·연속성보정. haldane은0셀 있는층만. 층간이질성 검정 아님 |
| bowker | 전후같은범주k×k, H0:pij=pji. df=k(k−1)/2, 빈쌍기여0/df유지. 희소쌍경고. 주변동질성과 구분 |
| cochran-q | 피험자×조건0/1, k≥2, 성공확률 동일. 변화없는 퇴화자료 오류 |
| chi-square-homogeneity | 독립모집단×범주 도수표, 범주분포 동일성. 기존Pearson 재사용, 등분산과 구분 |
| adf | 시간순x, nan_policy=raise만. c/ct/ctt/n, max_lag/autolag. H0단위근; 기각이 모든정상성 증명 아님 |

공통 결측 raise/omit: 독립표본은 각표본 제거, paired는 두값 함께, 반복측정/다변량은 결측행 전체 제거합니다. 무한대는 거부합니다.
호출자가 원 행의 대응을 보장합니다. Runs의 omit은 누락시점을 연결하므로 해석을 확인해야 합니다. ADF는 시점보존 때문에 omit을 거부합니다.
단측 모비율 구간은 alpha<0.5를 요구합니다. mean/variance/OR/RR단측 비유계 구간은 native infinity, JSON null입니다.

## 분포와 PDF 오류 교정

| 분포 | 정의·경계 |
| --- | --- |
| Bernoulli/Binomial | p∈[0,1], 이항n≥0정수 |
| Hypergeometric | N모집단/K성공/n비복원. 분산에 유한모집단(N−n)/(N−1) 보정, N1퇴화분산0 |
| Geometric | trials1부터/failures0부터, p>0 |
| Negative binomial | r성공 전failures 기본, 총trials=failures+r |
| Exponential | rate>0, scale=1/rate |
| Weibull | mean=scale·Gamma(1+1/shape); 평균을scale에 직접 대입금지 |
| Gamma/Beta | 감마shape·scale, 베타alpha·beta 모두 필요. 평균만으로 모수특정 불가 |
| Normal/Lognormal | 로그정규mu/sigma는 log(X)의 모수 |
| Chi-square/t/F | 양의자유도, t/F의 유한모멘트 존재조건 반환 |
| Cauchy | 평균·분산 미정의 |
| Poisson | 지정구간 평균발생수rate≥0 |

x는scalar/1차원, NaN거부·±inf허용, q∈[0,1], interval양끝포함입니다.
연속단일점 확률은0이고 PDF는밀도입니다. 이산구간은ceil(low)≤X≤floor(high), survival은P(X>x).
상측구간은SF차이로 CDF상쇄를 줄입니다. 분포확률구간과 모수신뢰구간을 혼동하지 않습니다.
PDF32예제의 부족한모수/방향은 examples/statistics_pdf_examples.py의 assumptions로 명시했습니다.
Kruskal은독립, CMH는층통제 연관, Bowker는대칭, Cochran Q는반복이진입니다. 원문의 잘못된분류는 교정했습니다.

## PDF 전수 대응

| 위치 | 항목 | 구현 |
| --- | --- | --- |
| 분포1~16 | 이산6종·연속10종 | distributions16개·교정예제32개 |
| 본문1/2/8 | 단일평균t/Z·독립평균t | 평균2모듈 표준화 |
| 본문3/6/9 | 단일/두모비율,3과6중복 | proportion2모듈 |
| 본문4 | AD | anderson-darling |
| 본문5/15 | 단일/대응Wilcoxon | 같은모듈의 두설계 |
| 본문7 | Runs | runs |
| 본문10/23 | χ²적합도/독립/동질성 | 기존2개+동질성API |
| 본문11/12/13/14/16/17 | Kruskal/Mood/pairedt/Sign/Mann–Whitney/Friedman | 각전용모듈 |
| 본문18/19/20 | Hotelling/MANOVA/BoxM | 신규2개+MANOVA이관 |
| 본문21/22/24/25 | OR/RR/Fisher/CMH | 각전용모듈 |
| 본문26/27/28 | McNemar/Bowker/CochranQ | 기존1개+신규2개 |
| 기타1/2/3/5 | Levene/Bartlett/Shapiro/Tukey | 기존유지 |
| 기타4/6 | KS/ADF | 신규2개 |
| 4쪽표/설명 | 반복측정ANOVA/단일분산χ²/두분산F | 신규2개+기존F |
| 8~9쪽이미지 | OR/RR log구간·층별CMH·대응McNemar | 입력/결과계약·예제 |

## 기존 호환 경로

OneWay/TwoWay/PostHoc/MANOVA/chi_square/independence/mcnemar 7개이관 파일은 호환export이며 이전 python -m예제도 동작합니다.
Univariate/Bivariate는 이전 TestProfile/Result필드·위치인자를 보존하고 _run_*를 새공개실행기로 연결합니다.
전달된 요약통계는 원표본에서 재계산하는 동작을 문서화했습니다. 이전Result에도 OutputDTO의 사전/JSON 기능을 추가했습니다.

```python
from src.stats.tests.parametric.Average.paired import PairedTProfile, run_paired_t
r = run_paired_t([120,135,128,142,130], [115,130,126,134,127],
    profile=PairedTProfile(alternative='greater'))
print(r.mean_difference, r.p_value, r.confidence_interval)
from src.stats.distributions.normal import NormalProfile, run_normal
p = run_normal(NormalProfile(mu=70, sigma=10), x=[70,85], interval=(70,85))
print(p.pdf, p.cdf, p.survival, p.interval_probability)
```

## one-sample-mean

- 파일: `src/stats/tests/parametric/Average/one_sample.py`
- 레시피: `src/Recipes/one-sample-mean.json`

```python
run_one_sample_mean(x: Any, profile: Optional[src.stats.tests.parametric.Average.one_sample.OneSampleMeanProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Average.one_sample.OneSampleMeanOutputDTO
```

### OneSampleMeanProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `mu_0` | `<class 'float'>` | `0.0` | 귀무 모집단 평균 기본 0.0. |
| `known_var` | `<class 'bool'>` | `False` | 알려진 모분산이면 Z 기본 False. |
| `sigma` | `Optional[float]` | `None` | Z의 양의 모표준편차 기본 None. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### OneSampleMeanOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `mean_difference` | `<class 'float'>` | `반환 필드` | 원 입력 x−y 또는 표본 평균−기준; 방향 명시. |
| `standard_error` | `<class 'float'>` | `반환 필드` | 평균차 표준오차. |
| `df` | `<class 'float'>` | `반환 필드` | t 자유도; Z는 infinity. |
| `confidence_interval` | `Tuple[float, float]` | `반환 필드` | 평균차/평균의 방향 일치 신뢰구간; 비유계 끝점 JSON null. |
| `n_effective` | `<class 'int'>` | `반환 필드` | 유효 표본/대응 쌍 수. |
| `sample_mean` | `<class 'float'>` | `반환 필드` | 표본 평균. |
| `sample_std` | `<class 'float'>` | `반환 필드` | ddof=1 표본 표준편차. |

## two-sample-mean

- 파일: `src/stats/tests/parametric/Average/two_sample.py`
- 레시피: `src/Recipes/two-sample-mean.json`

```python
run_two_sample_mean(x: Any, y: Any, profile: Optional[src.stats.tests.parametric.Average.two_sample.TwoSampleMeanProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Average.two_sample.TwoSampleMeanOutputDTO
```

### TwoSampleMeanProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `diff_0` | `<class 'float'>` | `0.0` | 귀무 평균차 기본 0.0. |
| `known_var` | `<class 'bool'>` | `False` | 알려진 두 모분산이면 Z 기본 False. |
| `sigma1` | `Optional[float]` | `None` | 집단1 모표준편차 기본 None. |
| `sigma2` | `Optional[float]` | `None` | 집단2 모표준편차 기본 None. |
| `equal_var` | `<class 'bool'>` | `True` | Student/Welch 선택 기본 True. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### TwoSampleMeanOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `mean_difference` | `<class 'float'>` | `반환 필드` | 원 입력 x−y 또는 표본 평균−기준; 방향 명시. |
| `standard_error` | `<class 'float'>` | `반환 필드` | 평균차 표준오차. |
| `df` | `<class 'float'>` | `반환 필드` | t 자유도; Z는 infinity. |
| `confidence_interval` | `Tuple[float, float]` | `반환 필드` | 평균차/평균의 방향 일치 신뢰구간; 비유계 끝점 JSON null. |
| `n_effective` | `<class 'int'>` | `반환 필드` | 유효 표본/대응 쌍 수. |
| `group_sizes` | `List[int]` | `반환 필드` | 첫/둘째 표본 수. |

## paired-t

- 파일: `src/stats/tests/parametric/Average/paired.py`
- 레시피: `src/Recipes/paired-t.json`

```python
run_paired_t(x: Any, y: Any, profile: Optional[src.stats.tests.parametric.Average.paired.PairedTProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Average.paired.PairedTOutputDTO
```

### PairedTProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `diff_0` | `<class 'float'>` | `0.0` | 귀무 평균차 기본 0.0. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### PairedTOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `mean_difference` | `<class 'float'>` | `반환 필드` | 원 입력 x−y 또는 표본 평균−기준; 방향 명시. |
| `standard_error` | `<class 'float'>` | `반환 필드` | 평균차 표준오차. |
| `df` | `<class 'float'>` | `반환 필드` | t 자유도; Z는 infinity. |
| `confidence_interval` | `Tuple[float, float]` | `반환 필드` | 평균차/평균의 방향 일치 신뢰구간; 비유계 끝점 JSON null. |
| `n_effective` | `<class 'int'>` | `반환 필드` | 유효 표본/대응 쌍 수. |

## one-sample-variance

- 파일: `src/stats/tests/parametric/Variance/one_sample.py`
- 레시피: `src/Recipes/one-sample-variance.json`

```python
run_one_sample_variance(x: Any, profile: Optional[src.stats.tests.parametric.Variance.one_sample.OneSampleVarianceProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Variance.one_sample.OneSampleVarianceOutputDTO
```

### OneSampleVarianceProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `variance_0` | `<class 'float'>` | `1.0` | 귀무 분산, 양수 기본 1.0. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### OneSampleVarianceOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `sample_variance` | `<class 'float'>` | `반환 필드` | ddof=1 표본 분산. |
| `df` | `<class 'int'>` | `반환 필드` | n−1 자유도. |
| `confidence_interval` | `Tuple[float, float]` | `반환 필드` | 모분산 신뢰구간. |
| `n_effective` | `<class 'int'>` | `반환 필드` | 유효 표본 수. |

## one-sample-proportion

- 파일: `src/stats/tests/parametric/Proportion/one_sample.py`
- 레시피: `src/Recipes/one-sample-proportion.json`

```python
run_one_sample_proportion(successes: Any, n: Any, profile: Optional[src.stats.tests.parametric.Proportion.one_sample.OneSampleProportionProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Proportion.one_sample.OneSampleProportionOutputDTO
```

### OneSampleProportionProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `p_0` | `<class 'float'>` | `0.5` | 귀무 성공 확률 0<p0<1 기본 0.5. |
| `method` | `<class 'str'>` | `'exact'` | exact/normal_score; exact는 이항 검정 기본 "exact". |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |

### OneSampleProportionOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `proportions` | `List[float]` | `반환 필드` | 표본 성공률. |
| `difference` | `<class 'float'>` | `반환 필드` | 단일 p_hat−p0 또는 p1−p2. |
| `confidence_intervals` | `List[Tuple[float, float]]` | `반환 필드` | 단일 exact Clopper–Pearson/score Wilson, 두 표본은 각 Wilson 구간. |
| `difference_interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 두 표본 독립 Newcombe 점수 차이 구간; 단일 None. |

## two-sample-proportion

- 파일: `src/stats/tests/parametric/Proportion/two_sample.py`
- 레시피: `src/Recipes/two-sample-proportion.json`

```python
run_two_sample_proportion(successes1: Any, n1: Any, successes2: Any, n2: Any, profile: Optional[src.stats.tests.parametric.Proportion.two_sample.TwoSampleProportionProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Proportion.two_sample.TwoSampleProportionOutputDTO
```

### TwoSampleProportionProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |

### TwoSampleProportionOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `proportions` | `List[float]` | `반환 필드` | 표본 성공률. |
| `difference` | `<class 'float'>` | `반환 필드` | 단일 p_hat−p0 또는 p1−p2. |
| `confidence_intervals` | `List[Tuple[float, float]]` | `반환 필드` | 단일 exact Clopper–Pearson/score Wilson, 두 표본은 각 Wilson 구간. |
| `difference_interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 두 표본 독립 Newcombe 점수 차이 구간; 단일 None. |

## sign

- 파일: `src/stats/tests/nonParametric/Location/sign.py`
- 레시피: `src/Recipes/sign.json`

```python
run_sign(x: Any, y: Any = None, profile: Optional[src.stats.tests.nonParametric.Location.sign.SignProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Location.sign.SignOutputDTO
```

### SignProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `sample_design` | `<class 'str'>` | `'one_sample'` | one_sample/paired 기본 "one_sample". |
| `location_0` | `<class 'float'>` | `0.0` | 단일 기준 중앙값 또는 귀무 차이 기본 0.0. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### SignOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `positive_count` | `<class 'int'>` | `반환 필드` | 양의 차이 수. |
| `negative_count` | `<class 'int'>` | `반환 필드` | 음의 차이 수. |
| `zero_count` | `<class 'int'>` | `반환 필드` | 제외한 0 차이 수. |
| `n_effective` | `<class 'int'>` | `반환 필드` | 양+음 부호 수. |

## wilcoxon

- 파일: `src/stats/tests/nonParametric/Location/wilcoxon.py`
- 레시피: `src/Recipes/wilcoxon.json`

```python
run_wilcoxon(x: Any, y: Any = None, profile: Optional[src.stats.tests.nonParametric.Location.wilcoxon.WilcoxonProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Location.wilcoxon.WilcoxonOutputDTO
```

### WilcoxonProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `sample_design` | `<class 'str'>` | `'one_sample'` | one_sample/paired 기본 "one_sample". |
| `location_0` | `<class 'float'>` | `0.0` | 귀무 위치 차이 기본 0.0. |
| `method` | `<class 'str'>` | `'auto'` | auto/exact/asymptotic 기본 "auto". |
| `zero_method` | `<class 'str'>` | `'wilcox'` | wilcox/pratt/zsplit 기본 "wilcox". |
| `correction` | `<class 'bool'>` | `False` | 근사 연속성 보정 기본 False. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### WilcoxonOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `n_effective` | `<class 'int'>` | `반환 필드` | 0이 아닌 차이 수. |
| `zero_count` | `<class 'int'>` | `반환 필드` | 0 차이 수. |
| `positive_rank_sum` | `<class 'float'>` | `반환 필드` | 양의 차이 순위 합. |
| `negative_rank_sum` | `<class 'float'>` | `반환 필드` | 음의 차이 순위 합. |

## runs

- 파일: `src/stats/tests/nonParametric/Sequence/runs.py`
- 레시피: `src/Recipes/runs.json`

```python
run_runs(x: Any, profile: Optional[src.stats.tests.nonParametric.Sequence.runs.RunsProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Sequence.runs.RunsOutputDTO
```

### RunsProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `cutoff` | `Optional[float]` | `None` | None은 입력 0/1만 허용. 수치 기준일 때 동일 값은 제외 기본 None. |
| `correction` | `<class 'bool'>` | `True` | 양측 근사 z 연속성 보정 기본 True. |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### RunsOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `runs` | `<class 'int'>` | `반환 필드` | 연속 동일 값 묶음 수. |
| `expected_runs` | `<class 'float'>` | `반환 필드` | 귀무 기대 run 수. |
| `variance_runs` | `<class 'float'>` | `반환 필드` | 귀무 run 분산. |
| `group_counts` | `List[int]` | `반환 필드` | 0/1 관측 수. |

## mann-whitney

- 파일: `src/stats/tests/nonParametric/Location/mann_whitney.py`
- 레시피: `src/Recipes/mann-whitney.json`

```python
run_mann_whitney(x: Any, y: Any, profile: Optional[src.stats.tests.nonParametric.Location.mann_whitney.MannWhitneyProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Location.mann_whitney.MannWhitneyOutputDTO
```

### MannWhitneyProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `method` | `<class 'str'>` | `'auto'` | auto/exact/asymptotic 기본 "auto". |
| `correction` | `<class 'bool'>` | `True` | 근사 연속성 보정 기본 True. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### MannWhitneyOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `u1` | `<class 'float'>` | `반환 필드` | 첫 집단 U. |
| `u2` | `<class 'float'>` | `반환 필드` | 둘째 집단 U. |
| `group_sizes` | `List[int]` | `반환 필드` | 집단별 n. |

## kruskal-wallis

- 파일: `src/stats/tests/nonParametric/Location/kruskal_wallis.py`
- 레시피: `src/Recipes/kruskal-wallis.json`

```python
run_kruskal_wallis(*groups: Any, profile: Optional[src.stats.tests.nonParametric.Location.kruskal_wallis.KruskalWallisProfile] = None, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Location.kruskal_wallis.KruskalWallisOutputDTO
```

### KruskalWallisProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### KruskalWallisOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `df` | `<class 'int'>` | `반환 필드` | 그룹 수−1. |
| `group_sizes` | `List[int]` | `반환 필드` | 그룹별 표본수. |

## mood-median

- 파일: `src/stats/tests/nonParametric/Location/mood_median.py`
- 레시피: `src/Recipes/mood-median.json`

```python
run_mood_median(*groups: Any, profile: Optional[src.stats.tests.nonParametric.Location.mood_median.MoodMedianProfile] = None, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Location.mood_median.MoodMedianOutputDTO
```

### MoodMedianProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |
| `ties` | `<class 'str'>` | `'below'` | 공동 중앙값 같은 값 처리 below/above/ignore 기본 "below". |
| `correction` | `<class 'bool'>` | `True` | 자유도1 Yates 보정 기본 True. |

### MoodMedianOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `df` | `<class 'int'>` | `반환 필드` | 그룹 수−1. |
| `group_sizes` | `List[int]` | `반환 필드` | 그룹별 표본수. |
| `pooled_median` | `<class 'float'>` | `반환 필드` | 공동 중앙값. |
| `table` | `<class 'numpy.ndarray'>` | `반환 필드` | 중앙값 위/아래의 2×k 도수표. |

## friedman

- 파일: `src/stats/tests/nonParametric/Location/friedman.py`
- 레시피: `src/Recipes/friedman.json`

```python
run_friedman(data: Any, profile: Optional[src.stats.tests.nonParametric.Location.friedman.FriedmanProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Location.friedman.FriedmanOutputDTO
```

### FriedmanProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### FriedmanOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `df` | `<class 'int'>` | `반환 필드` | 조건 수−1. |
| `n_subjects` | `<class 'int'>` | `반환 필드` | 완전 피험자 수. |
| `mean_ranks` | `<class 'numpy.ndarray'>` | `반환 필드` | 조건별 평균순위. |

## repeated-measures-anova

- 파일: `src/stats/tests/parametric/Average/anova/repeated_measures.py`
- 레시피: `src/Recipes/repeated-measures-anova.json`

```python
run_repeated_measures_anova(data: Any, profile: Optional[src.stats.tests.parametric.Average.anova.repeated_measures.RepeatedMeasuresANOVAProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Average.anova.repeated_measures.RepeatedMeasuresANOVAOutputDTO
```

### RepeatedMeasuresANOVAProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `correction` | `<class 'str'>` | `'none'` | none/greenhouse_geisser; GG는 추정 epsilon으로 자유도 보정 기본 "none". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### RepeatedMeasuresANOVAOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `df_num` | `<class 'float'>` | `반환 필드` | 조건 F 분자 자유도, 보정 적용. |
| `df_den` | `<class 'float'>` | `반환 필드` | 잔차 F 분모 자유도, 보정 적용. |
| `epsilon` | `<class 'float'>` | `반환 필드` | Greenhouse–Geisser epsilon; 보정 안 해도 추정. |
| `n_subjects` | `<class 'int'>` | `반환 필드` | 완전 피험자 수. |
| `condition_means` | `<class 'numpy.ndarray'>` | `반환 필드` | 조건 평균. |

## anderson-darling

- 파일: `src/stats/tests/diagnostics/anderson_darling.py`
- 레시피: `src/Recipes/anderson-darling.json`

```python
run_anderson_darling(x: Any, profile: Optional[src.stats.tests.diagnostics.anderson_darling.AndersonDarlingProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.diagnostics.anderson_darling.AndersonDarlingOutputDTO
```

### AndersonDarlingProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `distribution` | `<class 'str'>` | `'norm'` | 검정할 지원 분포. 모수는 표본으로 적합 기본 "norm". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### AndersonDarlingOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `critical_values` | `<class 'numpy.ndarray'>` | `반환 필드` | 공식 임계값. |
| `significance_levels` | `<class 'numpy.ndarray'>` | `반환 필드` | 임계값 유의수준(비율). |
| `critical_value` | `<class 'float'>` | `반환 필드` | 요청 alpha에 대응하는 임계값. |
| `decision_basis` | `<class 'str'>` | `반환 필드` | critical_value; p를 임의 생성하지 않음. |

## kolmogorov-smirnov

- 파일: `src/stats/tests/diagnostics/kolmogorov_smirnov.py`
- 레시피: `src/Recipes/kolmogorov-smirnov.json`

```python
run_kolmogorov_smirnov(x: Any, profile: Optional[src.stats.tests.diagnostics.kolmogorov_smirnov.KolmogorovSmirnovProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.diagnostics.kolmogorov_smirnov.KolmogorovSmirnovOutputDTO
```

### KolmogorovSmirnovProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `distribution` | `<class 'str'>` | `'norm'` | norm/expon/uniform/logistic 기본 "norm". |
| `parameters` | `Optional[Dict[str, float]]` | `None` | 고정 분포 모수 loc/scale; None은 기본 loc0/scale1 기본 None. |
| `estimated_parameters` | `<class 'bool'>` | `False` | True는 norm/expon Lilliefors 보정; 고정 parameters와 동시 금지 기본 False. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### KolmogorovSmirnovOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `distribution_parameters` | `Dict[str, float]` | `반환 필드` | 고정 또는 추정한 loc/scale. |
| `n_effective` | `<class 'int'>` | `반환 필드` | 유효 관측 수. |

## hotelling

- 파일: `src/stats/tests/parametric/Multivariate/hotelling.py`
- 레시피: `src/Recipes/hotelling.json`

```python
run_hotelling(x: Any, y: Any, profile: Optional[src.stats.tests.parametric.Multivariate.hotelling.HotellingProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.parametric.Multivariate.hotelling.HotellingOutputDTO
```

### HotellingProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### HotellingOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `t_squared` | `<class 'float'>` | `반환 필드` | Hotelling T². |
| `f_value` | `<class 'float'>` | `반환 필드` | 환산 F. |
| `df_num` | `<class 'int'>` | `반환 필드` | 특성 수. |
| `df_den` | `<class 'int'>` | `반환 필드` | n1+n2−p−1. |
| `mean_difference` | `<class 'numpy.ndarray'>` | `반환 필드` | 첫−둘째 평균 벡터. |
| `pooled_covariance` | `<class 'numpy.ndarray'>` | `반환 필드` | pooled 공분산. |
| `group_sizes` | `List[int]` | `반환 필드` | 집단별 n. |

## box-m

- 파일: `src/stats/tests/diagnostics/box_m.py`
- 레시피: `src/Recipes/box-m.json`

```python
run_box_m(*groups: Any, profile: Optional[src.stats.tests.diagnostics.box_m.BoxMProfile] = None, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.diagnostics.box_m.BoxMOutputDTO
```

### BoxMProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### BoxMOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `m_value` | `<class 'float'>` | `반환 필드` | 보정 전 M. |
| `correction_factor` | `<class 'float'>` | `반환 필드` | χ² 근사에 곱하는 1−c. |
| `df` | `<class 'int'>` | `반환 필드` | (k−1)p(p+1)/2. |
| `covariances` | `List[numpy.ndarray]` | `반환 필드` | 그룹 공분산. |
| `pooled_covariance` | `<class 'numpy.ndarray'>` | `반환 필드` | pooled 공분산. |
| `group_sizes` | `List[int]` | `반환 필드` | 그룹 n. |

## odds-ratio

- 파일: `src/stats/tests/Association/odds_ratio.py`
- 레시피: `src/Recipes/odds-ratio.json`

```python
run_odds_ratio(table: Any, profile: Optional[src.stats.tests.Association.odds_ratio.OddsRatioProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.Association.odds_ratio.OddsRatioOutputDTO
```

### OddsRatioProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `zero_correction` | `<class 'str'>` | `'none'` | none/haldane; 0셀 존재 시만 전체 셀에 0.5 더함 기본 "none". |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |

### OddsRatioOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `estimate` | `<class 'float'>` | `반환 필드` | OR 또는 RR. |
| `log_estimate` | `<class 'float'>` | `반환 필드` | log 효과크기. |
| `standard_error` | `<class 'float'>` | `반환 필드` | log 추정량 SE. |
| `confidence_interval` | `Tuple[float, float]` | `반환 필드` | 방향 일치 log-Wald 구간을 원 scale로 변환. |
| `table_used` | `<class 'numpy.ndarray'>` | `반환 필드` | 실제 계산 도수; 보정 여부 확인. |

## risk-ratio

- 파일: `src/stats/tests/Association/risk_ratio.py`
- 레시피: `src/Recipes/risk-ratio.json`

```python
run_risk_ratio(table: Any, profile: Optional[src.stats.tests.Association.risk_ratio.RiskRatioProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.Association.risk_ratio.RiskRatioOutputDTO
```

### RiskRatioProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `zero_correction` | `<class 'str'>` | `'none'` | none/haldane; 0셀 존재 시만 전체 셀에 0.5 더함 기본 "none". |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |

### RiskRatioOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `estimate` | `<class 'float'>` | `반환 필드` | OR 또는 RR. |
| `log_estimate` | `<class 'float'>` | `반환 필드` | log 효과크기. |
| `standard_error` | `<class 'float'>` | `반환 필드` | log 추정량 SE. |
| `confidence_interval` | `Tuple[float, float]` | `반환 필드` | 방향 일치 log-Wald 구간을 원 scale로 변환. |
| `table_used` | `<class 'numpy.ndarray'>` | `반환 필드` | 실제 계산 도수; 보정 여부 확인. |

## fisher-exact

- 파일: `src/stats/tests/nonParametric/Categorical/fisher_exact.py`
- 레시피: `src/Recipes/fisher-exact.json`

```python
run_fisher_exact(table: Any, profile: Optional[src.stats.tests.nonParametric.Categorical.fisher_exact.FisherExactProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Categorical.fisher_exact.FisherExactOutputDTO
```

### FisherExactProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `method` | `<class 'str'>` | `'auto'` | auto:2×2 exact, RxC enumeration; exact/enumeration/monte_carlo 기본 "auto". |
| `max_tables` | `<class 'int'>` | `100000` | RxC 정확 열거 최대 표 수. 초과 오류 기본 100000. |
| `n_resamples` | `<class 'int'>` | `10000` | Monte Carlo 표 수, 양수 기본 10000. |
| `random_state` | `<class 'int'>` | `1234` | Monte Carlo seed, 0 이상 기본 1234. |
| `alternative` | `<class 'str'>` | `'two-sided'` | two-sided/less/greater; 입력 x−y 또는 첫/둘째 집단 방향 기본 "two-sided". |

### FisherExactOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `odds_ratio` | `Optional[float]` | `반환 필드` | 2×2 cross-product OR; RxC None. |
| `n_tables` | `<class 'int'>` | `반환 필드` | 열거/재표본 표 수. 2×2 기본0. |
| `mc_standard_error` | `Optional[float]` | `반환 필드` | Monte Carlo p 추정 SE, 정확 모드 None. |
| `table_probability` | `<class 'float'>` | `반환 필드` | 관측표의 고정 주변합 조건부 확률. |

## cmh

- 파일: `src/stats/tests/Association/cmh.py`
- 레시피: `src/Recipes/cmh.json`

```python
run_cmh(tables: Any, profile: Optional[src.stats.tests.Association.cmh.CMHProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.Association.cmh.CMHOutputDTO
```

### CMHProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `correction` | `<class 'bool'>` | `False` | CMH 연속성 보정 기본 False. |
| `zero_correction` | `<class 'str'>` | `'none'` | none/haldane; 0셀 있는 층만 0.5 보정 기본 "none". |

### CMHOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `pooled_odds_ratio` | `<class 'float'>` | `반환 필드` | Mantel–Haenszel 공통 OR. |
| `confidence_interval` | `Tuple[float, float]` | `반환 필드` | 공통 OR 정규 근사 구간. |
| `n_strata` | `<class 'int'>` | `반환 필드` | 층 수. |
| `tables_used` | `<class 'numpy.ndarray'>` | `반환 필드` | 실제 계산 층별 표. |

## bowker

- 파일: `src/stats/tests/nonParametric/Categorical/bowker.py`
- 레시피: `src/Recipes/bowker.json`

```python
run_bowker(table: Any, profile: Optional[src.stats.tests.nonParametric.Categorical.bowker.BowkerProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Categorical.bowker.BowkerOutputDTO
```

### BowkerProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |

### BowkerOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `df` | `<class 'int'>` | `반환 필드` | k(k−1)/2; 빈쌍 기여는0, 자유도는 감소하지 않음. |
| `contributions` | `<class 'numpy.ndarray'>` | `반환 필드` | 위 삼각 쌍별 χ² 기여. |
| `zero_pairs` | `<class 'int'>` | `반환 필드` | 양쪽 도수가 모두0인 쌍 수. |

## cochran-q

- 파일: `src/stats/tests/nonParametric/Categorical/cochran_q.py`
- 레시피: `src/Recipes/cochran-q.json`

```python
run_cochran_q(data: Any, profile: Optional[src.stats.tests.nonParametric.Categorical.cochran_q.CochranQProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Categorical.cochran_q.CochranQOutputDTO
```

### CochranQProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise/omit. 대응 자료 omit은 쌍/블록 전체 제거; 무한대 거부 기본 "raise". |

### CochranQOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `df` | `<class 'int'>` | `반환 필드` | 조건 수−1. |
| `n_subjects` | `<class 'int'>` | `반환 필드` | 완전 피험자 수. |
| `proportions` | `<class 'numpy.ndarray'>` | `반환 필드` | 조건별 성공률. |

## adf

- 파일: `src/stats/tests/TimeSeries/adf.py`
- 레시피: `src/Recipes/adf.json`

```python
run_adf(x: Any, profile: Optional[src.stats.tests.TimeSeries.adf.ADFProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.TimeSeries.adf.ADFOutputDTO
```

### ADFProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `regression` | `<class 'str'>` | `'c'` | c/ct/ctt/n 결정항 기본 "c". |
| `max_lag` | `Optional[int]` | `None` | 최대 지연, None은 라이브러리 기본 기본 None. |
| `autolag` | `Optional[str]` | `'AIC'` | AIC/BIC/t-stat/None 기본 "AIC". |
| `nan_policy` | `<class 'str'>` | `'raise'` | raise만 허용. 시점 간격 보존을 위해 결측 제거 금지 기본 "raise". |

### ADFOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `used_lag` | `<class 'int'>` | `반환 필드` | 실제 선택 lag. |
| `nobs` | `<class 'int'>` | `반환 필드` | 검정 유효 관측 수. |
| `critical_values` | `Dict[str, float]` | `반환 필드` | 1%/5%/10% 임계값. |
| `information_criterion` | `Optional[float]` | `반환 필드` | autolag 최적 지표, 고정lag None. |

## chi-square-homogeneity

- 파일: `src/stats/tests/nonParametric/Categorical/homogeneity.py`
- 레시피: `src/Recipes/chi-square-homogeneity.json`

```python
run_chi_square_homogeneity(table: Any, profile: Optional[src.stats.tests.nonParametric.Categorical.homogeneity.ChiSquareHomogeneityProfile] = None, *, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.tests.nonParametric.Categorical.homogeneity.ChiSquareHomogeneityOutputDTO
```

### ChiSquareHomogeneityProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `0.05` | 유한한 0<alpha<1 유의수준 기본 0.05. |
| `correction` | `<class 'bool'>` | `True` | 자유도1 Yates 보정 기본 True. |

### ChiSquareHomogeneityOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `test_type` | `<class 'str'>` | `반환 필드` | 검정 이름. |
| `statistic` | `<class 'float'>` | `반환 필드` | 기법별 통계량; OR/RR은 log scale z. |
| `p_value` | `Optional[float]` | `반환 필드` | p 값; 임계값 기반 AD에는 None. |
| `reject_h0` | `<class 'bool'>` | `반환 필드` | p<alpha 또는 명시한 임계값 비교. |
| `alpha` | `<class 'float'>` | `반환 필드` | 적용 유의수준. |
| `null_hypothesis` | `<class 'str'>` | `반환 필드` | 검정한 귀무가설. |
| `alternative` | `<class 'str'>` | `반환 필드` | 대립가설 방향/omnibus. |
| `method_used` | `<class 'str'>` | `반환 필드` | 실제 사용한 정확/근사/보정 방법. |
| `details` | `Dict[str, Any]` | `반환 필드` | 문서화된 기법별 진단 필드. |
| `warnings` | `List[str]` | `반환 필드` | 가정·작은 표본·근사 제한. |
| `df` | `<class 'int'>` | `반환 필드` | (행−1)(열−1). |
| `expected` | `<class 'numpy.ndarray'>` | `반환 필드` | 동질성 귀무 기대도수. |
| `observed` | `<class 'numpy.ndarray'>` | `반환 필드` | 원 도수표. |

## distribution-bernoulli

- 파일: `src/stats/distributions/bernoulli.py`
- 레시피: `src/Recipes/distribution-bernoulli.json`

```python
run_bernoulli(profile: Optional[src.stats.distributions.bernoulli.BernoulliProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.bernoulli.BernoulliOutputDTO
```

### BernoulliProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `p` | `<class 'float'>` | `0.5` | 성공확률 0≤p≤1 기본 0.5. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### BernoulliOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-binomial

- 파일: `src/stats/distributions/binomial.py`
- 레시피: `src/Recipes/distribution-binomial.json`

```python
run_binomial(profile: Optional[src.stats.distributions.binomial.BinomialProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.binomial.BinomialOutputDTO
```

### BinomialProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `n` | `<class 'int'>` | `10` | 0 이상 시행 횟수 기본 10. |
| `p` | `<class 'float'>` | `0.5` | 성공확률 0≤p≤1 기본 0.5. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### BinomialOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-normal

- 파일: `src/stats/distributions/normal.py`
- 레시피: `src/Recipes/distribution-normal.json`

```python
run_normal(profile: Optional[src.stats.distributions.normal.NormalProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.normal.NormalOutputDTO
```

### NormalProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `mu` | `<class 'float'>` | `0.0` | 모평균 기본 0.0. |
| `sigma` | `<class 'float'>` | `1.0` | 양의 모표준편차 기본 1.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### NormalOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-hypergeometric

- 파일: `src/stats/distributions/hypergeometric.py`
- 레시피: `src/Recipes/distribution-hypergeometric.json`

```python
run_hypergeometric(profile: Optional[src.stats.distributions.hypergeometric.HypergeometricProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.hypergeometric.HypergeometricOutputDTO
```

### HypergeometricProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `N` | `<class 'int'>` | `50` | 모집단 개체 수 ≥1 기본 50. |
| `K` | `<class 'int'>` | `10` | 성공 개체 수 0≤K≤N 기본 10. |
| `n` | `<class 'int'>` | `5` | 비복원 추출 개체 수 0≤n≤N 기본 5. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### HypergeometricOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-geometric

- 파일: `src/stats/distributions/geometric.py`
- 레시피: `src/Recipes/distribution-geometric.json`

```python
run_geometric(profile: Optional[src.stats.distributions.geometric.GeometricProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.geometric.GeometricOutputDTO
```

### GeometricProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `p` | `<class 'float'>` | `0.5` | 첫 성공 확률 0<p≤1 기본 0.5. |
| `counting` | `<class 'str'>` | `'trials'` | trials(1부터)/failures(0부터) 기본 "trials". |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### GeometricOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-negative-binomial

- 파일: `src/stats/distributions/negative_binomial.py`
- 레시피: `src/Recipes/distribution-negative-binomial.json`

```python
run_negative_binomial(profile: Optional[src.stats.distributions.negative_binomial.NegativeBinomialProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.negative_binomial.NegativeBinomialOutputDTO
```

### NegativeBinomialProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `r` | `<class 'int'>` | `4` | 목표 성공 수 ≥1 기본 4. |
| `p` | `<class 'float'>` | `0.5` | 성공확률 0<p≤1 기본 0.5. |
| `counting` | `<class 'str'>` | `'failures'` | failures/r번째 성공까지 trials 기본 "failures". |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### NegativeBinomialOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-poisson

- 파일: `src/stats/distributions/poisson.py`
- 레시피: `src/Recipes/distribution-poisson.json`

```python
run_poisson(profile: Optional[src.stats.distributions.poisson.PoissonProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.poisson.PoissonOutputDTO
```

### PoissonProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `rate` | `<class 'float'>` | `3.0` | 지정 시간구간의 평균 발생 수 ≥0 기본 3.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### PoissonOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-exponential

- 파일: `src/stats/distributions/exponential.py`
- 레시피: `src/Recipes/distribution-exponential.json`

```python
run_exponential(profile: Optional[src.stats.distributions.exponential.ExponentialProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.exponential.ExponentialOutputDTO
```

### ExponentialProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `rate` | `<class 'float'>` | `0.25` | 사건 발생률 >0; 평균=1/rate 기본 0.25. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### ExponentialOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-weibull

- 파일: `src/stats/distributions/weibull.py`
- 레시피: `src/Recipes/distribution-weibull.json`

```python
run_weibull(profile: Optional[src.stats.distributions.weibull.WeibullProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.weibull.WeibullOutputDTO
```

### WeibullProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `shape` | `<class 'float'>` | `1.5` | 양의 형상 모수 기본 1.5. |
| `scale` | `<class 'float'>` | `1000.0` | 양의 척도; 평균 수명과 다름 기본 1000.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### WeibullOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-chi-square

- 파일: `src/stats/distributions/chi_square.py`
- 레시피: `src/Recipes/distribution-chi-square.json`

```python
run_chi_square(profile: Optional[src.stats.distributions.chi_square.ChiSquareDistributionProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.chi_square.ChiSquareDistributionOutputDTO
```

### ChiSquareDistributionProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `df` | `<class 'float'>` | `5.0` | 양의 자유도 기본 5.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### ChiSquareDistributionOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-gamma

- 파일: `src/stats/distributions/gamma.py`
- 레시피: `src/Recipes/distribution-gamma.json`

```python
run_gamma(profile: Optional[src.stats.distributions.gamma.GammaProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.gamma.GammaOutputDTO
```

### GammaProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `shape` | `<class 'float'>` | `2.0` | 양의 형상 k 기본 2.0. |
| `scale` | `<class 'float'>` | `3.0` | 양의 척도 theta, rate의 역수 기본 3.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### GammaOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-beta

- 파일: `src/stats/distributions/beta.py`
- 레시피: `src/Recipes/distribution-beta.json`

```python
run_beta(profile: Optional[src.stats.distributions.beta.BetaProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.beta.BetaOutputDTO
```

### BetaProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `alpha` | `<class 'float'>` | `2.0` | 양의 첫 형상 모수 기본 2.0. |
| `beta` | `<class 'float'>` | `3.0` | 양의 둘째 형상 모수 기본 3.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### BetaOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-student-t

- 파일: `src/stats/distributions/student_t.py`
- 레시피: `src/Recipes/distribution-student-t.json`

```python
run_student_t(profile: Optional[src.stats.distributions.student_t.StudentTProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.student_t.StudentTOutputDTO
```

### StudentTProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `df` | `<class 'float'>` | `10.0` | 양의 자유도; 평균·분산 존재 조건 명시 기본 10.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### StudentTOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-cauchy

- 파일: `src/stats/distributions/cauchy.py`
- 레시피: `src/Recipes/distribution-cauchy.json`

```python
run_cauchy(profile: Optional[src.stats.distributions.cauchy.CauchyProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.cauchy.CauchyOutputDTO
```

### CauchyProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `x0` | `<class 'float'>` | `0.0` | 유한 위치 모수 기본 0.0. |
| `gamma` | `<class 'float'>` | `1.0` | 양의 척도; 평균/분산 미정의 기본 1.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### CauchyOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-lognormal

- 파일: `src/stats/distributions/lognormal.py`
- 레시피: `src/Recipes/distribution-lognormal.json`

```python
run_lognormal(profile: Optional[src.stats.distributions.lognormal.LognormalProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.lognormal.LognormalOutputDTO
```

### LognormalProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `mu` | `<class 'float'>` | `0.0` | log(X)의 평균 기본 0.0. |
| `sigma` | `<class 'float'>` | `1.0` | log(X)의 양의 표준편차 기본 1.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### LognormalOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## distribution-f

- 파일: `src/stats/distributions/f.py`
- 레시피: `src/Recipes/distribution-f.json`

```python
run_f(profile: Optional[src.stats.distributions.f.FDistributionProfile] = None, *, x: Any = None, interval: Any = None, q: Any = None, n_samples: int = 0, recipe_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> src.stats.distributions.f.FDistributionOutputDTO
```

### FDistributionProfile

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `df_num` | `<class 'float'>` | `5.0` | 양의 분자 자유도 기본 5.0. |
| `df_den` | `<class 'float'>` | `10.0` | 양의 분모 자유도 기본 10.0. |
| `random_state` | `<class 'int'>` | `1234` | 난수 시드, 0 이상 기본 1234. |

### FDistributionOutputDTO

| 필드 | 타입 | 기본값 | 의미·제약 |
| --- | --- | --- | --- |
| `distribution_name` | `<class 'str'>` | `반환 필드` | 분포 이름. |
| `distribution_kind` | `<class 'str'>` | `반환 필드` | discrete/continuous. |
| `parameters` | `Dict[str, Any]` | `반환 필드` | 실제 적용 Profile 전체. |
| `evaluation_points` | `Optional[numpy.ndarray]` | `반환 필드` | 조회한 x 배열; 미요청 None. |
| `pmf` | `Optional[numpy.ndarray]` | `반환 필드` | 이산 점확률; 연속/미요청 None. |
| `pdf` | `Optional[numpy.ndarray]` | `반환 필드` | 연속 밀도; 이산/미요청 None. |
| `cdf` | `Optional[numpy.ndarray]` | `반환 필드` | P(X≤x); 미요청 None. |
| `survival` | `Optional[numpy.ndarray]` | `반환 필드` | P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None. |
| `interval` | `Optional[Tuple[float, float]]` | `반환 필드` | 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님. |
| `interval_probability` | `Optional[float]` | `반환 필드` | 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high). |
| `quantile_probabilities` | `Optional[numpy.ndarray]` | `반환 필드` | 조회 q∈[0,1]. |
| `quantiles` | `Optional[numpy.ndarray]` | `반환 필드` | CDF 역함수 PPF. |
| `inverse_survival` | `Optional[numpy.ndarray]` | `반환 필드` | SF 역함수 ISF. |
| `mean` | `Optional[float]` | `반환 필드` | 존재하고 유한한 기댓값; 미정의 None. |
| `variance` | `Optional[float]` | `반환 필드` | 존재하고 유한한 분산; 미정의 None. |
| `moment_defined` | `Dict[str, bool]` | `반환 필드` | 평균/분산의 유한 존재 여부. |
| `samples` | `Optional[numpy.ndarray]` | `반환 필드` | 선택 난수 표본; 미요청 None. |
| `distribution` | `Any` | `반환 필드` | SciPy frozen distribution. 기본 사전/JSON에서 제외. |
| `warnings` | `List[str]` | `반환 필드` | 정의역/모멘트/예제 해석 경고. |

## 검증·실행

tests/test_statistics_complete.py는 독립 SciPy/statsmodels참조값, Fisher RxC의 균등 미시상태전수 oracle·조건부Monte Carlo, 공분산 affine불변성, 설계오류·호환·명세를 검증합니다.
tests/test_probability_distributions.py는16분포·counting·구간·상측정밀도·모멘트·난수재현·JSON·PDF32예제를 검증합니다.
실행: python -m unittest discover -s tests -v, python -m examples.statistics_pdf_examples.
Python3.7/3.12 고정requirements를 사용합니다. SciPy1.17 Anderson은 임계값API의 향후변경 경고를 내지만 고정버전의 임계값계약을 유지합니다.
TensorFlow환경의 NumPy핀과 섞지 않습니다. 실제Python3.7.4시험서버·GPU는 검증범위 밖입니다.
이번 복구 후 검증은 Apple Silicon의 Python 3.12.14에서 수행했습니다. Python 3.7 구문은 검사했으며 실제 3.7 실행은 GitHub Actions에서 확인합니다.
공식근거: [SciPy통계API](https://docs.scipy.org/doc/scipy/reference/stats.html), [statsmodels통계API](https://www.statsmodels.org/stable/stats.html).

## 기초 검정 선택과 결과 해석

기존 평균 검정·ANOVA/MANOVA/사후검정에 다음 검정을 추가합니다. `src/stats/tests`는 분석 코드, 저장소 최상위 `tests`는 이를 검증하는 테스트 코드입니다.

| 질문 | 경로 (`src/stats/tests` 아래) | 실행 API | 귀무가설 / 조건 |
| --- | --- | --- | --- |
| 관측 범주 빈도가 지정 분포와 같은가? | `nonParametric/chi_square.py` | `run_chi_square(observed, expected=None)` | 기대 분포와 같음; 독립 관측, 빈도 입력, 동일 총빈도 |
| 두 범주 변수가 독립인가? | `nonParametric/independence.py` | `run_independence(table)` | 독립; 2×2 이상 교차 빈도표, 양의 주변합 |
| 같은 사람의 전후 이진 비율이 같은가? | `nonParametric/mcnemar.py` | `run_mcnemar(table)` | 대응 주변 비율이 같음; 2×2 대응 빈도표 |
| 두 정규 모집단의 분산비가 지정 값인가? | `parametric/Variance/f_test.py` | `run_f_test(x, y)` | var(x)/var(y)=ratio_0; 독립·정규 모집단 |
| 표본이 정규 모집단에서 왔는가? | `diagnostics/normality.py` | `run_normality(x)` | 정규성; Shapiro / D’Agostino–Pearson / Jarque–Bera |
| 여러 독립 집단의 분산이 같은가? | `diagnostics/homogeneity.py` | `run_homogeneity(*groups)` | 같은 모집단 분산; Levene / Bartlett / Fligner–Killeen |

각 검정은 전용 Profile/Output DTO와 별도 JSON 레시피를 제공합니다. 아래 기초 검정의 전체 Profile/DTO 필드는 [핵심 API 참조](reference/CORE_API.md)에 있습니다. 결측은 자동 제거하지 않으며 빈도 입력은 비음수 정수입니다. `expected`만 양의 실수 기대도수를 허용합니다. `alpha`는 (0,1)이고 기각 규칙은 `p_value < alpha`입니다.

```python
from src.stats.tests.nonParametric.Categorical.chi_square import run_chi_square
from src.stats.tests.nonParametric.Categorical.independence import run_independence
from src.stats.tests.nonParametric.Categorical.mcnemar import run_mcnemar
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

## ANOVA·MANOVA·사후검정 사용 조건

네 파일 모두 기존 모델과 동일하게 파일 첫머리 명세·메타데이터와 7개 섹션을 사용합니다.
Profile·Output DTO의 모든 필드, 입력 검증 및 함수의 Args·Returns·Raises를 명시합니다.
원래 ANOVA의 공개 실행 함수와 결과 클래스 이름은 유지하며 Result 이름은 전용 DTO의 별칭입니다.

| 파일 | Profile / Output DTO | 공개 실행기 | 지원 |
| --- | --- | --- | --- |
| `one_way.py` | `OneWayANOVAProfile` / `OneWayANOVAOutputDTO` | `run_oneway_anova` | 고전 일원배치·Welch ANOVA |
| `two_way.py` | `TwoWayANOVAProfile` / `TwoWayANOVAOutputDTO` | `run_twoway_anova` | 이원배치, 교호작용 선택, Type I·II·III |
| `manova.py` | `MANOVAProfile` / `MANOVAOutputDTO` | `run_manova` | 일원 요인, 여러 종속변수의 공동 평균 검정 |
| `posthoc.py` | `PostHocProfile` / `PostHocOutputDTO` | `run_posthoc` | Tukey HSD·Holm·Bonferroni |

현재 ANOVA·사후검정 경로는 `src/stats/tests/parametric/Average/anova/`이며, MANOVA는 `src/stats/tests/parametric/Multivariate/manova.py`입니다. 이전 `Average/Multivariate/`의 대문자 파일은 호환 경로입니다.

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
