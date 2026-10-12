> 구현 전 사전분석 기록입니다. 현재 동작은 [통계 전체 구현 명세](STATISTICS_COMPLETE.md)를 참조하세요.

# 통계 PDF 사전분석 및 구현 차이 명세

## 1. 조사 기준과 결론

- 조사 자료: 사용자 첨부 Statistics.pdf, 물리적 PDF 페이지 1~10 전체. 본문, 4쪽 분류표, 8~9쪽 이미지 표와 수식까지 확인.
- 대조 저장소: finn-sharp/adp-study.
- 기준 main: 81a76293594f831cd04601586fe533ede57f895f, PR #5 병합 커밋. 작업 브랜치 9c2d29c와 파일 내용은 동일함.
- 판단 근거: 실제 src 파일의 실행 함수·Profile·결과 필드·지원 method, 레시피 registry. 문서에 이름이 있는 것만으로 구현 완료로 판단하지 않음.
- 이 작업은 사전분석·명세 작성만 수행함. 분석 모듈 구현이나 저장소 문서 변경은 하지 않음.

추가 대상은 **확률분포 16종의 공개 분석 기능 + 신규 검정·효과크기 23개 기능 단위 + 기존 카이제곱 계산을 재사용하는 동질성 API/명세 확장 1개**이다.
23개는 API/기능 단위의 수이며 파일 수가 아니다. Wilcoxon의 단일 표본·대응 표본은 하나의 기법으로 세고 두 입력 모드를 제공한다. OR와 RR은 서로 다른 효과크기이므로 각각 센다. 23개 중 21개는 본문 항목, 2개는 4쪽 표/설명에만 있는 반복측정 ANOVA·단일 모집단 분산 검정이다.

PDF의 모든 항목을 반영하려면 분포·검정만 만들지 않고, 각 기법의 가정·입력 구조·귀무가설·대립가설·예제와 자료 내 오류 교정도 함께 명세해야 한다. PDF는 구현 범위의 기준이며, 통계학 전체를 포괄하는 목록이라는 의미는 아니다.

## 2. 기존 구현으로 충족되는 내용

| PDF 항목 | 페이지 | 확인한 구현 | 판단과 남은 범위 |
| --- | --- | --- | --- |
| 단일 표본 평균 t/Z | 4 | parametric/Average/Univariate.py의 _run_t_test/_run_z_test | 계산 구현 있음. 공개 실행기·표준 OutputDTO·레시피는 정비 대상 |
| 독립 두 표본 평균 t | 5 | parametric/Average/Bivariate.py의 _run_two_sample_t_test | Student/Welch와 대립가설 방향 지원. 동일한 표준화 정비 필요 |
| 카이제곱 적합도 | 4, 6 | nonParametric/chi_square.py, run_chi_square | 기대도수·ddof·기각·근사 경고 있음. 신규 계산 모듈 제외 |
| 카이제곱 독립성 | 6, 8 | nonParametric/independence.py, run_independence | Yates·기대도수·자유도·Cramér’s V 있음. 10번/23번 중복 |
| McNemar | 4, 9 | nonParametric/mcnemar.py, run_mcnemar | 정확 이항·근사 χ²·연속성 보정 있음 |
| 독립 일원 ANOVA | 4의 표 | parametric/Average/Multivariate/OneWay.py | 고전/Welch 지원. 대응 반복측정 ANOVA는 별도 미구현 |
| MANOVA | 7 | parametric/Average/Multivariate/MANOVA.py | 일원 요인, Wilks/Pillai/Hotelling–Lawley/Roy 지원. PDF 범위 충족 |
| Levene·Bartlett | 10 | diagnostics/homogeneity.py | method 선택 지원. median Levene는 Brown–Forsythe 변형 |
| Shapiro–Wilk | 10 | diagnostics/normality.py | method=shapiro 지원 |
| Tukey HSD | 10 | parametric/Average/Multivariate/PostHoc.py | 독립 집단 Tukey 지원 |
| 두 모집단 분산 비교 F | 4의 설명 및 분포 예시 | parametric/Variance/f_test.py | 분산비·귀무 비율·세 방향 지원. 신규 계산 모듈 제외 |

기존 MANOVA의 Hotelling–Lawley trace는 독립적인 Hotelling T² 분석 API가 아니다. MANOVA를 호출해 유사한 전체 집단 검정 결과를 얻을 수 있더라도, T² 통계량·평균차 벡터·공분산 등 전용 결과 계약이 없으므로 PDF의 Hotelling T²는 미구현으로 판단한다.

기존 normality 모듈은 Shapiro/D’Agostino/Jarque–Bera만 지원하고 Anderson–Darling·KS를 제공하지 않는다. nonParametric/onesample.py와 twosample.py는 실제로 0바이트이며 구현으로 집계하지 않는다.

현재 PDF보다 넓게 이미 제공하는 이원 ANOVA·Holm/Bonferroni 사후검정·Fligner·D’Agostino·Jarque–Bera·독립 두 표본 Z는 재구현하지 않는다.

## 3. 확률분포 16종: 모두 독립 공개 모듈/API 추가 대상

검정 내부에서 scipy.stats.norm/t/f/chi2를 사용한 것은 분포 조회·확률 계산용 Profile/DTO/API가 구현되었다는 뜻이 아니다. src에는 분포 전용 패키지나 registry가 없다.

공통 제안: src/stats/distributions 아래에서 기법별 Profile·전용 Output DTO·run 함수와 별도 레시피를 제공한다. 이산형은 PMF, 연속형은 PDF를 반환하고, CDF/SF·구간 확률·PPF/ISF·기댓값/분산을 구분한다. 난수 생성은 seed가 있는 선택 기능이다. PDF의 32개 예제는 이 기능을 사용하는 예제로 정리하며, 모수가 부족하거나 확률 대신 밀도를 요청하는 예제는 교정/보충 조건을 명시한다.

| 번호 | 분포 | PDF 페이지 | 필수 입력 및 명세할 정의 | 상태 |
| --- | --- | --- | --- | --- |
| 1 | Bernoulli | 1 | p, x∈{0,1}; 성공/실패 확률 | 신규 |
| 2 | Binomial | 1 | n, p, 성공 수 k; 정수 시행 횟수 | 신규 |
| 3 | Normal | 1 | mu, sigma>0; 점의 밀도와 구간 확률 구분 | 신규 |
| 4 | Hypergeometric | 1 | N(모집단), K(성공 개체), n(비복원 추출), k; SciPy 인자와 명시적 매핑 | 신규 |
| 5 | Geometric | 1~2 | p; 첫 성공까지 시행 횟수 기준 1부터인지 실패 수 기준 0부터인지 명시 | 신규 |
| 6 | Negative binomial | 2 | r, p; r번째 성공 전 실패 수와 총 시행 횟수를 구분 | 신규 |
| 7 | Poisson | 2 | lambda≥0, 발생 횟수 k | 신규 |
| 8 | Exponential | 2 | rate lambda>0; SciPy scale=1/lambda | 신규 |
| 9 | Weibull | 2 | shape k>0, scale lambda>0; 평균과 scale 구분 | 신규 |
| 10 | Chi-square | 2~3 | df>0; 상측 확률과 분산 검정 예제 | 신규 |
| 11 | Gamma | 3 | shape k>0, scale theta>0; rate와 혼동하지 않음 | 신규 |
| 12 | Beta | 3 | alpha,beta>0; 평균만으로 두 모수를 결정할 수 없음 | 신규 |
| 13 | Student t | 3 | df>0; 좌/우/양측 꼬리 확률, 모멘트 존재 조건 | 신규 |
| 14 | Cauchy | 3 | location x0, scale gamma>0; 평균·분산 미정의 | 신규 |
| 15 | Lognormal | 3~4 | underlying normal mu,sigma>0; SciPy s=sigma, scale=exp(mu), loc=0 | 신규 |
| 16 | F | 4 | df_num,df_den>0; 꼬리 확률과 모멘트 존재 조건 | 신규 |

공통 Output DTO 제안 필드: distribution_name, parameters, distribution_kind, evaluation_points, density_or_mass, cdf, survival, interval_probability, quantiles, mean, variance, moment_defined, warnings, runtime distribution 객체(기본 JSON 제외). 모든 필드를 무조건 계산하는 방식보다는 Profile의 요청 연산에 맞추고 미요청 값은 None으로 구분한다. PMF/PDF는 의미가 다르므로 실제 DTO에서는 구별된 필드를 권장한다.

분포의 중심 확률 구간은 모수에 대한 신뢰구간이 아니다. interval 결과 명칭을 구별한다. 이산형의 P(X>k), P(X≥k)와 구간 끝점 포함 정책, 연속형의 밀도/확률 구분은 명세와 테스트에 포함한다.

## 4. 신규 검정·효과크기 23개: 입력·검정 대상·반환 계약

다음 경로는 확정 구현이 아니라 기존 구조를 확장하는 제안이다. 아래 파일명은 src/stats/tests 기준 상대 경로이다. 각각 전용 Profile/Output DTO/레시피/공개 실행기를 제공한다.

| ID | 기법 / 제안 파일 | PDF 위치 | 필수 입력·귀무가설 | 전용 반환 내용·조건 |
| --- | --- | --- | --- | --- |
| T01 | 단일 모비율 / parametric/Proportion/one_sample.py | 5, #3·#6 중복 | 성공 수 s, 전체 n, p0; H0:p=p0 | 비율·차이·p·구간. 이항 정확과 정규 근사 구분, 근사 분산 기준 명시 |
| T02 | 두 독립 집단 모비율 / parametric/Proportion/two_sample.py | 5, #9 | s1,n1,s2,n2; H0:p1=p2 | 두 비율·차이·z·p·구간. pooled 귀무분산과 구간 방법 명시 |
| T03 | 대응 t / parametric/Average/paired.py | 6, #13 | 같은 개체의 x/y 또는 차이 d; H0:E(d)=diff0 | t·df=n−1·평균차·SE·신뢰구간·유효 쌍 수. 대응 위치와 차이의 정규성 |
| T04 | 단일 모집단 분산 χ² / parametric/Variance/one_sample.py | 4 설명 | 표본 x, sigma0²>0; H0:σ²=σ0² | χ²·df=n−1·표본분산·분산/표준편차 구간. 정규 모집단 가정 |
| T05 | 일원 반복측정 ANOVA / parametric/Average/Multivariate/repeated_measures.py | 4 표 | subject,condition,value; H0:조건 평균 동일 | F·df·p·조건 평균·완전 피험자 수. 대응 설계·구형성·불균형 지원 범위 명시 |
| T06 | Sign / nonParametric/sign.py | 4 표, 6 #14 | x와 기준 중앙값 또는 대응 차이; H0:중앙값 동일 | 양/음/0 개수·유효 n·정확 이항 p. 0차이 제외 정책 |
| T07 | Wilcoxon signed-rank / nonParametric/wilcoxon.py | 4 표, 5 #5, 7 #15 | 단일 기준차 또는 대응 x/y; 0 중심 대칭 차이분포의 귀무가설 | W+·W−·통계량·p·0/동률 개수·exact/근사·zero_method. 단순 중앙값 검정으로만 설명하지 않음 |
| T08 | Runs / nonParametric/runs.py | 4 표, 5 #7 | 관측 순서가 있는 이진 배열 또는 명시적 기준으로 이진화한 수치열; H0:배열 무작위 | run 수·두 값의 수·기대 run·분산·z·p. 입력을 정렬하지 않음 |
| T09 | Mann–Whitney U / nonParametric/mann_whitney.py | 4 표, 7 #16 | 독립 x/y; 분포 동일성/확률적 비교 | U1/U2·p·표본 수·exact/근사·동률 처리. 중앙값 차이 해석에는 분포 형태 가정 필요 |
| T10 | Kruskal–Wallis H / nonParametric/kruskal_wallis.py | 4 표, 6 #11 | k개 독립 표본; 순위 분포 차이 | H·df=k−1·p·그룹 수·동률 보정. 대응표본 검정이 아님 |
| T11 | Mood median / nonParametric/mood_median.py | 4 표, 6 #12 | 2개 이상 독립 표본; 모집단 중앙값 동일 | 공동 중앙값·위/아래 빈도표·χ²·df·p·ties 정책 |
| T12 | Friedman / nonParametric/friedman.py | 4 표, 7 #17 | 동일 피험자의 k≥3 조건 자료 | 순위 χ²·df=k−1·p·조건 평균순위·완전 쌍/블록 수·동률 보정 |
| T13 | Anderson–Darling / diagnostics/anderson_darling.py | 5, #4 | 표본, 지원하는 귀무 분포; 분포 적합성 | A²·임계값·지원 유의수준·기각. 사용 라이브러리가 p를 제공하지 않으면 p=None, 근거 없는 p 생성 금지 |
| T14 | KS / diagnostics/kolmogorov_smirnov.py | 10 기타 #4 | 연속 표본, 완전히 지정된 CDF와 모수; H0:F=F0 | D·p·대립가설·모수 출처. 같은 표본으로 정규 모수를 추정한 경우 일반 KS p 사용 금지; Lilliefors/보정 경로 필요 |
| T15 | Hotelling T² / parametric/Multivariate/hotelling.py | 7, #18 | PDF의 필수 범위는 두 독립 다변량 표본; H0:평균 벡터 동일 | T²·환산 F·df·p·평균차 벡터·pooled covariance. 비특이 공분산과 표본수 조건 |
| T16 | Box M / diagnostics/box_m.py | 7, #20 | 그룹별 다변량 표본; H0:공분산 행렬 동일 | M·보정 χ²/사용 근사·df·p·그룹 공분산·표본 수. logdet·정규성·특이성 처리 |
| T17 | OR / Association/odds_ratio.py | 8, #21 이미지 수식 포함 | 독립 2×2 표, 명시한 사건/노출 방향; H0:OR=1 | OR·logOR·SE·CI·z/p 또는 선택 정확 추론. 0셀 보정 정책을 명시 |
| T18 | RR / Association/risk_ratio.py | 8, #22 이미지 수식 포함 | 독립 2×2 표, cohort/risk 해석 가능한 설계; H0:RR=1 | 그룹 위험·RR·logRR·SE·CI·z/p. case-control 표에서 위험비 해석 불가 |
| T19 | Fisher exact / nonParametric/fisher_exact.py | 8, #24 | 비음수 정수 2×2 표, 행·열 사건 방향 | p·OR 추정 방식·대립 방향. PDF에 적힌 RxC도 범위에 포함하려면 일반화 정확 검정 경로 별도 필요 |
| T20 | Cochran–Mantel–Haenszel / Association/cmh.py | 9, #25 이미지 표 포함 | 층별 2×2 표; H0:층 조건부 연관 없음/공통 OR=1 | CMH χ²·p·pooled OR·CI·층별 표·보정 정책. 층간 효과차 검정과 구분 |
| T21 | McNemar–Bowker / nonParametric/bowker.py | 9, #27 | 같은 범주를 갖는 전후 k×k 대응표; H0:pij=pji | 대칭성 χ²·df·p·쌍별 기여·0쌍 정책. 주변동질성 검정과 다른 귀무가설 |
| T22 | Cochran Q / nonParametric/cochran_q.py | 4 표, 10 #28 | 동일 피험자의 k개 이진 조건 (n,k); H0:조건 성공확률 동일 | Q·df=k−1·p·조건 비율·피험자 수. 이진 대응 자료, 일반 다범주 자료가 아님 |
| T23 | ADF / TimeSeries/adf.py | 10 기타 #6 | 시간 순 1차원 시계열, lag/autolag/trend; H0:단위근 존재 | ADF 통계량·근사 p·선택 lag·nobs·임계값·정보기준. 결정항 설정과 유효 표본 수 |

T01의 One-Sample P Test는 문맥상 모비율 검정과 동일하다. 다중비교 보정 방법만 추가된 비모수 사후검정, 두 표본 KS, KPSS, 단일/대응 Hotelling, 다요인 MANOVA 등은 PDF에 직접 요구되지 않으므로 위 23개에 넣지 않았다.

## 5. 기존 모듈의 부분 구현과 재사용 확장

### 5.1 카이제곱 동질성: 새 통계량 계산은 불필요, 분석 계약은 추가

- PDF 6쪽 #10에 명시된 여러 독립 모집단의 범주분포 동일성 검정.
- 현재 run_independence의 Pearson 계산과 같은 수식을 사용할 수 있다.
- 그러나 현재 DTO.null_hypothesis는 독립성을 서술하고 동질성 설계는 공개 API/문서로 구분하지 않는다.
- 별도 run_chi_square_homogeneity 또는 profile.test_type='homogeneity'를 통해 H0·모집단 라벨·범주 라벨·기대도수·df·p를 올바르게 반환하도록 확장한다.
- 기존 diagnostics/homogeneity.py는 **분산의 동질성(등분산)**이며 범주분포 동질성을 대체하지 않는다.
- 대응 정사각표의 주변동질성(Stuart–Maxwell)도 이것과 다르다. PDF 필수 독립 동질성에 섞지 않는다.

### 5.2 기존 단일/독립 두 표본 Z/t의 표준화

계산 로직은 기존 구현으로 인정한다. 신규 알고리즘으로 집계하지 않고 기존 _run_* 함수를 유지하면서 공개 실행기, 전용 OutputDTO의 to_dict/to_json, 모델별 레시피, 일관된 전체 필드 설명·Args/Returns/Raises·입력 검증·예제를 정비한다.

표본 평균·표준편차를 외부에서 별도로 넘기는 기존 내부 함수 방식보다 공개 실행기에서 원 표본으로부터 계산하도록 한다. 기존 Result 이름은 호환 별칭 또는 변환 경로로 유지한다. 신뢰구간·자유도 등 결과 확장은 기존 호출 호환성을 고려해 적용한다.

### 5.3 Fisher RxC와 반복측정 ANOVA의 지원 경계

- PDF는 Fisher를 n×m 범주표로 설명한다. 2×2만 만들면 PDF 전체 반영이라는 표현을 쓸 수 없다. RxC 일반화(예: Fisher–Freeman–Halton)의 정확 열거/조건부 Monte Carlo를 별도 모드로 설계하고, 정확 계산 한도·시드·반복 수·MC 오차·실행 방법을 반환해야 한다. 2×2 지원과 일반 RxC 지원을 구분한다. Python 3.7에 고정된 구버전 SciPy의 지원을 별도 검증한다.
- 반복측정 ANOVA를 추가할 때 완전·균형 일원 반복측정 범위를 명시한다. statsmodels AnovaRM은 균형 설계 위주이며 구형성 보정이 자동 제공된다고 가정할 수 없다. 지원하지 않는 불균형/결측 설계를 조용히 독립 ANOVA로 계산하지 않는다. 결측/구형성 보정 확장은 필수 최소범위와 따로 명시한다.

## 6. PDF 원문 번호 전수 대응표

### 6.1 검정 본문 #1~#28

| PDF 번호 | 내용 | 처리 |
| --- | --- | --- |
| 1 | One-sample t | 기존 계산 유지, 표준화 |
| 2 | One-sample Z | 기존 계산 유지, 표준화 |
| 3 | One-Sample P | T01 신규; #6과 같은 기법 |
| 4 | Anderson–Darling | T13 신규 |
| 5 | 단일 Wilcoxon | T07 신규; #15와 공통 기법·다른 입력 모드 |
| 6 | One-sample proportion | T01; #3과 중복 |
| 7 | Runs | T08 신규 |
| 8 | Independent t | 기존 Student/Welch 유지; 잘못 적힌 H0 교정 |
| 9 | Two-sample proportion | T02 신규 |
| 10 | χ² 적합도·동질성·독립성 | 적합도/독립성 기존, 독립 동질성 의미/API 확장 |
| 11 | Kruskal–Wallis | T10 신규, independent로 교정 |
| 12 | Mood median | T11 신규 |
| 13 | Paired t | T03 신규 |
| 14 | Sign | T06 신규, 단일/대응 지원 |
| 15 | 대응 Wilcoxon | T07; #5와 공통 기법 |
| 16 | Mann–Whitney | T09 신규; rank-sum 별칭 설명 |
| 17 | Friedman | T12 신규 |
| 18 | Hotelling T² | T15 신규 |
| 19 | MANOVA | 기존 일원 MANOVA 유지 |
| 20 | Box M | T16 신규 |
| 21 | OR | T17 신규; 그림의 점추정·CI 수식까지 반영 |
| 22 | RR | T18 신규; 그림의 점추정·CI 수식까지 반영 |
| 23 | χ² 독립성 | 기존; #10의 독립성과 중복 |
| 24 | Fisher exact | T19 신규; 2×2/RxC 경계 명세 |
| 25 | CMH | T20 신규; 그림의 층별 2×2와 pooled 추론 반영 |
| 26 | McNemar | 기존 정확/근사 검정 유지 |
| 27 | McNemar–Bowker | T21 신규; 대칭성/주변동질성 구분 |
| 28 | Cochran Q | T22 신규; 이진 반복 조건으로 교정 |

### 6.2 기타 검정 6개

| 기타 번호 | 내용 | 처리 |
| --- | --- | --- |
| 1 | Levene | 기존 |
| 2 | Bartlett | 기존 |
| 3 | Shapiro–Wilk | 기존 |
| 4 | Kolmogorov–Smirnov | T14 신규 |
| 5 | Tukey HSD | 기존 |
| 6 | ADF | T23 신규 |

### 6.3 본문 번호 외 항목

| 위치 | 항목 | 처리 |
| --- | --- | --- |
| 4쪽 분포 설명 | 한 모집단의 분산 검정 χ² | T04 신규 |
| 4쪽 분포 설명 | 두 모집단의 분산 검정 F | 기존 F 검정 |
| 4쪽 분류표 | 독립 일원 ANOVA | 기존 고전/Welch |
| 4쪽 분류표 | Paired Oneway ANOVA | T05 반복측정 신규 |
| 4쪽 분류표 | 나머지 sign/signed-rank/runs/McNemar/Friedman/Cochran Q/Mann–Whitney/Mood/Kruskal 등 | 본문 및 T01~T23과 대응, 추가 중복 집계 없음 |
| 8쪽 이미지 | OR·RR 점추정·log scale CI·SE 수식 | T17/T18 결과 계약에 포함 |
| 9쪽 이미지 | 층별 치료×결과와 Mantel–Haenszel 집계 | T20에 포함 |
| 9쪽 이미지 | 같은 사람의 전후 2×2 표 | 기존 McNemar 예제/입력 명세 보강 |

## 7. 그대로 구현하면 안 되는 원문 오류·혼동

| 위치 | 원문 문제 | 반영할 교정 |
| --- | --- | --- |
| 1쪽 초기하 분포 | 분산에 유한 모집단 보정 누락 | Var(X)=n·(K/N)·(1−K/N)·(N−n)/(N−1), N>1. 퇴화 N=1 별도 처리 |
| 1~3쪽 및 여러 분포 예제 | 연속형 점의 확률과 PDF 혼용; 이하/이상의 사건에 PDF 사용 | 연속형 P(X=x)=0; PDF는 밀도. 이하 CDF, 이상 SF, 구간 CDF 차이 |
| 2쪽 음이항 | 식은 실패 수 기준인데 예시는 총 시행 횟수 | 실패 수 k_fail=t−r; 총 시행 수 평균 r/p, 실패 수 평균 r(1−p)/p |
| 2쪽 지수 | lambda를 사건 간격으로 표현 | lambda는 rate, 평균 간격은 1/lambda |
| 2쪽 Weibull | 예제에서 평균 수명을 scale에 그대로 대입 | mean=scale·Gamma(1+1/shape). 평균이 주어지면 scale 역산 |
| 3쪽 Beta·Gamma 일부 예제 | 평균/성공확률만으로 필요한 shape 모수를 특정할 수 없음 | 필요한 모수를 추가 입력하거나 예제 가정을 명시. 임의로 유일 답 산출 금지 |
| 3쪽 Lognormal | mu/sigma를 원자료 위치/척도처럼 표현 | log(X)의 정규 평균·표준편차임을 명시 |
| 5쪽 Independent t | H0가 관측분포=기대분포라고 적힘 | H0는 두 독립 모집단 평균 동일/평균차 diff0 |
| 6쪽 χ² 적합도 | 독립성 H0와 적합도 사례를 혼용 | 적합도: 지정 범주확률 동일; 독립성: 두 범주변수 연관 없음 |
| 6쪽 Kruskal–Wallis | paired로 표기 | 독립 집단 검정. 대응 여러 조건은 Friedman |
| 5~7쪽 Wilcoxon/Mann–Whitney/Kruskal/Friedman | 순위 검정을 모두 단순 중앙값 검정처럼 서술 | 기법별 분포/대칭성/순위 귀무가설 구분. 중앙값 해석의 추가 가정 설명 |
| 8쪽 OR/RR | 독립 설계를 설명하면서 전후 예시 사용 | 독립 2×2와 같은 사람의 대응 자료를 구분. 사건·노출 방향을 고정 |
| 8쪽 Fisher | 기대도수 경험칙과 일반 RxC를 지원 보장처럼 설명 | 소표본 경험칙은 사용 참고. 2×2와 RxC 방법·복잡도·버전 경계를 명시 |
| 9쪽 CMH | 층간 차이 검정처럼 설명 | 층을 통제한 연관성/공통 OR=1 검정. OR의 층간 이질성 검정은 별도 |
| 9쪽 Bowker | 주변 비율 동일성만을 H0로 서술 | H0는 pij=pji인 대칭성. 주변동질성과 동일한 검정이 아님 |
| 10쪽 Cochran Q | Bowker 확장·일반 다범주 자료로 설명 | McNemar의 여러 대응 이진 조건 확장. subjects×conditions 0/1 자료 |
| 10쪽 KS | 정규성 검정이라고만 표현 | 지정 연속 CDF 적합도 검정. 자료에서 모수를 추정한 경우 보정 필요 |
| 10쪽 ADF | 단위근과 모든 비정상성을 동일시 | H0는 단위근. 기각이 모든 형태의 정상성 증명은 아님 |

## 8. 구현 명세의 공통 계약

- 기존 파일의 작성 정보·메타데이터·분석 가정·변경 이력과 7개 섹션 유지.
- 각 기법의 Profile 필드에 타입·기본값·범위·설계 가정과 exact/근사 선택을 문서화.
- 기본 JSON 레시피와 사용자 recipe_path/overrides 우선순위를 기존과 동일하게 적용.
- 반환은 전용 dataclass Output DTO, to_dict/to_json, runtime 객체는 기본 JSON에서 제외.
- 검정 DTO 공통: test_type, statistic, p_value(미제공 시 None), alpha, reject_h0, null_hypothesis, alternative, method, 표본수/자유도/경고; 실제 제공 가능 여부를 기법별 정의.
- AD처럼 임계값을 사용하는 검정은 decision_basis/critical_values/significance_levels를 포함해 p 기반 여부를 구분. 지원하지 않는 alpha를 임의 보간하거나 p를 만들어내지 않음.
- OR/RR은 estimate·null_value=1·SE·CI·방향 정의, 비율은 count/n·point estimate·CI 방법을 명세.
- paired/block 자료는 subject ID 또는 명시적인 원 행 대응을 보존. 무작위 독립 표본 검정으로 대체하지 않음.
- NaN 정책, 0차이·동률, 희소 도수·0셀, 상수 표본, 특이 공분산, 부족한 n은 명시적인 검증/경고/오류로 처리.
- 정확 검정과 Monte Carlo 근사는 결과에서 구분. 일반 RxC 및 재표본 기반 추론은 계산량 제한·시드·반복 수를 노출.
- Python 3.7 고정 라이브러리와 Python 3.12의 인자/결과 차이를 격리하고 두 환경에서 참조값 테스트.
- 기존 기능은 재구현하지 않고 공개 계약을 확장하거나 공통 계산기를 재사용.

## 9. 권장 구현 순서

1. 분포 16종·범위/모수/확률 정의와 오류 교정, 단일 분산 χ²·모비율·대응 t.
2. Sign·Wilcoxon·Runs·Mann–Whitney·Kruskal·Mood·Friedman·반복측정 ANOVA.
3. OR·RR·Fisher(2×2와 RxC)·CMH·Bowker·Cochran Q·독립 동질성 API.
4. AD·KS/모수 추정 보정·ADF·Hotelling T²·Box M.
5. 구형성/특이성/희소표/분포 정의역의 경계 검증, PDF 항목 전수 대응표·README·기존 Z/t 표준화.

이 순서는 의존성과 공통 입력 구조에 따른 제안이며 아직 구현 승인이나 착수로 해석하지 않는다. Feature 전처리 설계와 통계 검정 설계를 섞지 않고 별도 도메인으로 유지한다.

## 10. 근거와 확인 범위

1. Statistics.pdf 1~10쪽 전체: 분포 #1~#16, 검정 본문 #1~#28, 기타 #1~#6, 4쪽 분류표 및 분산 설명, 8~9쪽 이미지 표·수식.
2. 저장소 main 기준 src/stats/tests 전체 실행 함수와 Profile/DTO, src/Recipes/loader.py, docs/ANALYSIS_CATALOG.md 및 지원 조건 대조.
3. 공식 API 문서로 원문 교정 및 구현 경계 확인:
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
   - https://docs.scipy.org/doc/scipy-1.14.0/tutorial/stats/discrete_nbinom.html
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
   - https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.cochrans_q.html
   - https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.SquareTable.symmetry.html
   - https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds.html
   - https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.lilliefors.html
   - https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html
   - https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html

문서 조사로 파악한 기능 차이이며, 미구현 기능의 동작 검증을 마친 결과는 아니다. 공식 문서 최신 버전 기능이 Python 3.7의 고정 라이브러리에 존재한다고 가정하지 않는다. 실제 구현 시 버전별 실행 검증이 필요하다.

## 11. PDF 4쪽 분류표를 반영하는 폴더 이관 전략

PDF 표의 축은 표본 관계(one-sample / independent / paired), 조건 수(1 / 2 / k), 척도(ordinal / nominal / interval·ratio), 검정 계열(parametric / nonparametric)이다. 모든 축을 폴더에 중첩하면 같은 기법이 중복된다. 예를 들어 Sign/Wilcoxon은 단일·대응 표본 모두를, Pearson 범주 검정은 2개·k개 집단 모두를 지원한다.

따라서 **폴더는 검정 계열과 분석 목적, Profile/registry는 표본 관계와 척도**를 담당한다. 표는 코드의 중복 복사 기준이 아니라 지원 범위 명세의 기준으로 사용한다.

| 권장 경로 (src/stats 기준) | 기법 | 표의 설계 구분 |
| --- | --- | --- |
| distributions/*.py | 16종 확률분포 | 표본 검정과 별도, 분포 파라미터와 확률 연산 |
| tests/parametric/Average/one_sample.py | 단일 평균 Z/t | one-sample, interval/ratio |
| tests/parametric/Average/two_sample.py | 독립 두 평균 Z/Student/Welch t | independent 2, interval/ratio |
| tests/parametric/Average/paired.py | 대응 t | paired 2, interval/ratio |
| tests/parametric/Average/anova/one_way.py | 고전/Welch 일원 ANOVA | independent k |
| tests/parametric/Average/anova/two_way.py | 이원 ANOVA | 독립 관측, 2개 요인 |
| tests/parametric/Average/anova/repeated_measures.py | 일원 반복측정 ANOVA | paired k |
| tests/parametric/Average/anova/posthoc.py | 기존 독립 집단 Tukey/Holm/Bonferroni | independent k 이후 쌍별 비교 |
| tests/parametric/Proportion/{one_sample,two_sample}.py | 모비율 | one-sample / independent 2, binary counts; 정확/근사는 method로 구별 |
| tests/parametric/Variance/{one_sample,f_test}.py | 단일 분산 χ² / 두 분산 F | one-sample / independent 2 |
| tests/parametric/Multivariate/{manova,hotelling}.py | MANOVA / Hotelling T² | 복수 수치 반응변수의 평균벡터 검정 |
| tests/nonParametric/Location/{sign,wilcoxon}.py | 부호·부호순위 | one-sample / paired 2, ordinal 또는 수치 |
| tests/nonParametric/Location/{mann_whitney,kruskal_wallis,mood_median}.py | 독립 순위·중앙값 비교 | independent 2 / k |
| tests/nonParametric/Location/friedman.py | Friedman | paired k, 순위 |
| tests/nonParametric/Categorical/{chi_square,independence,homogeneity,fisher_exact}.py | 적합도·독립성·독립 동질성·정확 검정 | one-sample / independent 2 / k, nominal |
| tests/nonParametric/Categorical/{mcnemar,bowker,cochran_q}.py | 대응 범주 검정 | paired 2 / k; 이진·다범주 구분 |
| tests/nonParametric/Sequence/runs.py | Runs | 순서가 있는 표본 |
| tests/diagnostics/{normality,anderson_darling,kolmogorov_smirnov,homogeneity,box_m}.py | 분포·분산·공분산 가정 진단 | 대상 자료와 가정별 구별 |
| tests/Association/{odds_ratio,risk_ratio,cmh}.py | OR/RR/층화 연관성 | 독립 2×2 또는 층별 2×2 |
| tests/TimeSeries/adf.py | 단위근 진단 | 순서·lag·결정항 |

현재 nonParametric 등의 대소문자 이름을 이 단계에서 일괄 변경하지 않는다. src/Association/apriori.py는 기존 데이터마이닝 연관규칙 도메인으로 유지하고, 통계적 OR/RR/CMH는 src/stats/tests/Association에서 관리한다. diagnostics/homogeneity(분산)와 Categorical/homogeneity(범주분포)는 파일 내용·API 이름·registry key를 명확히 구분한다.

### 11.1 기존 파일의 이관과 호환 경로

| 기존 파일 (src/stats/tests 기준) | 새 정식 위치 | 이관 방식 |
| --- | --- | --- |
| parametric/Average/Univariate.py | parametric/Average/one_sample.py | 계산 이관·공개 runner/DTO 정비, 기존 파일은 호환 export |
| parametric/Average/Bivariate.py | parametric/Average/two_sample.py | Student/Welch/Z 이관, 기존 Profile/Result/내부 함수 호환 유지 |
| parametric/Average/Multivariate/OneWay.py | parametric/Average/anova/one_way.py | 단일 반응변수 ANOVA를 평균 비교 폴더로 이동 |
| parametric/Average/Multivariate/TwoWay.py | parametric/Average/anova/two_way.py | 같은 방식; 2개 요인과 복수 반응변수 구분 |
| parametric/Average/Multivariate/PostHoc.py | parametric/Average/anova/posthoc.py | 기존 독립 집단 사후검정 범위 유지 |
| parametric/Average/Multivariate/MANOVA.py | parametric/Multivariate/manova.py | 실제 다변량 반응 검정을 별도 영역으로 이동 |
| nonParametric/chi_square.py | nonParametric/Categorical/chi_square.py | 기존 API/DTO/class 호환 export |
| nonParametric/independence.py | nonParametric/Categorical/independence.py | 기존 API 유지, 독립 범주 동질성 wrapper는 별도 |
| nonParametric/mcnemar.py | nonParametric/Categorical/mcnemar.py | 같은 방식 |
| parametric/Variance/f_test.py | 동일 위치 | 재구현·이동 없음 |
| diagnostics/normality.py, diagnostics/homogeneity.py | 동일 위치 | 기존 method와 동작 유지 |
| nonParametric/onesample.py, twosample.py | 구현 경로로 사용하지 않음 | 빈 placeholder로 명시; 구현 기능으로 집계하지 않음 |

현재 Average/Multivariate에 있는 OneWay/TwoWay/PostHoc는 여러 집단을 비교하지만 반응변수는 하나이다. 여러 집단(k samples), 여러 요인(factors), 여러 반응변수(multivariate responses)를 폴더/문서에서 혼동하지 않도록 바로잡는다.

순서: 새 정식 모듈 생성 → 기존 모듈의 얇은 호환 export → recipe registry의 import를 새 위치로 전환 → 문서/CI는 새 위치 예제 사용 → 기존 위치와 새 위치가 같은 클래스·결과를 반환하는지 검증. 기존 파일을 즉시 삭제하거나 기존 함수 이름을 바꾸지 않는다. 기존 python -m 예제 경로도 유지한다.

### 11.2 표본 관계는 명시적인 입력 계약

모든 테스트에서 sample_design 속성을 임의로 통일하는 것보다는, 해당 기법이 지원하는 설계만 정의한다. 단일·대응을 모두 지원하는 Sign/Wilcoxon Profile은 sample_design을 제공하고, paired t는 명시적인 paired API로 제한한다. registry 메타데이터 예:

```python
# 설계 예시. 아직 구현된 registry가 아닙니다.
TestCapability(
    key="wilcoxon",
    sample_designs=("one_sample", "paired"),
    measurement_levels=("ordinal", "continuous"),
    group_counts=(1, 2),
    exact_supported=True,
)
```

이 메타데이터는 가능한 기법 안내와 입력 검증용이다. 척도/표본 수만으로 최적 검정을 자동 결정하지 않는다. 독립성, 대응 여부, 정규성/대칭성, 원하는 귀무가설은 추가 조건이다.

## 12. 구현 후 사용할 API 예시

아래 이름과 세부 필드는 설계 제안이며 현재 import하여 실행할 수 있는 코드가 아니다. 사용자는 기법을 정의하지 않고 이미 정의된 모듈을 불러와 자료와 Profile만 전달한다. 모든 기법의 공개 API는 run_* + 전용 Profile/OutputDTO + 기본 JSON/overrides라는 동일한 형태를 따른다.

### 12.1 대응 t: 자료·조건 입력 → 전용 결과

```python
from src.stats.tests.parametric.Average.paired import (
    PairedTProfile,
    run_paired_t,
)

before = [120, 135, 128, 142, 130]
after = [115, 130, 126, 134, 127]

result = run_paired_t(
    before,
    after,
    profile=PairedTProfile(
        diff_0=0.0,
        alternative="greater",
        alpha=0.05,
    ),
)
# 差의 정의: before - after. greater는 이 차이의 평균 > 0.
print(result.mean_difference)
print(result.statistic, result.df, result.p_value, result.reject_h0)
print(result.confidence_interval)  # 단측이면 비유계 끝점 정책 명세
print(result.to_json(indent=2))
```

동일 기법의 레시피 호출:

```python
result = run_paired_t(
    before,
    after,
    overrides={"alternative": "greater"},
)
# Profile 생략 → src/Recipes/paired-t.json → overrides 적용.
```

레시피 제안:

```json
{
  "diff_0": 0.0,
  "alternative": "two-sided",
  "alpha": 0.05,
  "nan_policy": "raise"
}
```

명시적 Profile과 overrides/recipe_path는 기존 규칙대로 동시에 지정하지 않는다.

### 12.2 부호순위: 단일·대응을 같은 모듈에서 구분

```python
from src.stats.tests.nonParametric.Location.wilcoxon import (
    WilcoxonProfile,
    run_wilcoxon,
)

result = run_wilcoxon(
    before,
    after,
    profile=WilcoxonProfile(
        sample_design="paired",
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    ),
)
print(result.statistic, result.p_value)
print(result.method_used, result.n_effective, result.zero_count)
```

단일 표본은 y 없이 sample_design="one_sample"와 기준 location_0를 명시한다. exact 사용 가능 조건·동률/0차이·버전별 fallback은 결과에 기록한다. 기법 정의나 통계 수식을 사용자가 다시 작성하지 않는다.

### 12.3 확률분포: 밀도·누적·상측·구간 확률을 별도 조회

```python
from src.stats.distributions.normal import NormalProfile, run_normal

result = run_normal(
    profile=NormalProfile(mu=70.0, sigma=10.0),
    x=[70.0, 85.0],
    interval=(70.0, 85.0),
)
print(result.pdf)                   # 점에서의 밀도
print(result.cdf)                   # P(X <= x)
print(result.survival)              # P(X > x)
print(result.interval_probability)  # P(70 <= X <= 85)
print(result.mean, result.variance)
```

CDF/SF와 PDF의 의미를 섞지 않으며, 이산 분포는 pdf 대신 pmf를 제공한다. 분포 조회 인자(x/interval)와 분포 모수 Profile을 분리해 재사용할 수 있도록 한다.

## 13. 착수 전 합의할 범위

1. 위 폴더 구조와 기존 import 호환 이관 정책.
2. PDF 전체 범위: 16종 분포, 23개 신규 기능, 동질성 API 의미 확장, 기존 Z/t 표준화.
3. Fisher RxC를 PDF 범위로 포함하는 계산 모드와 반복측정 ANOVA의 최초 지원 범위.
4. 모든 기법의 Profile/OutputDTO/개별 JSON/7개 섹션/참조값 테스트와 Python 3.7·3.12 검증.

이 문서는 구현 전략과 사용 API의 제안이다. 승인 전에는 저장소의 코드·경로·기존 API를 변경하지 않는다.
