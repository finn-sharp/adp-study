# `src/stats/analysis.py`

[소스 코드](../../src/stats/analysis.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
analysis.py - 기술통계, 정규성·등분산·상관·범주형·분산 검정
================================================================================
설명:
    - 통계분석 및 카이제곱/F 분포 PDF의 절차를 수치형 실행기로 구현합니다.
    - reject_h0=False는 귀무가설이 참이라는 뜻이 아니라 기각 근거 부족입니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 통계 Profile/Result와 입력 검증 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `StatisticalProfile`

통계 검정 조건.

Attributes:
    test: shapiro, normaltest, levene, bartlett, pearson, spearman, kendall,
        paired_t, wilcoxon, mannwhitney, kruskal, friedman,
        chi2_independence, chi2_goodness, variance_chi2, variance_f.
    alpha: 유의수준 (0 < alpha < 1).
    alternative: two-sided, less, greater. 방향 지원 검정에만 사용.
    null_value: variance_chi2는 귀무 모분산, variance_f는 귀무 분산비.
    correction: chi2_independence에서 2x2 Yates 보정 여부.
    center: Levene 중심값 (median, mean, trimmed).

## `StatisticalOutputDTO`

통계 검정 결과.

Attributes:
    test_type: 수행 검정 이름.
    statistic: 검정통계량 (상관 검정은 상관계수).
    p_value: 지정한 대립가설의 p-value.
    reject_h0: p_value < alpha 여부. NaN 결과는 반환하지 않음.
    details: 표본 수, 자유도, 기대도수, 귀무가설 등 검정별 보조 정보.

## `DescriptiveOutputDTO`

기술통계 반환 DTO.

Attributes:
    n: 표본 수.
    mean, median, modes: 평균·중앙값·최빈값 배열.
    sample_variance, sample_std: ddof=1 표본 분산/표준편차.
    q1, q3, iqr, quartile_deviation: 사분위수·IQR·반사분위수.
    range: 최대값과 최소값 차이.
    skewness, excess_kurtosis: 왜도·초과첨도. 정의되지 않으면 None/NaN.
    outlier_indices: 원본 표본 순서의 IQR 이상치 위치.

## `numeric_sample`

유한한 1차원 수치 표본을 검증합니다.

Args:
    data: 표본 시퀀스.
    min_size: 최소 표본 수.
Returns:
    np.ndarray: 1차원 float 표본.
Raises:
    ValueError: 차원/크기 오류 또는 NaN·무한대.

## `validate_test_profile`

공통 유의수준과 검정 방향을 검증합니다.

Args:
    profile: alpha 및 선택적 alternative 필드가 있는 Profile.
Returns:
    None: 유효한 경우 정상 종료.
Raises:
    ValueError: Profile 누락, 잘못된 alpha 또는 alternative.

## `run_descriptive`

기술통계와 IQR 이상치 위치를 반환합니다.

Args:
    data: 최소 2개의 유한한 1차원 수치 표본.
    iqr_weight: 이상치 경계의 IQR 배수 (0 이상).
Returns:
    DescriptiveOutputDTO: n/mean/median/modes/sample_variance/sample_std/q1/q3/iqr/
        quartile_deviation/range/skewness/excess_kurtosis/outlier_indices.
        상수 표본의 왜도·첨도는 정의되지 않으므로 None.
Raises:
    ValueError: 잘못된 표본 또는 iqr_weight.

## `run_statistical_test`

검정 유형에 따라 표본·집단·도수 입력을 처리합니다.

Args:
    data: 단일 검정은 1차원 표본; 두 표본 검정은 첫 표본; 집단 검정은 표본 목록;
        chi2_independence는 2차원 교차표, chi2_goodness는 1차원 관측도수.
    profile: 검정 이름과 조건. None이면 Shapiro 정규성 검정.
    other: 두 표본·상관·대응 검정의 두 번째 표본.
    expected: chi2_goodness의 기대도수 (관측도수와 합이 같아야 함). None이면 균등 도수.
Returns:
    StatisticalOutputDTO: 유한한 통계량·p-value와 보조 정보.
Raises:
    ValueError: 잘못된 입력/프로필, 상수·퇴화 표본 또는 정의 불가능한 검정.

Notes:
    paired_t/wilcoxon/friedman 입력은 같은 관측 대상 순서로 정렬되어야 합니다.
    상관 검정은 scipy 1.7 호환성을 위해 양측만 지원합니다.
    분산 검정은 독립성과 정규성을 가정합니다. 기대도수가 작은 교차표는 해석에 주의합니다.
