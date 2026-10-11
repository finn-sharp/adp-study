# `src/stats/tests/parametric/Average/Bivariate.py`

[소스 코드](../../src/stats/tests/parametric/Average/Bivariate.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
Bivariate.py - 두 변수 간의 평균 검정(Bivariate Average Test) 기능 구현
================================================================================
설명:
    - 모집단의 정보와 두 표본의 정보를 기반으로 두 표본 평균 검정을 수행하는 기능을 제공합니다.

작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)       v1.0.0    두 표본 평균 검정(Two-sample Average Test) 기능 구현
    2026-10-10  김재현(Finn)       v1.1.0    공개 실행기와 입력 검증 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.2.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `TwoSampleTestProfile`

두 집단 검정에 필요한 모집단 사전 정보 및 조건을 정의하는 데이터 클래스.

Attributes:
    diff_0: 귀무가설 모평균 차이 (H0: μ1 - μ2 = diff_0, 기본값 0.0)
    known_var: 알려진 모분산(모표준편차) 유무
    sigma1: 집단 1의 알려진 모표준편차 (known_var=True 일 때 필수)
    sigma2: 집단 2의 알려진 모표준편차 (known_var=True 일 때 필수)
    equal_var: 모분산을 모를 때 등분산 가정 여부 (True: Student T-test, False: Welch T-test)
    alpha: 유의수준 (기본값 0.05)
    alternative: 대립가설 방향 ('two-sided', 'less', 'greater')

## `TwoSampleTestOutputDTO`

두 집단 검정 결과를 통일된 규격으로 반환하는 데이터 클래스.

Attributes:
    test_type: 수행된 검정 유형 (예: "Two-sample Z-test", "Welch's T-test")
    statistic: 검정 통계량 값
    p_value: p-value 값
    reject_h0: 귀무가설 기각 여부 (True/False)
    diff_mean: 두 표본의 평균 차이 (mean1 - mean2)
    sample_mean1: 집단 1 표본 평균
    sample_mean2: 집단 2 표본 평균
    sample_std1: 집단 1 표본 표준편차
    sample_std2: 집단 2 표본 표준편차
    n1: 집단 1 표본 크기
    n2: 집단 2 표본 크기

## `run_two_sample_test`

독립 두 표본 평균 검정을 검증 후 실행합니다.

Args:
    data1: 첫 번째 유한한 1차원 표본 (최소 2개).
    data2: 두 번째 유한한 1차원 표본 (최소 2개).
    profile: 귀무 평균 차이, 모표준편차, 등분산 여부 및 대립가설.
Returns:
    TwoSampleTestOutputDTO: Z/Student/Welch 검정과 표본 통계.
Raises:
    ValueError: 잘못된 입력·조건 또는 두 집단 모두 0 표본분산인 T 검정.

Notes:
    대응 표본은 src.stats.analysis의 paired_t 검정을 사용합니다.
