# `src/stats/tests/parametric/Average/Univariate.py`

[소스 코드](../../src/stats/tests/parametric/Average/Univariate.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
Univariate.py - 단일 표본 평균 검정(One-sample Average Test) 기능 구현
================================================================================
설명:
    - 모집단의 정보가 알려진 경우와 알려지지 않은 경우에 따라 단일 표본 평균 검정을 수행하는 기능을 제공합니다.

작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)       v1.0.0    단일 표본 평균 검정(One-sample Average Test) 기능 구현
    2026-10-10  김재현(Finn)       v1.1.0    공개 실행기와 입력 검증 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.2.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `TestProfile`

검정에 필요한 모집단 사전 정보 및 조건을 정의하는 데이터 클래스.

Attributes:
    mu_0: 귀무가설 모평균 (H0: μ = mu_0)
    known_var: 알려진 모분산(모표준편차) 유무
    sigma: 알려진 모표준편차 (known_var=True 일 때 지정)
    alpha: 유의수준 (0 < alpha < 1)
    alternative: 대립가설 방향 (two-sided, less, greater)

## `TestOutputDTO`

검정 결과를 통일된 규격으로 반환하는 데이터 클래스.

Attributes:
    test_type: 수행된 검정 유형 (예: "One-sample Z-test", "One-sample T-test")
    statistic: 검정 통계량 값
    p_value: p-value 값
    reject_h0: 귀무가설 기각 여부 (True/False)
    sample_mean: 표본 평균
    sample_std: 표본 표준편차
    n: 표본 크기

## `run_one_sample_test`

단일 표본 평균 검정을 검증 후 실행합니다.

Args:
    data: 최소 2개의 유한한 1차원 수치 표본.
    profile: 귀무 평균, 모표준편차 유무, alpha 및 대립가설 설정.
Returns:
    TestOutputDTO: Z/T 검정 결과 및 표본 통계.
Raises:
    ValueError: 잘못된 조건, 입력 또는 T 검정의 0 표본분산.
