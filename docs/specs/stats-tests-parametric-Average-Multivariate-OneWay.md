# `src/stats/tests/parametric/Average/Multivariate/OneWay.py`

[소스 코드](../../src/stats/tests/parametric/Average/Multivariate/OneWay.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
OneWay.py - 세 집단 이상의 평균 검정(One-way ANOVA) 기능 구현
================================================================================
설명:
    - 세 집단 이상의 평균 검정을 수행하는 기능을 제공합니다.
    - 고전적 ANOVA 및 Welch ANOVA 수행 시 ANOVA 분산분석표(pandas DataFrame)를 생성합니다.

작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력:
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)       v1.0.0    최초 작성
    2026-09-10  김재현(Finn)       v1.1.0    ANOVA 분산분석표(anova_table) 출력 기능 추가
    2026-10-10  김재현(Finn)       v1.2.0    유의수준·유한 표본·퇴화 분산 검증 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.3.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `OneWayANOVAProfile`

세 집단 이상의 평균 검정에 필요한 모집단 사전 정보 및 조건을 정의하는 데이터 클래스.

Attributes:
    equal_var: 등분산 가정 여부 (True: Classic ANOVA, False: Welch's ANOVA)
    alpha: 유의수준 (기본값 0.05)

## `OneWayANOVAOutputDTO`

세 집단 이상의 평균 검정 결과를 통일된 규격으로 반환하는 데이터 클래스.

Attributes:
    test_type: 수행된 검정 유형 (예: "One-way ANOVA (Equal Variance)", "Welch's One-way ANOVA")
    statistic: 검정 통계량 값 (F 값)
    p_value: p-value 값
    reject_h0: 귀무가설 기각 여부 (True/False)
    num_groups: 집단 수
    group_means: 각 집단의 평균 리스트
    group_stds: 각 집단의 표준편차 리스트
    group_ns: 각 집단의 표본 크기 리스트
    df_between: 집단 간 자유도 (df1)
    df_within: 집단 내 자유도 (df2)
    anova_table: 전체 ANOVA 분산분석표 (pandas DataFrame)

## `run_oneway_anova`

세 집단 이상의 평균 차이 검정(One-way ANOVA)을 유효성 검증 후 적절한 알고리즘으로 수행합니다.

Args:
    groups: 각 집단의 데이터 리스트/배열 모음 (최소 2개 이상의 집단 시퀀스).
    profile: 검정 조건을 정의한 OneWayANOVAProfile 객체.

Returns:
    OneWayANOVAOutputDTO: 검정 결과 및 ANOVA 테이블을 통일된 규격으로 담은 객체.

Raises:
    ValueError: 집단 수가 2개 미만이거나, 특정 집단의 표본 크기가 2 미만인 경우.
    ValueError: profile이 None이거나, alpha가 (0, 1) 범위 밖인 경우.
