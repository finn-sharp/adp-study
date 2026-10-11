# `src/stats/tests/parametric/Average/Multivariate/TwoWay.py`

[소스 코드](../../src/stats/tests/parametric/Average/Multivariate/TwoWay.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
TwoWay.py - 두 요인에 대한 평균 검정(Two-way ANOVA) 기능 구현
================================================================================
설명:
    - 두 개의 범주형 독립변수(요인)에 따른 수치형 반응변수의 평균 차이를 검정합니다.
    - 교호작용항(Interaction Effect) 포함 여부 및 제곱합 타입(Type I, II, III) 선택 기능을 제공합니다.

작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-09-10
    - 라이선스: Proprietary

변경 이력:
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-09-10  김재현(Finn)       v1.0.0    최초 작성
    2026-10-10  김재현(Finn)       v1.1.0    Type III 행 추출·Sum 대비 및 설계 검증 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.2.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `TwoWayANOVAProfile`

이원배치 분산분석(Two-way ANOVA) 사전 정보 및 검정 조건을 정의하는 데이터 클래스.

Attributes:
    interaction: 교호작용항(Factor1 x Factor2) 포함 여부 (기본값 True)
    ss_type: 제곱합(Sum of Squares) 계산 방식 (Type 1, 2, 3 / 기본값 2)
    alpha: 유의수준 (기본값 0.05)

## `TwoWayANOVAOutputDTO`

Two-way ANOVA 검정 결과를 통일된 규격으로 반환하는 데이터 클래스.

Attributes:
    test_type: 수행된 검정 유형 (예: "Two-way ANOVA (With Interaction)", "Two-way ANOVA (Main Effects Only)")
    factor1_name: 요인 1 변수명
    factor2_name: 요인 2 변수명
    f_factor1: 요인 1 F 통계량
    p_factor1: 요인 1 p-value
    reject_h0_factor1: 요인 1 귀무가설 기각 여부
    f_factor2: 요인 2 F 통계량
    p_factor2: 요인 2 p-value
    reject_h0_factor2: 요인 2 귀무가설 기각 여부
    f_interaction: 교호작용항 F 통계량 (interaction=False 시 None)
    p_interaction: 교호작용항 p-value (interaction=False 시 None)
    reject_h0_interaction: 교호작용항 귀무가설 기각 여부 (interaction=False 시 None)
    anova_table: 전체 ANOVA 분산분석표 (pandas DataFrame)

## `run_twoway_anova`

두 요인에 대한 평균 차이 및 교호작용 검정(Two-way ANOVA)을 유효성 검증 후 적절한 알고리즘으로 수행합니다.

Args:
    y: 반응 변수 데이터 (수치형 데이터 시퀀스).
    factor1: 첫 번째 독립변수 요인 데이터 (범주형 데이터 시퀀스).
    factor2: 두 번째 독립변수 요인 데이터 (범주형 데이터 시퀀스).
    profile: 검정 조건을 정의한 TwoWayANOVAProfile 객체.
    factor1_name: 첫 번째 요인의 변수명 (기본값 "Factor1").
    factor2_name: 두 번째 요인의 변수명 (기본값 "Factor2").
    response_name: 반응 변수의 변수명 (기본값 "Response").

Returns:
    TwoWayANOVAOutputDTO: 검정 결과를 통일된 규격으로 담은 객체.

Raises:
    ValueError: 입력 데이터 간 길이가 다르거나, 표본 크기가 부족한 경우.
