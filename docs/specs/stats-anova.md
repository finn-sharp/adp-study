# `src/stats/anova.py`

[소스 코드](../../src/stats/anova.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
anova.py - 일원·Welch·이원·반복측정 ANOVA 통합 공개 API
================================================================================
설명:
    - 기존 OneWay/TwoWay/RepeatedANOVA 구현을 재사용하며 long-form 입력을 통일합니다.
    - 새로운 계산식을 중복하지 않고 원래 Result 객체와 ANOVA 표를 반환합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-11
    - 버전: 1.0.0
변경 이력: 2026-10-11 v1.0.0 통합 실행기·입력 검증·가정 명세 추가
사용 예제: python -m examples.multivariate_workflow
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 모든 방식의 명시적 공통 반환 DTO 추가

## `ANOVAProfile`

long-form ANOVA 설정.

Attributes:
    response: 수치 반응 열 이름.
    factors: one_way는 한 요인, two_way는 두 요인, repeated는 대상 내 요인 목록.
    method: one_way, two_way, repeated 중 하나.
    equal_var: one_way에서 True=Classic, False=Welch. 다른 방법은 True만 허용.
    interaction: two_way 교호작용 포함 여부.
    ss_type: two_way 제곱합 타입 1/2/3. Type III는 Sum 대비 사용.
    subject: repeated 대상 식별 열 이름. 다른 방법에는 지정하지 않음.
    alpha: 유의수준 (0,1).

## `ANOVAOutputDTO`

모든 ANOVA 방식의 공통 반환 DTO.

Attributes:
    method: one_way/two_way/repeated.
    anova_table: 방법별 ANOVA DataFrame.
    analysis: 기존 방법별 Output DTO. statistic/p_value/f_factor1 등은 이 DTO에 정의.
Notes:
    이전의 result.statistic/result.f_factor1 접근은 analysis로 위임해 호환성을 유지합니다.

## `run_anova`

long-form 입력으로 일원/Welch/이원/반복측정 분산분석을 실행합니다.

Args:
    data: 반응·요인 및 선택적 subject 열을 가진 DataFrame. 행은 독립 관측 또는 명시된 반복 관측.
    profile: 방법/반응/요인/대상 및 검정 조건.
Returns:
    ANOVAOutputDTO: method, anova_table과 방법별 analysis DTO.
        일원 analysis.group_means 순서는 pd.unique 요인 순서.
Raises:
    ValueError: 방법/열/요인 수/결측·무한대/표본 부족/잘못된 설정.
    ImportError: statsmodels 미설치.
Notes:
    독립 ANOVA는 잔차 정규성·등분산을, Welch는 독립성과 집단별 분산 추정을 가정합니다.
    반복측정은 균형 설계만 지원하며 구형성 보정과 집단 간 요인을 제공하지 않습니다.
    검정은 가정을 자동으로 확정하지 않으며 사후검정은 posthoc.py에서 별도 호출합니다.
