# `src/stats/manova.py`

[소스 코드](../../src/stats/manova.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
manova.py - 다변량 반응의 요인별 MANOVA
================================================================================
설명:
    - 두 개 이상 수치 반응에 대해 Wilks/Pillai/Hotelling-Lawley/Roy 검정표를 제공합니다.
    - 요인 및 선택적 교호작용을 Sum 대비 수식으로 검정하며 rank 결손을 사전에 거부합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-11
    - 버전: 1.0.0
변경 이력: 2026-10-11 v1.0.0 Profile/Result·mv_test 및 rank 검증 추가
사용 예제: python -m examples.multivariate_workflow
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `MANOVAProfile`

MANOVA 조건.

Attributes:
    responses: 두 개 이상의 서로 다른 수치 반응 열 이름.
    factors: 하나 이상의 범주 요인 열 이름. 각 요인 최소 두 수준.
    interaction: True이면 모든 요인의 교호작용을 포함, False이면 주효과만.
    alpha: 각 요인/통계량의 유의수준 (0,1). 요인 간 다중검정 보정은 하지 않음.

## `MANOVAOutputDTO`

요인별 다변량 검정 결과.

Attributes:
    model: statsmodels MANOVA 객체; fit() 대신 mv_test() 사용.
    multivariate_test: statsmodels MultivariateTestResults 원 객체.
    tables: 요인명/교호작용명 -> 네 통계량의 Value/Num DF/Den DF/F Value/Pr > F/reject_h0 표.
    responses: 입력 반응 열 순서.
    factors: 입력 요인 열 순서.
    residual_df: 설계행렬에 따른 잔차 자유도.

## `run_manova`

독립 관측의 여러 반응을 함께 검정합니다.

Args:
    data: 반응과 범주 요인의 long-form DataFrame. 완전 관측만 허용.
    profile: 반응 목록·요인 목록·교호작용·유의수준.
Returns:
    MANOVAOutputDTO: 요인/교호작용별 네 통계량의 검정표. 절편 검정은 tables에서 제외.
Raises:
    ValueError: 열/수준/표본 부족, rank 결손 설계, 특이 잔차 공분산 또는 비유한 검정표.
    ImportError: statsmodels/patsy 미설치.
Notes:
    독립 관측·집단 내 다변량 정규성과 공분산 동질성을 가정합니다. 반복측정 MANOVA나
    강건 MANOVA는 아닙니다. 네 통계량 중 무엇을 보고할지는 분석 전에 정합니다.
    개별 반응의 사후검정을 여러 번 실행할 때 반응 간 추가 다중성도 고려해야 합니다.
