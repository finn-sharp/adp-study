# `src/stats/tests/nonParametric/onesample.py`

[소스 코드](../../src/stats/tests/nonParametric/onesample.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
onesample.py - 단일 표본 Wilcoxon 부호순위 검정
================================================================================
설명: 통계분석의 비모수 검정 구조를 단일 표본 위치 검정으로 확장합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 단일 표본 비모수 검정 구현
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 모듈 전용 결과 DTO 및 저장용 변환 추가

## `OneSampleRankProfile`

단일 표본 검정 조건.

Attributes:
    mu_0: 귀무 위치. 차이 분포의 대칭성 가정하에 위치 차이를 검정합니다.
    alpha: 유의수준 (0, 1).
    alternative: two-sided, less, greater.

## `OneSampleRankOutputDTO`

OneSampleRank 모듈 전용 반환 DTO.

Attributes:
    공통 StatisticalOutputDTO의 검정 유형·통계량·p-value·보조 정보 필드를 상속합니다.
    to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.

## `run_one_sample_rank_test`

귀무 위치와의 차이로 Wilcoxon 검정을 수행합니다.

Args:
    data: 최소 2개의 유한한 1차원 수치 표본.
    profile: OneSampleRankProfile. None이면 귀무 위치 0.
Returns:
    OneSampleRankOutputDTO: 검정통계량, p-value, 기각 여부, 귀무 위치.
Raises:
    ValueError: 잘못된 조건 또는 모든 차이가 0인 표본.
