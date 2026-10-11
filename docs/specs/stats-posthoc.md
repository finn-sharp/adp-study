# `src/stats/posthoc.py`

[소스 코드](../../src/stats/posthoc.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
posthoc.py - Tukey HSD·Games-Howell·보정된 쌍별 t 사후검정
================================================================================
설명:
    - 독립 집단의 모든 쌍을 비교하고 평균차(group2-group1), p-value 및 구간을 반환합니다.
    - 등분산은 Tukey, 이분산은 Games-Howell 또는 Welch 쌍별 t 검정을 지원합니다.
    - 쌍별 t는 Holm/Bonferroni 보정을 적용합니다. Games-Howell은 scipy studentized_range 사용.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-11
    - 버전: 1.0.0
변경 이력: 2026-10-11 v1.0.0 사후검정 3종·다중성·구간의 의미 명세 추가
사용 예제: python -m examples.multivariate_workflow
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `PostHocProfile`

독립 집단 사후검정 조건.

Attributes:
    method: tukey, games_howell, pairwise_t.
    equal_var: tukey는 True, games_howell은 False 필수. pairwise_t는 pooled/Welch 분기.
    correction: pairwise_t에서 holm 또는 bonferroni. 다른 방법은 자체 familywise 보정.
    alpha: 가족 단위 유의수준 (0,1).

## `PostHocOutputDTO`

모든 집단 쌍의 비교 결과.

Attributes:
    method: 실행 기법 이름.
    table: group1/group2/mean_diff/statistic/df/p_raw/p_adjusted/ci_lower/ci_upper/reject_h0.
        평균차는 group2-group1. Tukey statistic은 NaN, Games-Howell statistic은 양의 q,
        pairwise_t는 부호 있는 t. Holm의 구간은 계산하지 않아 NaN.
        Tukey/Games-Howell p_raw는 자체 보정된 검정이라 NaN.
    groups: 최초 등장 순서의 원래 집단 라벨 목록.
    alpha: 가족 단위 유의수준.
    correction: tukey, games_howell, holm 또는 bonferroni.

## `run_posthoc`

독립 집단 전체 쌍에 대해 사후검정과 다중성 보정을 수행합니다.

Args:
    y: 결측/무한대 없는 1차원 수치 반응.
    groups: 같은 길이의 결측 없는 이산 집단 라벨. 각 집단 최소 두 관측.
    profile: 기법·등분산·다중성·유의수준 설정. None이면 Tukey HSD.
Returns:
    PostHocOutputDTO: 원래 라벨의 모든 쌍 비교 DataFrame. reject_h0는 p_adjusted < alpha.
Raises:
    ValueError: 입력 길이/집단 수/표본 수/분산 부족 또는 기법·가정 불일치.
    ImportError: statsmodels 미설치. Games-Howell은 scipy >=1.7 필요.
Notes:
    Tukey HSD는 독립성·정규성·등분산을, Games-Howell은 독립성 및 집단별 정규성을
    가정합니다. 후자는 이분산·불균형을 허용합니다. Games-Howell 구간은 근사 simultaneous 구간.
    pairwise_t의 Bonferroni 구간은 가족 단위 보정 구간, Holm은 보정 p만 제공합니다.
    이 함수는 ANOVA 결과로 실행 여부를 자동 결정하지 않습니다. 사후 비교 계획을 명시하세요.
    반복측정/이원 ANOVA의 단순효과를 독립 일원 비교로 대체하지 않습니다.
