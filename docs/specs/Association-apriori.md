# `src/Association/apriori.py`

[소스 코드](../../src/Association/apriori.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
apriori.py - 빈발 항목집합과 연관규칙
================================================================================
설명: 연관분석 PDF의 TransactionEncoder → Apriori → association_rules를 모듈화합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 거래 인코딩과 규칙 결과 명세 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `AssociationProfile`

Apriori/연관규칙 조건.

Attributes:
    min_support: 빈발 항목집합의 최소 지지도 (0 < 값 <= 1).
    metric: 규칙 필터 기준 confidence, lift, support.
    min_threshold: 기준 지표의 최소 값 (0 이상).
    max_len: 항목집합 최대 길이. None이면 제한 없음.

## `AssociationOutputDTO`

연관분석 결과.

Attributes:
    encoder: 적합 TransactionEncoder.
    encoded: 거래 x 항목 bool DataFrame. 한 거래의 중복 항목은 존재 여부로 처리.
    frequent_itemsets: support, itemsets를 가진 DataFrame.
    rules: antecedents/consequents/support/confidence/lift 등의 규칙 표.
        규칙이 없으면 빈 표. conviction은 완전 신뢰 규칙에서 inf일 수 있음.

## `run_association`

거래 목록으로 빈발 항목집합과 규칙을 추출합니다.

Args:
    transactions: 문자열 항목 목록들의 목록. 빈 거래는 전체 거래 수에 포함됩니다.
    profile: 연관분석 설정.
Returns:
    AssociationOutputDTO: 인코더, bool 거래표, 항목집합 및 규칙.
Raises:
    ValueError: 비어 있는 전체 입력, 비문자 항목, 잘못된 임계값·지표.
    ImportError: mlxtend 미설치.

Notes:
    연관규칙은 관측된 동시 발생 관계이며 인과관계를 의미하지 않습니다.
