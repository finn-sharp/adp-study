# `src/output.py`

[소스 코드](../../src/output.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
output.py - 분석 Output DTO의 공통 변환 및 필드 접근 계약
================================================================================
설명:
    - 분석 결과 dataclass에 결과 사전/JSON 변환과 기존 키 접근 호환성을 제공합니다.
    - 모델/그래프 등 실행 객체는 필드에 유지하고 기본 저장용 사전에서는 명시적으로 제외합니다.
작성 정보: 작성자 김재현(Finn), 작성일 2026-10-11, 버전 1.0.0
변경 이력: 2026-10-11 v1.0.0 모든 분석 모듈의 Output DTO 공통 계약 추가
사용 예제: result.to_dict(); result.to_json(); result.predictions
================================================================================

## `OutputDTO`

모든 분석 결과 dataclass의 공통 기반. 클래스 자체는 분석 반환 객체가 아닙니다.

Attributes:
    runtime_fields: JSON/기본 사전에서 제외할 실행 객체 필드 이름.
        원 객체는 result.model 등으로 직접 접근하거나 include_runtime=True로 조회합니다.

### `to_dict`

DTO 필드를 사전으로 변환합니다. 원본 배열·모델을 deep copy하지 않습니다.

Args:
    include_runtime: True이면 실행 객체도 포함. JSON 변환에는 False만 허용.
    json_safe: True이면 배열/표/날짜/집합/희소행렬을 JSON용 구조로 변환.
Returns:
    Dict[str, Any]: native 필드 또는 JSON 호환 구조. 표는 index/columns/data를 보존.
Raises:
    TypeError: dataclass가 아닌 객체 또는 지원하지 않는 JSON 값.
    ValueError: JSON에 runtime 포함 요청 또는 문자열 사전 키 충돌.
Notes:
    NaN/무한대는 JSON null. 희소행렬은 COO 좌표로 저장해 dense 확장을 피합니다.
    JSON은 모델 저장/복원 형식이 아니며 포함되지 않은 runtime 객체는 복원하지 못합니다.

### `to_json`

실행 객체를 제외한 분석 데이터의 엄격한 JSON 문자열을 반환합니다.

Args:
    indent: 선택적 들여쓰기 크기.
Returns:
    str: UTF-8 의미의 유니코드 JSON. NaN 토큰을 허용하지 않음.
Raises:
    TypeError/ValueError: to_dict(json_safe=True)의 변환 오류.

### `from_result`

같은 결과 필드를 가진 공통 DTO를 모듈 전용 DTO로 변환합니다.

Args:
    result: 필요한 필드를 가진 결과 dataclass.
Returns:
    OutputDTO: 현재 하위 클래스 인스턴스. 실행 객체는 동일 참조 유지.
Raises:
    AttributeError: 필요한 원본 필드 누락.

### `keys`

실행 객체를 포함한 전체 필드 이름을 반환합니다.

### `items`

실행 객체를 포함한 필드 이름/값 쌍을 반환합니다.

### `get`

기존 사전의 get 접근을 지원합니다.
