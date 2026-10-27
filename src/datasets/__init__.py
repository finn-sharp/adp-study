"""
================================================================================
__init__.py - 분야별 UCI 로더와 로컬 다중 형식 데이터 로딩 API.
================================================================================
설명:
    - 분야별 UCI 로더와 로컬 다중 형식 데이터 로딩 API.
분석 가정:
    - 패키지 초기화와 공개 가져오기 경로를 정의합니다. 실제 계산의 입력 조건은 하위 모듈 명세를 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__all__ = [
    "TabularLoaderProfile",
    "TabularLoaderOutputDTO",
    "run_tabular_loader",
    "read_table",
    "PersistenceProfile",
    "PersistenceOutputDTO",
    "save_artifact",
    "load_artifact",
    "save",
    "load",
]


def __getattr__(name):
    """공개 API를 지연 로드하여 모듈 실행과 선택 의존성 분리를 유지합니다."""
    if name in __all__:
        from importlib import import_module

        module = (
            "tabular"
            if name
            in (
                "TabularLoaderProfile",
                "TabularLoaderOutputDTO",
                "run_tabular_loader",
                "read_table",
            )
            else "persistence"
        )
        return getattr(import_module("." + module, __name__), name)
    raise AttributeError(name)
