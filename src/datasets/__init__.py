"""분야별 UCI 로더와 로컬 다중 형식 데이터 로딩 API."""

__all__ = [
    "TabularLoaderProfile",
    "TabularLoaderOutputDTO",
    "run_tabular_loader",
    "read_table",
]


def __getattr__(name):
    """공개 API를 지연 로드하여 모듈 실행과 선택 의존성 분리를 유지합니다."""
    if name in __all__:
        from . import tabular

        return getattr(tabular, name)
    raise AttributeError(name)
