"""
================================================================================
persistence.py - Parquet 표와 joblib/pickle 분석 객체의 저장·복원. 전처리나 재학습을 하지 않습니다.
================================================================================
설명:
    - Parquet 표와 joblib/pickle 분석 객체의 저장·복원. 전처리나 재학습을 하지 않습니다.
분석 가정:
    - Parquet에는 DataFrame을 사용하며, pickle/joblib 복원은 신뢰한 파일에만 명시적으로 허용합니다.
    - 객체의 원래 클래스·라이브러리 환경 호환이 필요합니다. 전처리 적합·재학습은 수행하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

# 1. Imports
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
import os
import pickle
import tempfile
from typing import Any, Dict, List
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from ._readers import optional_package, validate_columns


# 2. Input Profile
@dataclass
class PersistenceProfile:
    """원본 표·모형 객체의 직렬화 설정. 환경 의존 객체의 버전 호환을 보장하지 않습니다.

    Attributes:
        format (str): auto/parquet/joblib/pickle. 기본 auto; 확장자로 선택.
        trusted (bool): pickle/joblib 읽기에 필수인 명시적 신뢰 확인. 기본False.
        overwrite (bool): 기존 로컬 목적 파일 교체 허용. 기본False.
        pickle_protocol (int): pickle/joblib 저장 프로토콜0..최대 지원값. 기본4; Python3.7 문법 지원.
        joblib_compress (int): joblib 압축 수준0..9. 기본3; mmap은 지원하지 않음.
        parquet_compression (Any): Parquet 압축 snappy/gzip/brotli/lz4/zstd/None. 기본snappy.
        parquet_index (bool): Parquet 저장 시 index 보존 여부. 기본True.
    """

    format: str = "auto"
    trusted: bool = False
    overwrite: bool = False
    pickle_protocol: int = 4
    joblib_compress: int = 3
    parquet_compression: Any = "snappy"
    parquet_index: bool = True


# 3. Output DTO
@dataclass
class PersistenceOutputDTO(OutputDTO):
    """저장·복원의 출처와 실행 객체. value는 JSON에서 제외하고 직접 접근합니다.

    Attributes:
        operation (str): save/load.
        format (str): 적용한 직렬화 형식.
        source_name (str): 로컬 경로 또는 bytes/stream 이름.
        source_kind (str): path/bytes/stream.
        n_bytes (int): 저장/읽은 바이트 수.
        object_type (str): 객체의 module.class 이름.
        metadata (Dict[str,Any]): 적용 엔진·스트림 위치 복원 여부.
        parameters (Dict[str,Any]): 적용 Profile 전체.
        warnings (List[str]): pickle/joblib의 코드 실행·환경 의존성 조건.
        value (Any): load의 원 객체; save는None. JSON 제외 실행 필드.
    """

    operation: str
    format: str
    source_name: str
    source_kind: str
    n_bytes: int
    object_type: str
    metadata: Dict[str, Any]
    parameters: Dict[str, Any]
    warnings: List[str]
    value: Any = None
    runtime_fields = OutputDTO.runtime_fields | frozenset(("value",))


# 4. Input Validation
_EXTENSIONS = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".joblib": "joblib",
    ".pkl": "pickle",
    ".pickle": "pickle",
}


def _configuration(source, profile, recipe_path, overrides, operation):
    """명시 Profile/레시피·선택 형식의 타입과 적용 범위를 검증합니다."""
    from .tabular import _source_info

    p = resolve_profile(
        "artifact-persistence", profile, recipe_path=recipe_path, overrides=overrides
    )
    if not isinstance(p, PersistenceProfile):
        raise TypeError("PersistenceProfile이 필요합니다.")
    if p.format not in ("auto", "parquet", "joblib", "pickle"):
        raise ValueError("format은 auto/parquet/joblib/pickle입니다.")
    for key in ("trusted", "overwrite", "parquet_index"):
        if type(getattr(p, key)) is not bool:
            raise TypeError(key + "는 bool입니다.")
    if (
        type(p.pickle_protocol) is not int
        or not 0 <= p.pickle_protocol <= pickle.HIGHEST_PROTOCOL
    ):
        raise ValueError("pickle_protocol이 현재 Python의 지원 범위를 벗어났습니다.")
    if type(p.joblib_compress) is not int or not 0 <= p.joblib_compress <= 9:
        raise ValueError("joblib_compress는0..9 정수입니다.")
    if p.parquet_compression not in (None, "snappy", "gzip", "brotli", "lz4", "zstd"):
        raise ValueError("지원하지 않는 Parquet 압축입니다.")
    name, kind = _source_info(source, writable=operation == "save")
    fmt = p.format if p.format != "auto" else _EXTENSIONS.get(Path(name).suffix.lower())
    if fmt is None:
        raise ValueError("확장자로 형식을 판단할 수 없습니다. format을 명시하십시오.")
    if fmt != "joblib" and p.joblib_compress != 3:
        raise ValueError("joblib_compress는 joblib에만 적용합니다.")
    if fmt == "parquet" and p.pickle_protocol != 4:
        raise ValueError("pickle_protocol은 pickle/joblib에만 적용합니다.")
    if fmt != "parquet" and (p.parquet_compression != "snappy" or not p.parquet_index):
        raise ValueError("Parquet 저장 설정은 parquet에만 적용합니다.")
    if operation == "load":
        if (
            p.overwrite
            or p.joblib_compress != 3
            or p.pickle_protocol != 4
            or p.parquet_compression != "snappy"
            or not p.parquet_index
        ):
            raise ValueError("저장 전용 설정은 load에서 지정하지 않습니다.")
        if fmt in ("joblib", "pickle") and not p.trusted:
            raise ValueError(
                "pickle/joblib은 읽을 때 코드가 실행될 수 있습니다. 신뢰한 파일에만 trusted=True를 지정하십시오."
            )
    return p, name, kind, fmt


# 5. Individual Model Runners


def _deserialize(raw, fmt):
    """선택 형식만 로드합니다. 신뢰 검사는 호출 전에 끝나야 합니다."""
    if fmt == "pickle":
        return pickle.loads(raw)
    if fmt == "joblib":
        return optional_package("joblib", "joblib").load(BytesIO(raw))
    optional_package("pyarrow", "pyarrow")
    frame = pd.read_parquet(BytesIO(raw), engine="pyarrow")
    validate_columns(list(frame.columns))
    return frame


def _serialize(value, fmt, p):
    """외부 파일을 건드리기 전에 직렬화를 완료합니다."""
    if fmt == "pickle":
        return pickle.dumps(value, protocol=p.pickle_protocol)
    buffer = BytesIO()
    if fmt == "joblib":
        optional_package("joblib", "joblib").dump(
            value, buffer, compress=p.joblib_compress, protocol=p.pickle_protocol
        )
    else:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Parquet에는 DataFrame이 필요합니다.")
        validate_columns(list(value.columns))
        optional_package("pyarrow", "pyarrow")
        value.to_parquet(
            buffer,
            engine="pyarrow",
            index=p.parquet_index,
            compression=p.parquet_compression,
        )
    return buffer.getvalue()


def _write_path(path, payload, overwrite):
    """같은 디렉터리의 임시 파일로 기록한 뒤 완성된 파일만 공개합니다."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        if overwrite:
            os.replace(str(temporary), str(path))
        else:
            # link는 기존 경로가 있으면 실패하므로 검사와 저장 사이의 덮어쓰기도 방지합니다.
            os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            _unlink(temporary)


def _unlink(path):
    """Python3.7에서도 이미 정리된 임시 경로를 허용합니다."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _result(operation, fmt, name, kind, raw, value, p, metadata):
    """실행 객체와 JSON 기록을 분리하는 전용 DTO를 구성합니다."""
    warnings = []
    if fmt in ("joblib", "pickle"):
        warnings.append(
            "신뢰한 파일만 복원하십시오. 객체 복원에는 원래 Python/라이브러리/클래스 정의와 호환되는 환경이 필요합니다."
        )
    return PersistenceOutputDTO(
        operation,
        fmt,
        name,
        kind,
        len(raw),
        type(value).__module__ + "." + type(value).__name__,
        metadata,
        asdict(p),
        warnings,
        value if operation == "load" else None,
    )


# 6. Public Runner


def save_artifact(
    value, destination, profile=None, *, recipe_path=None, overrides=None
):
    """표 또는 분석 객체를 로컬 파일/바이너리 스트림에 저장합니다.

    Args:
        value: Parquet은 DataFrame, pickle/joblib은 직렬화 가능한 분석 객체.
        destination: 로컬 경로 또는 write() 가능한 바이너리 스트림. 부모 폴더는 존재해야 함.
        profile: 명시 PersistenceProfile 또는None.
        recipe_path: 사용자 JSON 레시피 경로.
        overrides: 명시 설정 재정의.
    Returns:
        PersistenceOutputDTO: 저장 출처·형식·크기·설정. value는None.
    Raises:
        TypeError/ValueError: 설정·입력·직렬화 오류.
        OSError/ImportError: 파일 쓰기·선택 의존성 오류. 기존 파일은 기본 FileExistsError.
    """
    p, name, kind, fmt = _configuration(
        destination, profile, recipe_path, overrides, "save"
    )
    if kind == "path" and not p.overwrite and Path(name).exists():
        raise FileExistsError(name)
    if kind != "path" and not callable(getattr(destination, "write", None)):
        raise TypeError("로컬 경로 또는 바이너리 쓰기 스트림이 필요합니다.")
    raw = _serialize(value, fmt, p)
    if kind == "path":
        _write_path(name, raw, p.overwrite)
    else:
        if p.overwrite:
            raise ValueError("overwrite는 로컬 파일에만 적용합니다.")
        written = destination.write(raw)
        if written is not None and written != len(raw):
            raise OSError("스트림에 전체 바이트를 기록하지 못했습니다.")
    return _result(
        "save",
        fmt,
        name,
        kind,
        raw,
        value,
        p,
        {"engine": "pyarrow" if fmt == "parquet" else fmt},
    )


def load_artifact(source, profile=None, *, recipe_path=None, overrides=None):
    """원 표 또는 저장된 분석 객체를 재학습 없이 복원합니다.

    Args:
        source: 로컬 경로·bytes·읽기 가능한 바이너리 스트림.
        profile: 명시 PersistenceProfile 또는None.
        recipe_path: 사용자 JSON 레시피 경로.
        overrides: 명시 설정. pickle/joblib에는 trusted=True 필수.
    Returns:
        PersistenceOutputDTO: 복원 객체value·출처·설정·경고. value는 JSON 제외.
    Raises:
        TypeError/ValueError: 설정·입력·구조 오류.
        OSError/ImportError/pickle.UnpicklingError: 파일·의존성·직렬화/환경 호환 오류.
    """
    from .tabular import _payload

    p, name, kind, fmt = _configuration(source, profile, recipe_path, overrides, "load")
    raw, metadata = _payload(source, name, kind)
    if not isinstance(raw, bytes):
        raise TypeError("직렬화 파일에는 바이너리 입력이 필요합니다.")
    value = _deserialize(raw, fmt)
    result = _result("load", fmt, name, kind, raw, value, p, metadata)
    if kind == "stream" and not metadata["stream_position_restored"]:
        result.warnings.append("seek 불가 스트림은 현재 위치부터 소비했습니다.")
    return result


def save(value, destination, profile=None, *, recipe_path=None, overrides=None):
    """save_artifact의 편의 API.

    Args:
        value: 저장할 표/분석 객체.
        destination: 로컬 경로 또는 바이너리 스트림.
        profile: PersistenceProfile 또는None.
        recipe_path: 선택 JSON 경로.
        overrides: 명시 설정 재정의.
    Returns:
        PersistenceOutputDTO: 저장 결과.
    Raises:
        TypeError/ValueError/OSError/ImportError: save_artifact와 동일.
    """
    return save_artifact(
        value, destination, profile, recipe_path=recipe_path, overrides=overrides
    )


def load(source, profile=None, *, recipe_path=None, overrides=None):
    """메타데이터 없이 원 객체를 반환하는 load_artifact 편의 API.

    Args:
        source: 로컬 경로·bytes·바이너리 스트림.
        profile: PersistenceProfile 또는None.
        recipe_path: 선택 JSON 경로.
        overrides: 명시 설정; pickle/joblib에는 trusted=True 필수.
    Returns:
        Any: 복원된 원 표/분석 객체.
    Raises:
        TypeError/ValueError/OSError/ImportError/pickle.UnpicklingError: load_artifact와 동일.
    """
    return load_artifact(
        source, profile, recipe_path=recipe_path, overrides=overrides
    ).value


# 7. Example Execution
if __name__ == "__main__":
    buffer = BytesIO()
    saved = save(
        {"features": ["x"], "target": "y"}, buffer, overrides={"format": "pickle"}
    )
    restored = load(buffer.getvalue(), overrides={"format": "pickle", "trusted": True})
    print(saved.to_json(indent=2))
    print(restored)
