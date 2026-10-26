"""
================================================================================
tabular.py - JSON·XML·YAML·TXT·XLSX·CSV 공통 데이터 로딩
================================================================================
설명:
    - JSON 레시피 → 전용 Profile → 로컬 파일/스트림 해석 → 전용 Output DTO.
분석 가정:
    - 로딩과 Feature 적합을 구분합니다. 행 제거·결측 대치·학습 스케일링은 하지 않습니다.
    - 중첩 레코드 경로·시트·표 구조를 명시하며 임의 추측으로 자료를 펼치지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin-klg@naver.com)
    - 작성일: 2026-10-26
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-26  김재현(Finn)     v1.0.0    파일 형식별 읽기·스키마·DTO 계약
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field, asdict
from io import StringIO
from pathlib import Path
import codecs
import os
from typing import Any, Dict, List, Optional
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from . import _readers


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class TabularLoaderProfile:
    """파일 해석 조건. 행·열 순서와 명시하지 않은 결측/중첩 값은 보존합니다.

    Attributes:
        format (str): auto/csv/tsv/txt/json/jsonl/xml/yaml/xlsx. 기본 auto; 확장자로 판단.
        encoding (str): 텍스트 바이트 디코딩. 기본 utf-8-sig; XML 바이트는 자체 선언 사용.
        sep (Optional[str]): 구분자 한 문자 또는 r'\\s+'. 기본 None; csv 쉼표, tsv 탭. 구분자 TXT에는 필수.
        header (Optional[int]): CSV/구분자 TXT/XLSX의 첫 행 헤더0 또는 헤더 없음None. 기본0.
        names (Optional[List[str]]): 명시 열 이름; 기본None. header0이면 기존 첫 행을 대체.
        text_mode (str): TXT lines/delimited. 기본 lines; line_number·text 두 열.
        keep_blank_lines (bool): TXT/구분자 표/XLSX의 빈 행 보존. 기본True.
        sheet_name (Any): XLSX의 시트 이름 또는0 이상 위치. 기본0; 한 시트만 반환.
        excel_data_only (bool): XLSX 수식 대신 저장된 캐시값 읽기. 기본True; 수식을 계산하지 않음.
        orient (str): JSON/YAML records/columns/index/split. 기본 records; JSONL은records만.
        record_path (List[str]): JSON/YAML 루트 객체부터 레코드까지 키 목록. 기본[]; JSONL에는 미지원.
        flatten (bool): records의 중첩 dict만 점 표기로 평탄화. 기본False; 목록은 explode하지 않음.
        xml_xpath (str): ElementTree 형식 행 선택 경로. 기본 './*'; 전체 XPath가 아님.
        xml_namespaces (Dict[str,str]): XML 경로용 prefix→URI. 기본{}; 출력 태그 URI는 보존.
        xml_attributes (bool): XML 행 속성 @key·자식 속성 tag@key 포함. 기본True.
        dtype (Dict[str,str]): 명시 열→pandas dtype 변환; 숫자형은 엄격한 숫자 파싱 후 변환. 기본{}; 자동 학습 전처리 아님.
        parse_dates (List[str]): 엄격한 날짜 파싱 열. 기본[]; dtype과 중복 불가.
        date_format (Optional[str]): parse_dates의 명시 형식. 기본None; 파싱 실패는 오류.
        required_columns (List[str]): 필수 열; 기본[]. 이름·행을 임의 변경하지 않음.
        allow_empty (bool): 행 또는 열이 없는 표 허용. 기본False.
    """

    format: str = "auto"
    encoding: str = "utf-8-sig"
    sep: Optional[str] = None
    header: Optional[int] = 0
    names: Optional[List[str]] = None
    text_mode: str = "lines"
    keep_blank_lines: bool = True
    sheet_name: Any = 0
    excel_data_only: bool = True
    orient: str = "records"
    record_path: List[str] = field(default_factory=list)
    flatten: bool = False
    xml_xpath: str = "./*"
    xml_namespaces: Dict[str, str] = field(default_factory=dict)
    xml_attributes: bool = True
    dtype: Dict[str, str] = field(default_factory=dict)
    parse_dates: List[str] = field(default_factory=list)
    date_format: Optional[str] = None
    required_columns: List[str] = field(default_factory=list)
    allow_empty: bool = False


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class TabularLoaderOutputDTO(OutputDTO):
    """해석된 표와 출처/설정. data는 분석용 원 자료이며 전처리 적합 결과가 아닙니다.

    Attributes:
        data (pd.DataFrame): 입력 행·열 순서의 표; names/dtype/날짜/flatten은 명시한 경우만 적용.
        source_name (str): 파일 경로 또는 <bytes>/<stream> 이름; 원문 payload는 메타데이터에 넣지 않음.
        source_kind (str): path/bytes/stream. URL 다운로드는 수행하지 않음.
        format (str): 실제 적용한 정규 형식 이름.
        n_rows (int): 반환한 표의 행 수; 자동 dropna 없음.
        n_columns (int): 반환한 표의 열 수.
        columns (List[str]): 반환 열 순서; 고유하고 비어 있지 않은 문자열.
        dtypes (Dict[str,str]): 열별 실제 pandas dtype.
        metadata (Dict[str,Any]): 실제 시트명·스트림 위치 복원 등 형식별 읽기 정보.
        parameters (Dict[str,Any]): 적용한 전체 Profile; 원 데이터 경로는 별도source_name.
        warnings (List[str]): 수식 캐시·중첩 값·스트림 소비 등의 조건과 한계.
    """

    data: pd.DataFrame
    source_name: str
    source_kind: str
    format: str
    n_rows: int
    n_columns: int
    columns: List[str]
    dtypes: Dict[str, str]
    metadata: Dict[str, Any]
    parameters: Dict[str, Any]
    warnings: List[str]


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
_FORMATS = {"csv", "tsv", "txt", "json", "jsonl", "xml", "yaml", "xlsx"}
_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "txt",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xlsx": "xlsx",
}


def _string_list(value, name, unique=True):
    """문자열 목록을 복사 없이 검증합니다."""
    if not isinstance(value, list) or any(
        not isinstance(v, str) or not v.strip() for v in value
    ):
        raise ValueError(name + "은 비어 있지 않은 문자열 목록이어야 합니다.")
    if unique and len(set(value)) != len(value):
        raise ValueError(name + "의 중복 항목은 지원하지 않습니다.")


def _validate_profile(p):
    """형식별 타입·정의역·명시적 스키마 변환 조건을 검증합니다.

    Args:
        p: TabularLoaderProfile 설정.
    Returns:
        None: 정상 설정.
    Raises:
        TypeError/ValueError: 설정 타입·정의역 오류.
    """
    if not isinstance(p, TabularLoaderProfile):
        raise TypeError("TabularLoaderProfile이 필요합니다.")
    if p.format not in _FORMATS | {"auto"}:
        raise ValueError("지원하지 않는 format입니다.")
    if not isinstance(p.encoding, str):
        raise TypeError("encoding은 문자열입니다.")
    try:
        codecs.lookup(p.encoding)
    except LookupError as error:
        raise ValueError("알 수 없는 encoding입니다.") from error
    if p.sep is not None and (
        not isinstance(p.sep, str)
        or not (len(p.sep) == 1 or p.sep == r"\s+")
        or p.sep in ("\n", "\r", "\x00")
    ):
        raise ValueError('sep는 구분자 한 문자 또는 r"\\s+"입니다.')
    if p.header is not None and (type(p.header) is not int or p.header != 0):
        raise ValueError("header는0 또는None입니다.")
    if p.names is not None:
        _string_list(p.names, "names")
        _readers.validate_columns(p.names)
    if p.names == []:
        raise ValueError("names는 빈 목록일 수 없습니다.")
    for key in (
        "keep_blank_lines",
        "excel_data_only",
        "flatten",
        "xml_attributes",
        "allow_empty",
    ):
        if not isinstance(getattr(p, key), bool):
            raise TypeError(key + "은 bool입니다.")
    if p.text_mode not in ("lines", "delimited"):
        raise ValueError("text_mode는 lines/delimited입니다.")
    if (
        isinstance(p.sheet_name, bool)
        or not isinstance(p.sheet_name, (str, int))
        or (isinstance(p.sheet_name, int) and p.sheet_name < 0)
        or (isinstance(p.sheet_name, str) and not p.sheet_name)
    ):
        raise ValueError("sheet_name은 비어 있지 않은 이름 또는0 이상 위치입니다.")
    if p.orient not in ("records", "columns", "index", "split"):
        raise ValueError("orient는 records/columns/index/split입니다.")
    _string_list(p.record_path, "record_path", unique=False)
    if p.flatten and p.orient != "records":
        raise ValueError("flatten은 records orient에서만 적용합니다.")
    if not isinstance(p.xml_xpath, str) or not p.xml_xpath:
        raise ValueError("xml_xpath가 필요합니다.")
    if not isinstance(p.xml_namespaces, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v
        for k, v in p.xml_namespaces.items()
    ):
        raise ValueError("xml_namespaces는 prefix→URI 문자열 사전입니다.")
    if not isinstance(p.dtype, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in p.dtype.items()
    ):
        raise ValueError("dtype은 열 이름→dtype 문자열 사전입니다.")
    _string_list(p.parse_dates, "parse_dates")
    _string_list(p.required_columns, "required_columns")
    if set(p.dtype) & set(p.parse_dates):
        raise ValueError("dtype과 parse_dates의 열이 겹칩니다.")
    if p.date_format is not None and (
        not isinstance(p.date_format, str) or not p.date_format or not p.parse_dates
    ):
        raise ValueError("date_format은 parse_dates와 함께 지정하는 문자열입니다.")


def _source_info(source):
    """데이터 내용과 파일 경로를 혼동하지 않고 소스 이름을 결정합니다."""
    if isinstance(source, (str, os.PathLike)):
        name = os.fspath(source)
        if not isinstance(name, str):
            raise TypeError("파일 경로는 문자열이어야 합니다.")
        if "://" in name:
            raise ValueError(
                "로컬 파일만 지원합니다. UCI 다운로드는 전용 로더를 사용하십시오."
            )
        return name, "path"
    if isinstance(source, (bytes, bytearray)):
        return "<bytes>", "bytes"
    if callable(getattr(source, "read", None)):
        name = getattr(source, "name", "<stream>")
        return name if isinstance(name, str) else "<stream>", "stream"
    raise TypeError("파일 경로·bytes·읽을 수 있는 스트림이 필요합니다.")


def _format_options(p, fmt):
    """선택 형식에 적용되지 않는 비기본 옵션을 조용히 무시하지 않습니다."""
    delim = fmt in ("csv", "tsv") or (fmt == "txt" and p.text_mode == "delimited")
    if p.sep is not None and not delim:
        raise ValueError("sep는 구분자 표에만 적용합니다.")
    if fmt == "txt" and p.text_mode == "delimited" and p.sep is None:
        raise ValueError("구분자 TXT에는 sep를 명시하십시오.")
    if (p.names is not None or p.header is None) and not (delim or fmt == "xlsx"):
        raise ValueError("names/header는 구분자 표·XLSX에만 적용합니다.")
    if p.text_mode != "lines" and fmt != "txt":
        raise ValueError("text_mode는 TXT에만 적용합니다.")
    if (p.sheet_name != 0 or not p.excel_data_only) and fmt != "xlsx":
        raise ValueError("sheet_name/excel_data_only는 XLSX에만 적용합니다.")
    if (p.orient != "records" or p.record_path or p.flatten) and fmt not in (
        "json",
        "jsonl",
        "yaml",
    ):
        raise ValueError("orient/record_path/flatten은 JSON/YAML에만 적용합니다.")
    if fmt == "jsonl" and (p.orient != "records" or p.record_path):
        raise ValueError("JSONL은 records이며 루트 record_path를 적용하지 않습니다.")
    if (
        p.xml_xpath != "./*" or p.xml_namespaces or not p.xml_attributes
    ) and fmt != "xml":
        raise ValueError("XML 경로/속성 설정은 XML에만 적용합니다.")
    if not p.keep_blank_lines and fmt not in ("csv", "tsv", "txt", "xlsx"):
        raise ValueError("keep_blank_lines는 TXT·구분자 표·XLSX에만 적용합니다.")


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _payload(source, name, kind):
    """소스를 끝까지 읽으며 가능한 경우 스트림의 기존 위치를 복원합니다."""
    if kind == "path":
        return Path(name).read_bytes(), {}
    if kind == "bytes":
        return bytes(source), {}
    position = None
    try:
        position = source.tell()
    except (AttributeError, OSError, ValueError):
        pass
    restored = False
    try:
        value = source.read()
    finally:
        if position is not None:
            try:
                source.seek(position)
                restored = True
            except (AttributeError, OSError, ValueError):
                pass
    if not isinstance(value, (str, bytes)):
        raise TypeError("스트림 read()는 str 또는bytes를 반환해야 합니다.")
    return value, dict(stream_position_restored=restored)


def _run(source, p):
    """형식 해석·명시 변환·스키마 검증 후 원 자료 DTO를 생성합니다."""
    name, kind = _source_info(source)
    fmt = p.format if p.format != "auto" else _EXTENSIONS.get(Path(name).suffix.lower())
    if fmt is None:
        raise ValueError("확장자로 형식을 판단할 수 없습니다. format을 명시하십시오.")
    _format_options(p, fmt)
    raw, metadata = _payload(source, name, kind)
    warnings = []
    if fmt == "xlsx":
        if not isinstance(raw, bytes):
            raise TypeError("XLSX에는 bytes 기반 파일/스트림이 필요합니다.")
        data, sheet = _readers.excel_table(raw, p)
        metadata["sheet_name"] = sheet
        if p.excel_data_only:
            warnings.append(
                "XLSX 수식을 계산하지 않고 저장된 캐시값을 읽습니다. 캐시 없는 수식은 결측일 수 있습니다."
            )
    elif fmt == "xml":
        data = _readers.xml_table(raw, p)
    else:
        text = raw.decode(p.encoding) if isinstance(raw, bytes) else raw
        if fmt in ("csv", "tsv") or (fmt == "txt" and p.text_mode == "delimited"):
            data = _readers.delimited(text, fmt, p)
        elif fmt == "txt":
            lines = [
                (i, line)
                for i, line in enumerate(text.splitlines(), 1)
                if p.keep_blank_lines or line.strip()
            ]
            data = pd.DataFrame(lines, columns=["line_number", "text"])
        else:
            value = (
                _readers.yaml_value(text)
                if fmt == "yaml"
                else _readers.json_value(text, lines=fmt == "jsonl")
            )
            data = _readers.structured_table(value, p)
    _readers.validate_columns(list(data.columns))
    if data.empty and not p.allow_empty:
        raise ValueError(
            "행 또는 열이 없는 자료입니다. allow_empty로 명시할 수 있습니다."
        )
    needed = set(p.required_columns) | set(p.dtype) | set(p.parse_dates)
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(
            "필수/변환 열을 찾을 수 없습니다: " + ", ".join(sorted(missing))
        )
    for column, dtype in p.dtype.items():
        target_dtype = pd.api.types.pandas_dtype(dtype)
        # pandas1.3은 object의 숫자 문자열→nullable Float64 직접 astype을 거부합니다.
        # 명시 숫자형에만 엄격한 파싱을 적용하여 두 지원 환경의 계약을 맞춥니다.
        if pd.api.types.is_integer_dtype(target_dtype) or pd.api.types.is_float_dtype(
            target_dtype
        ):
            data[column] = pd.to_numeric(data[column], errors="raise")
        data[column] = data[column].astype(target_dtype)
    for column in p.parse_dates:
        data[column] = pd.to_datetime(
            data[column], format=p.date_format, errors="raise"
        )
    if any(
        data[col].map(lambda v: isinstance(v, (dict, list, tuple))).any()
        for col in data.select_dtypes(include=["object"]).columns
    ):
        warnings.append(
            "중첩 dict/list 값은 셀에 보존했습니다. Feature 입력으로 쓰기 전에 명시적으로 처리하십시오."
        )
    if kind == "stream" and not metadata["stream_position_restored"]:
        warnings.append("seek 불가 스트림은 현재 위치부터 소비했습니다.")
    return TabularLoaderOutputDTO(
        data,
        name,
        kind,
        fmt,
        len(data),
        data.shape[1],
        list(data.columns),
        {str(c): str(t) for c, t in data.dtypes.items()},
        metadata,
        asdict(p),
        warnings,
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_tabular_loader(source, profile=None, *, recipe_path=None, overrides=None):
    """JSON→overrides→Profile 계약으로 사용자 파일을 DataFrame DTO로 읽습니다.

    Args:
        source: 로컬 경로·bytes·text/binary 스트림. 익명 소스는 format을 명시.
        profile: 명시 TabularLoaderProfile 또는None; 레시피 옵션과 동시 지정 불가.
        recipe_path: 기본 레시피 대신 읽을 평면 JSON 경로.
        overrides: 명시적 설정 재정의; 알 수 없는 키는 오류.
    Returns:
        TabularLoaderOutputDTO: data·출처·행/열·dtype·설정·조건 경고.
    Raises:
        TypeError/ValueError: 설정·구조·스키마·파싱 오류; 빈 자료/중복 키·열.
        OSError/UnicodeError: 파일 읽기·텍스트 디코딩 실패.
        ImportError: YAML/XML/XLSX 선택 의존성 누락.
    """
    p = resolve_profile(
        "tabular-loader", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_profile(p)
    return _run(source, p)


def read_table(source, profile=None, *, recipe_path=None, overrides=None):
    """같은 검증으로 DataFrame만 반환하는 기존 분석 코드용 편의 함수.

    Args:
        source: run_tabular_loader와 같은 입력 소스.
        profile: 명시 TabularLoaderProfile 또는None.
        recipe_path: 선택 레시피 경로.
        overrides: 명시 설정 재정의.
    Returns:
        pd.DataFrame: 전처리 적합 없이 읽은 표. 경고/출처는 DTO API에서 확인.
    Raises:
        TypeError/ValueError/OSError/ImportError: run_tabular_loader의 입력·파싱 오류.
    """
    return run_tabular_loader(
        source, profile, recipe_path=recipe_path, overrides=overrides
    ).data


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_tabular_loader(
        StringIO('[{"feature":1,"target":2},{"feature":3,"target":4}]'),
        overrides={"format": "json", "required_columns": ["target"]},
    )
    print(result.to_json(indent=2))
