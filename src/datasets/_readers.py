"""형식별 디코딩. 학습·결측 대치·행 필터링은 수행하지 않습니다."""

from io import BytesIO, StringIO
from collections.abc import Mapping
import csv
import json
import re
from zipfile import BadZipFile
import pandas as pd


def optional_package(name, package):
    """필요한 형식의 의존성만 로드하고 설치 위치를 안내합니다."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            package + "가 필요합니다. requirements-data-loading.txt를 설치하십시오."
        ) from error


def unique_pairs(pairs):
    """JSON 객체의 중복 키를 덮어쓰지 않습니다."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("중복 JSON 키: " + str(key))
        result[key] = value
    return result


def check_tree(value, active=None):
    """순환 alias·문자열 이외 키를 거부하고 공유된 비순환 구조는 허용합니다."""
    active = set() if active is None else active
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError("순환 레코드 구조는 표로 읽을 수 없습니다.")
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                if any(not isinstance(k, str) for k in value):
                    raise ValueError("구조화 자료의 키는 문자열이어야 합니다.")
                children = value.values()
            else:
                children = value
            for child in children:
                check_tree(child, active)
        finally:
            active.remove(identity)
    elif isinstance(value, (set, bytes, bytearray)):
        raise ValueError("집합·binary YAML 값은 표 레코드에서 지원하지 않습니다.")


def yaml_value(text):
    """SafeLoader를 사용하며 중복 키·merge 키·사용자 객체 생성을 허용하지 않습니다."""
    yaml = optional_package("yaml", "PyYAML")

    class UniqueSafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ValueError("YAML merge 키는 지원하지 않습니다.")
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML 키는 문자열이어야 합니다.")
            if key in result:
                raise ValueError("중복 YAML 키: " + key)
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        return yaml.load(text, Loader=UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError("YAML 파싱 실패: " + str(error)) from error


def flattened(record, prefix="", result=None):
    """중첩 dict만 점 표기로 평탄화하며 충돌과 빈 dict의 유실을 방지합니다."""
    result = {} if result is None else result
    for key, value in record.items():
        name = prefix + "." + key if prefix else key
        if isinstance(value, Mapping) and value:
            flattened(value, name, result)
        else:
            if name in result:
                raise ValueError("평탄화 열 이름 충돌: " + name)
            result[name] = value
    return result


def structured_table(value, profile):
    """명시한 레코드 경로와 orient만 적용하며 리스트를 임의 explode하지 않습니다."""
    check_tree(value)
    for key in profile.record_path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError("record_path를 찾을 수 없습니다: " + key)
        value = value[key]
    orient = profile.orient
    if orient == "records":
        records = [value] if isinstance(value, Mapping) else value
        if not isinstance(records, list) or any(
            not isinstance(r, Mapping) for r in records
        ):
            raise ValueError(
                "records는 객체 또는 객체 목록이어야 합니다. record_path/orient를 지정하십시오."
            )
        return pd.DataFrame(
            [flattened(r) for r in records] if profile.flatten else records
        )
    if orient == "columns":
        if not isinstance(value, Mapping) or any(
            not isinstance(v, list) for v in value.values()
        ):
            raise ValueError("columns는 열 이름→같은 길이 목록 객체입니다.")
        return pd.DataFrame(value)
    if orient == "index":
        if not isinstance(value, Mapping) or any(
            not isinstance(v, Mapping) for v in value.values()
        ):
            raise ValueError("index는 행 이름→레코드 객체입니다.")
        return pd.DataFrame.from_dict(value, orient="index")
    if (
        not isinstance(value, Mapping)
        or set(value) - {"data", "columns", "index"}
        or not {"data", "columns"} <= set(value)
    ):
        raise ValueError("split에는 data·columns 및 선택 index만 필요합니다.")
    if (
        not isinstance(value["columns"], list)
        or not isinstance(value["data"], list)
        or any(
            not isinstance(row, list) or len(row) != len(value["columns"])
            for row in value["data"]
        )
    ):
        raise ValueError("split의 열 수와 행 길이가 일치해야 합니다.")
    return pd.DataFrame(
        value["data"], columns=value["columns"], index=value.get("index")
    )


def json_value(text, lines=False):
    """엄격한 JSON 및 JSON Lines를 해석합니다."""

    def invalid_constant(value):
        raise ValueError("표준 JSON이 아닌 상수: " + value)

    def read(value):
        return json.loads(
            value, object_pairs_hook=unique_pairs, parse_constant=invalid_constant
        )

    if not lines:
        return read(text)
    records = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = read(line)
        except ValueError as error:
            raise ValueError("JSON Lines %d행: %s" % (number, error)) from error
        if not isinstance(value, Mapping):
            raise ValueError("JSON Lines의 각 행은 객체여야 합니다: %d행" % number)
        records.append(value)
    return records


def delimited(text, fmt, profile):
    """중복 헤더를 먼저 검사하고 CSV/TSV/구분자 TXT를 읽습니다."""
    sep = profile.sep or ("\t" if fmt == "tsv" else ",")
    if sep == r"\s+":
        raw_rows = [
            re.split(r"\s+", line.strip()) if line.strip() else []
            for line in text.splitlines()
        ]
    else:
        try:
            raw_rows = list(csv.reader(StringIO(text), delimiter=sep, strict=True))
        except csv.Error as error:
            raise ValueError("구분자 표 파싱 실패: " + str(error)) from error
    if not profile.keep_blank_lines:
        raw_rows = [r for r in raw_rows if r and (len(r) > 1 or r[0].strip())]
    width = (
        len(profile.names)
        if profile.names is not None
        else len(next((r for r in raw_rows if r), []))
    )
    if any(len(row) > width for row in raw_rows):
        raise ValueError(
            "헤더/명시 열 수보다 넓은 행이 있습니다. 행을 index로 자동 전환하지 않습니다."
        )
    if profile.header == 0 and profile.names is None:
        if sep == r"\s+":
            lines = text.splitlines()
            line = (
                next((v for v in lines if v.strip()), "")
                if not profile.keep_blank_lines
                else (lines[0] if lines else "")
            )
            if '"' in line:
                raise ValueError("공백 구분자의 헤더에는 인용부호를 지원하지 않습니다.")
            names = re.split(r"\s+", line.strip()) if line.strip() else []
        else:
            rows = csv.reader(StringIO(text), delimiter=sep)
            names = (
                next((r for r in rows if r and (len(r) > 1 or r[0].strip())), [])
                if not profile.keep_blank_lines
                else next(rows, [])
            )
        validate_columns(names)
        if not names and text.strip():
            raise ValueError(
                "첫 헤더 행이 비었습니다. keep_blank_lines=False 또는 header/names를 지정하십시오."
            )
    try:
        frame = pd.read_csv(
            StringIO(text),
            sep=sep,
            header=profile.header,
            names=profile.names,
            skip_blank_lines=not profile.keep_blank_lines,
        )
    except pd.errors.EmptyDataError:
        if text.strip():
            raise ValueError("비어 있지 않은 자료에서 표 열을 해석하지 못했습니다.")
        return pd.DataFrame(columns=profile.names or [])
    if profile.header is None and profile.names is None:
        frame.columns = ["column_" + str(i) for i in range(frame.shape[1])]
    return frame


def validate_columns(columns):
    """문자열 열 이름·중복·빈 이름을 확인합니다."""
    if any(not isinstance(c, str) or not c.strip() for c in columns):
        raise ValueError(
            "열 이름은 비어 있지 않은 문자열이어야 합니다. names로 지정할 수 있습니다."
        )
    if len(set(columns)) != len(columns):
        raise ValueError("중복 열 이름은 지원하지 않습니다.")


def xml_table(payload, profile):
    """DTD/외부 entity 없이 선택된 반복 요소를 행으로 읽습니다."""
    parser = optional_package("defusedxml.ElementTree", "defusedxml")
    try:
        root = parser.fromstring(
            payload, forbid_dtd=True, forbid_entities=True, forbid_external=True
        )
    except Exception as error:
        raise ValueError("XML 파싱 실패: " + str(error)) from error
    try:
        elements = root.findall(profile.xml_xpath, profile.xml_namespaces)
    except (SyntaxError, KeyError) as error:
        raise ValueError("xml_xpath/namespace 오류") from error
    records = []
    for element in elements:
        row = (
            {("@" + key): value for key, value in element.attrib.items()}
            if profile.xml_attributes
            else {}
        )
        if list(element) and (element.text or "").strip():
            raise ValueError("혼합 내용 XML은 표 행으로 변환하지 않습니다.")
        if not list(element) and element.text is not None:
            row["#text"] = element.text
        seen = set()
        for child in element:
            if child.tag in seen:
                raise ValueError("XML 행 안의 중복 요소: " + child.tag)
            seen.add(child.tag)
            if list(child) or (child.tail or "").strip():
                raise ValueError("중첩/혼합 XML은 행 경로를 구체적으로 지정하십시오.")
            row[child.tag] = child.text
            if profile.xml_attributes:
                row.update(
                    {
                        child.tag + "@" + key: value
                        for key, value in child.attrib.items()
                    }
                )
        records.append(row)
    return pd.DataFrame(records)


def excel_table(payload, profile):
    """한 시트를 읽으며 파일·워크북을 닫고 수식을 실행하지 않습니다."""
    excel = optional_package("openpyxl", "openpyxl")
    with BytesIO(payload) as stream:
        try:
            book = excel.load_workbook(
                stream, read_only=True, data_only=profile.excel_data_only
            )
        except (
            BadZipFile,
            excel.utils.exceptions.InvalidFileException,
            KeyError,
        ) as error:
            raise ValueError("유효한 XLSX 워크북이 아닙니다.") from error
        try:
            name = profile.sheet_name
            if isinstance(name, int):
                if name >= len(book.worksheets):
                    raise ValueError("sheet_name 인덱스가 시트 수를 넘습니다.")
                sheet = book.worksheets[name]
            else:
                if name not in book.sheetnames:
                    raise ValueError("sheet_name을 찾을 수 없습니다: " + name)
                sheet = book[name]
            rows = list(sheet.iter_rows(values_only=True))
            if not profile.keep_blank_lines:
                rows = [r for r in rows if any(v is not None for v in r)]
            if profile.header == 0:
                names = (
                    profile.names
                    if profile.names is not None
                    else (list(rows[0]) if rows else [])
                )
                rows = rows[1:]
            else:
                names = profile.names or [
                    "column_" + str(i) for i in range(len(rows[0]) if rows else 0)
                ]
            validate_columns(names)
            if any(len(r) != len(names) for r in rows):
                raise ValueError("XLSX 행 길이와 열 수가 다릅니다.")
            return pd.DataFrame(rows, columns=names), sheet.title
        finally:
            book.close()
