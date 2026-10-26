# 파일 데이터 로딩

공통 파일 로더는 **입력 파일 → JSON 레시피 → Profile → 형식 해석·스키마 검증 → DataFrame과 전용 DTO**로 동작합니다. 기존 분석 모듈의 설정·반환 명세를 따릅니다. 모델 설정과 로딩 설정은 분리하며, 파일을 읽은 뒤 평가 분할과 Feature 전처리로 연결합니다.

- 구현: [tabular.py](../src/datasets/tabular.py), 내부 파서: [_readers.py](../src/datasets/_readers.py)
- 기본 설정: [tabular-loader.json](../src/Recipes/tabular-loader.json), 등록 키: `tabular-loader`
- 실행 예제: [데이터 로딩 스켈레톤](../src/datasets/skeleton/tabular.py)
- 분석 단계 연결: [전체 파이프라인](ANALYSIS_PIPELINE.md)

## 제공 기능과 입력 범위

| 형식 | 기본 해석 | 필요한 경우 지정할 설정 |
| --- | --- | --- |
| CSV / TSV | 첫 행 헤더, 쉼표 / 탭 구분 표 | `encoding`, `sep`, `header`, `names` |
| JSON | 객체 목록 또는 단일 객체를 레코드로 변환 | `record_path`, `orient`, `flatten` |
| JSONL / NDJSON | 비어 있지 않은 각 줄의 JSON 객체를 한 행으로 읽음 | `format="jsonl"`; 루트 `record_path` 미지원 |
| YAML / YML | JSON과 같은 표 구조; 안전한 YAML 타입만 사용 | `record_path`, `orient`, `flatten` |
| XML | 루트의 직계 자식 요소를 행으로 읽음 | `xml_xpath`, `xml_namespaces`, `xml_attributes` |
| TXT | 원래 줄 번호와 문자열을 `line_number`, `text` 열로 반환 | 구분자 표는 `text_mode="delimited"`와 `sep` 필수 |
| XLSX | 첫 번째 시트, 첫 행 헤더 | `sheet_name`, `header`, `names`, `excel_data_only` |

로컬 `str`/`Path` 경로, `bytes`/`bytearray`, `read()` 가능한 텍스트·바이너리 스트림을 받습니다. 확장자의 대소문자를 구분하지 않으며 익명 스트림·bytes에는 `format`이 필요합니다. 문자열은 파일 경로로 해석하므로 문자열 본문은 `StringIO`로 전달합니다. URL 다운로드는 UCI 전용 로더가 담당합니다. `.xls`, `.xlsb`, 압축 파일, 여러 시트 동시 반환은 제공하지 않습니다.

스트림은 **현재 위치부터** 읽고, seek가 가능하면 원래 위치로 복원합니다. 호출자의 스트림을 닫지 않습니다. seek 불가 스트림의 소비는 DTO 경고에 기록합니다. 파일 전체를 메모리로 읽으므로 청크 단위 대용량 로더가 아닙니다.

## 공개 API와 설정 우선순위

```python
from src.datasets import read_table, run_tabular_loader

# DataFrame만 필요한 기존 코드
frame = read_table("data.xlsx", overrides={"sheet_name": "관측자료"})

# 출처·설정·경고까지 남기는 분석 코드
loaded = run_tabular_loader(
    "data.json",
    overrides={"record_path": ["payload", "rows"], "required_columns": ["target"]},
)
frame = loaded.data
print(loaded.to_json(indent=2))
```

두 함수 모두 `(source, profile=None, *, recipe_path=None, overrides=None)` 서명을 사용합니다. `read_table`은 같은 검증을 거쳐 `DataFrame`만 반환하며 출처·경고는 DTO API로 확인합니다. 기존과 같이 **기본 또는 사용자 JSON → overrides → Profile** 순서로 설정을 적용합니다. 명시 Profile은 레시피/overrides와 동시 지정할 수 없습니다. 알 수 없는 키와 중복 레시피 키는 오류입니다.

```python
from src.Recipes import load_profile
from src.datasets import run_tabular_loader

profile = load_profile("tabular-loader", overrides={"format": "csv", "encoding": "cp949"})
loaded = run_tabular_loader("data.csv", profile)
```

## Profile 기능 명세

| 필드 | 타입 / 기본값 | 동작·적용 조건 |
| --- | --- | --- |
| `format` | str / `auto` | auto, csv, tsv, txt, json, jsonl, xml, yaml, xlsx. auto는 확장자로 선택 |
| `encoding` | str / `utf-8-sig` | JSON/YAML/CSV/TSV/TXT 바이트 디코딩. XML 바이트는 XML 선언, XLSX는 워크북 인코딩 사용. 텍스트 스트림은 이미 디코딩된 값 |
| `sep` | Optional[str] / None | 구분자 한 문자 또는 `r"\s+"`. CSV/TSV 기본값은 쉼표/탭, 구분자 TXT에는 명시 필수 |
| `header` | Optional[int] / 0 | 구분자 표/XLSX에서 첫 행 헤더0 또는 헤더 없음None |
| `names` | Optional[List[str]] / None | 구분자 표/XLSX의 명시 열 이름. header0이면 첫 행을 건너뛰고 대체 |
| `text_mode` | str / `lines` | TXT의 lines 또는 delimited. lines는 공백·문자열을 보존 |
| `keep_blank_lines` | bool / True | TXT/구분자 표/XLSX의 빈 행 보존. False면 빈 행 제외; JSONL 빈 줄은 항상 제외 |
| `sheet_name` | str 또는 int / 0 | XLSX 시트 이름 또는0부터 시작하는 위치. 한 시트만 반환 |
| `excel_data_only` | bool / True | 저장된 수식 캐시를 읽음. False면 수식 문자열 보존 |
| `orient` | str / `records` | JSON/YAML의 records, columns, index, split. JSONL은 records만 |
| `record_path` | List[str] / [] | JSON/YAML 루트 객체부터 레코드까지 키 목록; 배열 위치 탐색/임의 JSONPath 미지원 |
| `flatten` | bool / False | records의 중첩 dict를 점 표기로 펼침. 목록은 셀에 보존, explode하지 않음 |
| `xml_xpath` | str / `./*` | 행 요소를 고르는 ElementTree 경로. 전체 XPath 언어와 다름 |
| `xml_namespaces` | Dict[str,str] / {} | 경로용 prefix→URI. 출력 열 이름의 namespace URI는 보존 |
| `xml_attributes` | bool / True | 행 속성은 `@id`, 자식 속성은 `tag@unit`로 포함 |
| `dtype` | Dict[str,str] / {} | 명시한 열만 dtype 변환. 정수/실수형은 엄격한 숫자 파싱 후 astype 적용; 실패는 오류 |
| `parse_dates` | List[str] / [] | 명시한 열만 엄격하게 날짜 변환. dtype과 같은 열 지정 불가 |
| `date_format` | Optional[str] / None | parse_dates가 있을 때 날짜 형식 지정. 잘못된 날짜는 오류 |
| `required_columns` | List[str] / [] | 필수 열 존재 확인. 추가 열의 제거·순서 변경 없음 |
| `allow_empty` | bool / False | 행 또는 열이 없는 자료는 기본 오류; True로 허용 |

형식에 맞지 않는 **비기본 옵션**은 조용히 무시하지 않고 오류로 알립니다. 열 이름은 중복되지 않는 비어 있지 않은 문자열이어야 합니다. 헤더 없는 구분자 표/XLSX에서 names도 생략하면 `column_0`부터 생성합니다.

## 구조화 자료의 해석

JSON/YAML의 orient별 입력 예시는 다음과 같습니다. records의 단일 객체는 한 행, columns는 같은 길이 열 목록, index는 행 이름별 객체, split은 명시 열·행·선택 인덱스입니다.

```python
# records
[{"x": 1, "y": 2}, {"x": 3, "y": 4}]
# columns
{"x": [1, 3], "y": [2, 4]}
# index
{"row_a": {"x": 1, "y": 2}, "row_b": {"x": 3, "y": 4}}
# split
{"columns": ["x", "y"], "data": [[1, 2], [3, 4]], "index": ["row_a", "row_b"]}
```

중첩 dict/list는 기본적으로 셀에 보존하고 경고를 남깁니다. flatten은 dict만 펼치며, `a.b`와 `a → b`처럼 같은 열 이름을 만들면 오류입니다. JSON의 중복 객체 키와 NaN/Infinity 상수는 거부합니다. YAML은 SafeLoader를 기반으로 중복 키·merge 키·문자열 아닌 키·순환 alias·사용자 객체 생성을 거부합니다.

XML은 선택된 행의 단순 자식 요소를 열로 읽습니다. 값은 기본 문자열이므로 숫자가 필요하면 dtype을 명시합니다. 같은 행 안의 반복 태그와 중첩/혼합 내용은 덮어쓰거나 버리지 않고 오류로 알립니다. 행 경로를 더 구체적으로 지정하거나 별도 도메인 변환이 필요합니다. DTD·entity는 허용하지 않습니다.

```python
frame = read_table("data.xml", overrides={
    "xml_xpath": ".//item",
    "dtype": {"quantity": "Int64"},
})
frame = read_table("data.txt", overrides={"text_mode": "delimited", "sep": "|"})
```

XLSX 수식은 실행하지 않습니다. 기본값은 마지막 저장된 캐시값이며 캐시가 없으면 결측일 수 있어 DTO에 경고를 남깁니다. 암호화 워크북은 지원하지 않습니다. [openpyxl 동작 명세](https://openpyxl.readthedocs.io/en/stable/tutorial.html)를 참고하세요.

## DTO 기능 명세

`TabularLoaderOutputDTO`는 기존 `OutputDTO`를 상속하며 `to_dict()`·`to_json()`·필드 접근 계약을 유지합니다. data도 직렬화되므로 DTO JSON을 저장할 때 원 자료가 포함됩니다.

| 필드 | 타입 | 제공 내용 |
| --- | --- | --- |
| `data` | pd.DataFrame | 읽은 표. 명시한 dtype/날짜/열 이름/flatten 외 학습 전처리 없음 |
| `source_name` | str | 경로 또는 스트림 이름, `<bytes>`/`<stream>` |
| `source_kind` | str | path, bytes, stream |
| `format` | str | 실제 적용 형식 |
| `n_rows`, `n_columns` | int | 반환 표의 행·열 수 |
| `columns` | List[str] | 열 순서 |
| `dtypes` | Dict[str,str] | 열별 실제 pandas dtype |
| `metadata` | Dict[str,Any] | XLSX 실제 시트명, 스트림 위치 복원 여부 등 |
| `parameters` | Dict[str,Any] | 적용한 전체 Profile |
| `warnings` | List[str] | 중첩 값·수식 캐시·seek 불가 스트림 등 조건과 한계 |

## Feature·모형 연결

파일 로딩은 평가 분할 전의 입력 해석입니다. 값의 의미·타깃·시간·그룹을 확인한 뒤 분석 설계를 정합니다. 결측 대치·스케일링·인코딩·변수 선택은 학습/CV 안에서 기존 Feature 모듈로 적합합니다. 임의 행 제거, 자동 결측 대치, 샘플링은 로더가 수행하지 않습니다. 구분자 표는 pandas의 기본 NA 토큰 해석과 타입 추론을 사용하므로 문자열 NA나 선행0이 중요한 열은 자료 계약과 dtype을 확인하세요. TXT lines 모드는 원 문자열을 유지합니다.

참조 [선형회귀](../src/MachineLearning/skeleton/linear_regression.py)와 [XGBoost](../src/MachineLearning/skeleton/xgboost.py) 스켈레톤에 사용자 파일 인자를 연결했습니다. `source` 생략 시 기존 UCI 데이터·파이프라인·결과 명세를 그대로 사용합니다. 모델 overrides와 data_overrides는 별개입니다.

```python
from src.MachineLearning.skeleton.linear_regression import main

result = main(
    source="measurements.xlsx",
    target_column="outcome",  # 분석 어댑터가 target으로 변경; 충돌/결측 타깃은 오류
    data_overrides={"sheet_name": "관측자료", "required_columns": ["outcome"]},
    overrides={"scaling": "standard"},
)
```

이 두 스켈레톤은 기존 수치 Feature 입력 조건을 유지합니다. 범주·중첩·텍스트를 자동으로 숫자로 만들지 않습니다. 그 밖의 도메인 스켈레톤은 고유 시간/시퀀스/거래 입력 계약을 유지하므로, `read_table` 결과에서 해당 입력을 명시적으로 구성해 전달합니다. 사용자 파일의 `target_column` 설정은 원본 파일을 수정하지 않습니다. `cache_dir`는 UCI 전용으로 사용자 source와 동시에 지정할 수 없습니다.

## 의존성과 실패 조건

기존 분석 환경에 형식별 선택 의존성을 추가합니다. JSON/JSONL/TXT/CSV/TSV는 기본 환경만 사용하며, YAML은 PyYAML, XML은 defusedxml, XLSX는 openpyxl을 실제 호출할 때 로드합니다.

```bash
python -m pip install -r requirements-data-loading.txt
python -m unittest discover -s tests -p test_data_loading.py -v
python -m src.datasets.skeleton.tabular
```

[선택 의존성](../requirements-data-loading.txt)은 Python3.7/3.12 환경을 구분합니다. 누락된 패키지는 설치 파일을 가리키는 ImportError로 알립니다. 파일 누락·권한 오류는 OSError, 인코딩 오류는 UnicodeError, 잘못된 형식/구조/스키마·빈 자료는 TypeError/ValueError로 전달합니다. XML/YAML 파싱 실패와 유효하지 않은 XLSX는 ValueError로 알립니다. 경고를 예외로 감추거나 빈 표로 대체하지 않습니다.
