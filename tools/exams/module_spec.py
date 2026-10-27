"""
================================================================================
module_spec.py - 저장소 Python 모듈 헤더 표준화·전수 검사. 실행 AST와 기존 본문 설명을 보존합니다.
================================================================================
설명:
    - 저장소 Python 모듈 헤더 표준화·전수 검사. 실행 AST와 기존 본문 설명을 보존합니다.
분석 가정:
    - 저장소 경로와 입력 명세를 기준으로 실행합니다. 생성·검사 대상과 외부 의존성은 각 실행 함수의 조건을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

import argparse
import ast
from collections import Counter
from io import StringIO
import json
from pathlib import Path
import re
import subprocess
import tokenize

ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-10-27"
AUTHOR = "김재현(Finn)"
EMAIL = "penguin.klg@gmail.com"
RULE = "=" * 80
LABELS = ("설명:", "분석 가정:", "작성 정보:", "변경 이력 (Revision History):")


def _section(doc, name, next_names):
    match = re.search(
        re.escape(name)
        + r"\s*\n?(.*?)(?="
        + "|".join(re.escape(n) for n in next_names)
        + r"|\Z)",
        doc,
        re.S,
    )
    return match.group(1).strip() if match else ""


def _default_description(path):
    if path.name == "__init__.py":
        parent = (
            path.parent.relative_to(ROOT)
            if path.is_absolute() and ROOT in path.parents
            else path.parent
        )
        return parent.as_posix() + " 패키지의 가져오기 경로를 구성합니다."
    if "tests" in path.parts:
        return path.stem + "의 입력·결과·호환성 계약을 검증합니다."
    return path.stem + "의 실행 진입점과 보조 기능을 제공합니다."


def _assumption(path):
    if path.name == "__init__.py":
        return "패키지 초기화와 공개 가져오기 경로를 정의합니다. 실제 계산의 입력 조건은 하위 모듈 명세를 따릅니다."
    if "tests" in path.parts:
        return "테스트의 고정 입력·합성 자료와 명시한 허용오차를 사용합니다. 시험 데이터의 성능 보장이 아닙니다."
    if "skeleton" in path.parts:
        return "자료의 열·타깃·분할 조건을 확인하고, 학습 전처리는 학습 구간에서만 적합합니다. 실행은 명시한 진입점에서 수행합니다."
    if "기출문제" in path.parts:
        return "제공 자료·문항 명세·실행 환경을 따릅니다. 복원·창작 연습 자료의 수치를 공식 기출 정답으로 해석하지 않습니다."
    if "tools" in path.parts:
        return "저장소 경로와 입력 명세를 기준으로 실행합니다. 생성·검사 대상과 외부 의존성은 각 실행 함수의 조건을 따릅니다."
    return "입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다."


def _doc_span(source, tree):
    if ast.get_docstring(tree, clean=False) is None:
        return None
    token = next(
        t
        for t in tokenize.generate_tokens(StringIO(source).readline)
        if t.type == tokenize.STRING
    )
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[: token.start[0] - 1])) + token.start[1]
    end = sum(map(len, lines[: token.end[0] - 1])) + token.end[1]
    return start, end


def format_python(source, path):
    """각 파일의 기존 설명·가정·작성일·이력을 보존하면서 표준 헤더를 구성합니다."""
    path = Path(path)
    tree = ast.parse(source)
    doc = ast.get_docstring(tree, clean=False) or ""
    # 재생성된 실행 wrapper는 이미 표준화된 기존 파일의 이력을 보존합니다.
    if not all(label in doc for label in LABELS) and path.exists():
        previous = (
            ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False)
            or ""
        )
        if all(label in previous for label in LABELS):
            old_span = _doc_span(source, tree)
            raw = (
                '"""'
                + previous.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
                + '"""'
            )
            if old_span:
                source = source[: old_span[0]] + raw + source[old_span[1] :]
            else:
                source = raw + "\n" + source
            tree = ast.parse(source)
            doc = ast.get_docstring(tree, clean=False) or ""
    lines = [
        line.strip()
        for line in doc.splitlines()
        if line.strip() and not set(line.strip()) <= set("=-")
    ]
    heading = next(
        (line.split(".py -", 1)[1].strip() for line in lines if ".py -" in line), None
    )
    short = heading or (
        lines[0] if lines and not lines[0].endswith(":") else _default_description(path)
    )
    desc = _section(doc, "설명:", ("분석 가정:", "작성 정보:", "변경 이력"))
    assumptions = _section(doc, "분석 가정:", ("작성 정보:", "변경 이력"))
    if not desc:
        # 짧은 모듈 문서의 실행 방법·제약 등은 설명 부분에 모두 보존합니다.
        desc = (
            "\n".join("    - " + line for line in lines)
            if lines and not heading
            else "    - " + short
        )
    else:
        desc = "\n".join(
            "    " + line.strip() for line in desc.splitlines() if line.strip()
        )
    if not assumptions:
        assumptions = "    - " + _assumption(path)
    else:
        assumptions = "\n".join(
            "    " + line.strip() for line in assumptions.splitlines() if line.strip()
        )
    created = re.search(r"작성일:\s*(\d{4}-\d\d-\d\d)", doc)
    date = created.group(1) if created else DATE
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)', source)
    version = version.group(1) if version else "1.0.0"
    history = [
        line.strip()
        for line in doc.splitlines()
        if re.match(r"^\s*\d{4}-\d\d-\d\d\s+", line)
    ]
    entry = DATE + "  " + AUTHOR + "     v" + version + "    모듈 명세 헤더 표준화"
    if not any("모듈 명세 헤더 표준화" in line for line in history):
        history.append(entry)
    license_match = re.search(r"라이선스:\s*([^\n]+)", doc)
    license_text = license_match.group(1).strip() if license_match else "Proprietary"
    header = (
        "\n"
        + RULE
        + "\n"
        + path.name
        + " - "
        + short
        + "\n"
        + RULE
        + "\n설명:\n"
        + desc
        + "\n분석 가정:\n"
        + assumptions
        + "\n작성 정보:\n    - 작성자: "
        + AUTHOR
        + " ("
        + EMAIL
        + ")\n    - 작성일: "
        + date
        + "\n    - 라이선스: "
        + license_text
        + "\n변경 이력 (Revision History):\n    DATE        AUTHOR          VERSION   DESCRIPTION\n"
        + "\n".join("    " + line for line in history)
        + "\n"
        + RULE
        + "\n"
    )
    raw = '"""' + header.replace("\\", "\\\\").replace('"""', '\\"\\"\\"') + '"""'
    span = _doc_span(source, tree)
    if span:
        return source[: span[0]] + raw + source[span[1] :]
    # shebang/encoding declarations retain their required first/second positions.
    lines = source.splitlines(keepends=True)
    pos = 0
    for i, line in enumerate(lines[:2]):
        if line.startswith("#!") or re.search(r"coding[:=]\s*[-\w.]+", line):
            pos = i + 1
    return "".join(lines[:pos]) + raw + "\n\n" + "".join(lines[pos:])


def write_python(path, source, **kwargs):
    """생성 Python 파일에도 동일 헤더를 적용해 재생성 시 누락을 막습니다."""
    path = Path(path)
    return path.write_text(format_python(source, path), **kwargs)


def files():
    """Git 추적 파일과 새 비무시 파일을 모두 분류 대상으로 반환합니다."""
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(ROOT),
        )
        .decode()
        .split("\0")
    )
    return sorted({ROOT / name for name in names if name and (ROOT / name).is_file()})


def inspect_files(paths):
    """전체 형식의 역할을 분류하고 Python 모듈 헤더 누락을 나열합니다."""
    counts = Counter()
    failures = []
    python = []
    for path in paths:
        kind = (
            "python"
            if path.suffix == ".py"
            else (
                "notebook"
                if path.suffix == ".ipynb"
                else (
                    "documentation"
                    if path.suffix == ".md"
                    else (
                        "configuration"
                        if path.suffix in (".json", ".yml", ".yaml", ".toml")
                        else "data-or-resource"
                    )
                )
            )
        )
        counts[kind] += 1
        if kind != "python":
            continue
        text = path.read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(text)) or ""
        required = (
            *LABELS,
            "작성자: " + AUTHOR + " (" + EMAIL + ")",
            "작성일:",
            "라이선스:",
            "DATE        AUTHOR",
        )
        missing = [value for value in required if value not in doc]
        if not doc.startswith(RULE + "\n" + path.name + " - "):
            missing.append("구분선/파일명 제목")
        if missing:
            failures.append(
                dict(path=path.relative_to(ROOT).as_posix(), missing=missing)
            )
        python.append(path.relative_to(ROOT).as_posix())
    return dict(counts=dict(counts), python_files=python, failures=failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paths = files()
    changed = []
    if args.write:
        for path in paths:
            if path.suffix != ".py":
                continue
            before = path.read_text(encoding="utf-8")
            after = format_python(before, path)
            if before != after:
                path.write_text(after, encoding="utf-8")
                changed.append(str(path.relative_to(ROOT)))
    report = inspect_files(paths)
    print(
        json.dumps(
            dict(
                counts=report["counts"],
                changed=len(changed),
                failures=report["failures"],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
