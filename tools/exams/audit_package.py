"""
================================================================================
audit_package.py - 출처 질문의 누락, 구문·실행 출력, 파일 해시 및 문제지 경계를 검증합니다.
================================================================================
설명:
    - 출처 질문의 누락, 구문·실행 출력, 파일 해시 및 문제지 경계를 검증합니다.
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

from pathlib import Path
import argparse
import ast
import gzip
import hashlib
import json
import os
import subprocess
import sys
import nbformat
import fitz

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "기출문제"
parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
# 이 검사는 기존 상황형22묶음의 계약입니다. 이후 독립 모의는 각 전용 검사로 검증합니다.
scenario = json.loads((ROOT / "상황형_문항명세.json").read_text())
notebooks = [next((ROOT / folder).glob("*풀이.ipynb")) for folder in sorted(scenario)]
if args.execute:
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg"
    )
    for path in notebooks:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("execute_notebooks.py")),
                str(path),
            ],
            check=True,
            env=env,
        )

additional = json.loads((ROOT / "추가_문항명세.json").read_text())
mock_source = json.loads((REPO / "tools/exams/mock_source_questions.json").read_text())
for exam, original in mock_source.items():
    items = [
        q
        for s in additional["모의고사/" + exam + "회"]["sections"]
        for q in s["questions"]
    ]
    for cell, src in original.items():
        assert any(q["id"] == src["id"] and q["text"] == src["text"] for q in items), (
            exam,
            cell,
        )

rows = []
for path in notebooks:
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    ids = book.metadata.adp.subquestions
    assert len(ids) == len(set(ids)), path
    code = [c for c in book.cells if c.cell_type == "code"]
    for c in code:
        ast.parse(c.source, feature_version=(3, 7))
        assert c.execution_count is not None, path
        assert not any(o.output_type == "error" for o in c.outputs), path
    pdf = next(path.parent.glob("*문제지.pdf"))
    doc = fitz.open(pdf)
    for page in doc:
        for x0, y0, x1, y1, *_ in page.get_text("words"):
            assert x0 >= 40 and x1 <= 558 and y1 <= 822, (pdf, x0, y0, x1, y1)
    rows.append(
        {
            "folder": str(path.parent.relative_to(ROOT)),
            "questions": len(ids),
            "code_cells": len(code),
            "pdf_pages": len(doc),
            "figures": sum(
                "image/png" in o.get("data", {}) for c in code for o in c.outputs
            ),
            "status": "passed",
            "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        }
    )

# Preserve original uncompressed CSV byte identity when gzip is used in git.
source = json.loads((ROOT / "데이터_출처.json").read_text())
for entry in source:
    path = ROOT / (str(entry["exam"]) + "회") / "data" / entry["file"]
    if path.exists():
        original_bytes = path.read_bytes()
    else:
        original_bytes = gzip.decompress(Path(str(path) + ".gz").read_bytes())
    assert hashlib.sha256(original_bytes).hexdigest() == entry["sha256"], path
# Record exact current file bytes as well as uncompressed identities.
files = []
for path in sorted(ROOT.rglob("data/*")):
    if path.is_file():
        data = path.read_bytes()
        decoded = gzip.decompress(data) if path.suffix == ".gz" else data
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "uncompressed_sha256": hashlib.sha256(decoded).hexdigest(),
            }
        )

result = {
    "all_passed": True,
    "notebooks": rows,
    "dataset_files": files,
    "total_notebooks": len(rows),
    "total_questions": sum(r["questions"] for r in rows),
    "total_code_cells": sum(r["code_cells"] for r in rows),
    "total_pdf_pages": sum(r["pdf_pages"] for r in rows),
    "total_figures": sum(r["figures"] for r in rows),
    "notes": [
        "Actual sequential isolated Python-process execution outputs retained.",
        "Python 3.7 grammar checked; Python 3.7 runtime uses GitHub CI for modules.",
        "PDF coordinates checked; visual review performed separately.",
    ],
}
expected_questions = 57 + sum(
    len(s["questions"]) for exam in additional.values() for s in exam["sections"]
)
assert (
    len(rows) == len(additional) + 5 and result["total_questions"] == expected_questions
)
(ROOT / "검증결과.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(
    {
        k: v
        for k, v in result.items()
        if k not in ("notebooks", "dataset_files", "notes")
    }
)
