"""모의21묶음의 독자 안내·변수 전수 설명·지문 보존·실행 출력·PDF 경계를 검사합니다."""

import ast
import csv
import json
import re
from pathlib import Path
import fitz
import nbformat
from mock_reading import reading

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "기출문제/모의고사"


def audit():
    assert not (ROOT / "응용10회").exists(), "종합 모의는 1~20회로 통합합니다."
    for readme in (REPO / "README.md", ROOT / "README.md"):
        listed = re.findall(r"^\| \[(\d+)회\]\(", readme.read_text(), re.M)
        assert listed == [str(i) for i in range(1, 21)], readme
    for i in range(1, 21):
        folder = ROOT / (str(i) + "회")
        assert (folder / ("모의고사_%d회_문제지.pdf" % i)).is_file()
        assert (folder / ("모의고사_%d회_풀이.ipynb" % i)).is_file()
    folders = [ROOT / (str(i) + "회") for i in range(1, 21)] + [ROOT / "생존분석_통계"]
    legacy = json.loads((REPO / "기출문제/상황형_문항명세.json").read_text())
    records = []
    for folder in folders:
        number = None
        if folder.name in ["1회", "2회", "3회", "4회"]:
            sections = legacy["모의고사/" + folder.name]["sections"]
        else:
            spec = json.loads((folder / "문항명세.json").read_text())
            sections = spec["sections"]
            number = spec.get("number")
        book = nbformat.read(
            next(
                iter(
                    list(folder.glob("*풀이.ipynb")) + list(folder.glob("*해설.ipynb"))
                )
            ),
            4,
        )
        assert (
            book.metadata.adp.reader_standard
            == "analysis major without domain background"
        )
        prose = "\n".join(c.source for c in book.cells if c.cell_type == "markdown")
        guides = (folder / "자료안내.md").read_text()
        answers = (folder / "해설.md").read_text()
        script = (folder / "분석코드.py").read_text()
        pdf = fitz.open(next(folder.glob("*문제지.pdf")))
        compact = lambda text: re.sub(r"\s+", "", text)
        pdftext = compact("".join(p.get_text() for p in pdf))
        count = 0
        for sec in sections:
            guide = reading(folder, sec, number)
            for key in ("background", "unit", "decision"):
                assert guide[key] in prose and guide[key] in guides
                assert compact(guide[key]) in pdftext, (folder, key)
            for name, meaning in guide["fields"]:
                assert name in prose and name in guides and meaning in guides
                assert compact(meaning) in pdftext, (folder, name)
            for question in sec["questions"]:
                qid, text = (
                    (question["id"], question["text"])
                    if isinstance(question, dict)
                    else question[:2]
                )
                assert text in prose and compact(text) in pdftext, (folder, qid)
                assert "### " + qid in answers
                count += 1
        for cell in book.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source, feature_version=(3, 7))
                assert cell.source in script
                assert cell.execution_count is not None and not any(
                    o.output_type == "error" for o in cell.outputs
                )
        for page in pdf:
            assert (
                abs(page.rect.width - 595.276) < 1
                and abs(page.rect.height - 841.89) < 1
            )
            for x0, y0, x1, y1, *_ in page.get_text("words"):
                assert x0 >= 40 and x1 <= 558 and y0 >= 20 and y1 <= 822, (
                    folder,
                    x0,
                    y0,
                    x1,
                    y1,
                )
        records.append(
            dict(
                folder=str(folder.relative_to(REPO)),
                questions=count,
                pages=len(pdf),
                status="passed",
            )
        )
    result = dict(
        status="passed",
        bundles=len(records),
        questions=sum(r["questions"] for r in records),
        pdf_pages=sum(r["pages"] for r in records),
        checks=[
            "전수 변수 사전",
            "관측·도메인 설명",
            "문제/풀이 지문 보존",
            "실행 오류·독립 코드 정합성",
            "A4 좌표 경계",
        ],
        records=records,
    )
    (ROOT / "가독성_검증결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    result = audit()
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "records"}, ensure_ascii=False
        )
    )
