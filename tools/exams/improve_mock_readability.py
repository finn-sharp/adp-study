"""모의16묶음의 자료 안내·문제지·풀이·실행 해설을 같은 독자 기준으로 정리합니다.

지문·수식·배점·데이터·수치 답안은 유지하며 표현과 배치만 다룹니다.
생성기 뒤에서 호출하거나 python tools/exams/improve_mock_readability.py로 실행합니다.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import re
import sys
import black
import nbformat
from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer, KeepTogether
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import render_question_papers as paper
from mock_reading import reading, markdown


DELIVERABLES = {
    "reg_design": "관측·열 역할표, 결측 요약, 학습/평가 원행 위치",
    "reg_baseline": "학습·CV R², 계수 t·전체 F, AIC·BIC와 추론 조건",
    "reg_diagnostics": "가정별 검정표·그림, VIF·Cook의 해석 대상",
    "reg_selection": "AIC/BIC 선택 경로와 남은 열, 동일 행·반응 확인",
    "reg_tuning": "동일 CV 후보 비교표, 선택 설정, 적용 조건",
    "reg_final": "같은 보류 행의 기준/선택 점수·ΔR²와 선택 후 진단",
    "cls_design": "사건 정의, 양성 라벨, 클래스 수, 층화 분리 위치",
    "cls_tuning": "무보완/보완 CV AUC 비교와 선택 파라미터·fold",
    "cls_logit": "자연 발생률 GLM 계수·오즈비, 추가항 우도비 검정",
    "cls_final": "ROC·PR·calibration 그림, AUC·AP·Brier·고정 임계값 혼동행렬",
    "cls_decision": "FN/FP 비용 계산과 별도 운영 검증 제안",
    "ts_design": "시간·horizon·과거/미래 경계와 과거 ADF 진단",
    "ts_tuning": "rolling 구간·차수별 오차·수렴 기록과 선택 차수",
    "ts_diagnostics": "AIC·BIC, 자유도 보정 잔차 검정과 해석",
    "ts_final": "같은 미래의 선택/naive 비교, MASE 분모와 예측구간",
}

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "기출문제/모의고사"


def folders():
    return (
        [ROOT / (str(i) + "회") for i in range(1, 6)]
        + [ROOT / "생존분석_통계"]
        + [ROOT / "응용10회" / ("%02d회" % i) for i in range(1, 11)]
    )


def specification(folder):
    if folder.name in ["1회", "2회", "3회", "4회"]:
        value = json.loads((REPO / "기출문제/상황형_문항명세.json").read_text())[
            "모의고사/" + folder.name
        ]
        sections = [
            dict(
                s, context="\n".join(s.get("info", [])) + "\n" + s.get("condition", "")
            )
            for s in value["sections"]
        ]
        return "모의고사 " + folder.name, sections, None
    value = json.loads((folder / "문항명세.json").read_text())
    sections = []
    for original in value["sections"]:
        sec = dict(original)
        sec["questions"] = [
            dict(q) if isinstance(q, dict) else dict(id=q[0], text=q[1], points=q[2])
            for q in sec["questions"]
        ]
        sections.append(sec)
    number = value.get("number")
    title = (
        ("응용 모의 제%d회 | " % number + value["title"])
        if number
        else (
            "생존분석 통계 모의"
            if folder.name == "생존분석_통계"
            else "모의고사 5회 | 생산 의사결정"
        )
    )
    return title, sections, number


def format_code(source):
    return black.format_str(
        source,
        mode=black.Mode(target_versions={black.TargetVersion.PY37}, line_length=88),
    ).rstrip()


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def task_list(question):
    # 요구사항의 문장을 분리하되 원 수치·방법·가정을 수정하지 않습니다.
    return "\n".join(
        str(i) + ". " + text for i, text in enumerate(sentences(question["text"]), 1)
    )


def render(folder, title, sections, number):
    paper.ST = paper.styles()
    paper.ST["small"].fontSize = 10
    paper.ST["small"].leading = 17
    paper.ST["table"].fontSize = 9
    paper.ST["table"].leading = 14
    story = [
        paper.p("분석전문가(ADP) 실기", "title"),
        paper.p(title, "subtitle"),
        paper.section_heading("학습 안내"),
        Spacer(1, 10),
        paper.p(
            "대상 독자: 데이터 분석 지식은 있으나 이 산업·업무를 처음 접하는 학습자입니다. 각 대문항의 배경과 변수 사전을 먼저 읽으십시오."
        ),
        paper.p(
            "본 자료는 공식 기출 원문·공식 배점·정답이 아닌 학습용 모의입니다. 기존 자료·후기·창작 합성 여부와 상세 출처는 각 폴더 안내를 따릅니다."
        ),
        paper.p(
            "읽는 순서: 상황과 관측 단위 → 변수 사전 → 고정 분석 조건 → 수행 과제 → 별도 실행 풀이·해설. 대문항 간 자료는 지문에서 결합을 요구하지 않으면 별개입니다."
        ),
        paper.ruled_table(
            [["대문항", "자료", "문항 수"]]
            + [
                [s["title"], s.get("data", "지문 자료"), len(s["questions"])]
                for s in sections
            ],
            [paper.WIDTH * 0.46, paper.WIDTH * 0.38, paper.WIDTH * 0.16],
        ),
        Spacer(1, 15),
        paper.p(
            "답안 작성: 방법 선택 근거·가정·실제 수치·판정·상황 결론을 구분합니다. 통계적 비기각을 가정의 증명으로 쓰지 않고, 예측 점수와 업무 의사결정을 구분합니다."
        ),
        paper.p(
            "자료 사용: 목표·식별자·시간·그룹의 역할을 구분합니다. 학습용 전처리·선택·sampling·튜닝은 학습/CV 안에서 수행하고 최종 평가 자료는 선택에 사용하지 않습니다.",
            "small",
        ),
    ]
    if number:
        story.append(
            paper.p(
                "15문항·100점·권장 180분: 회귀 36점 / 분류 40점 / 시계열 24점. 후보·분할·시드는 실행설정.json과 아래 지문을 따릅니다.",
                "small",
            )
        )
    for sec in sections:
        guide = reading(folder, sec, number)
        story += [
            PageBreak(),
            paper.section_heading(sec["title"] + " | 자료 이해"),
            Spacer(1, 12),
            paper.p("상황과 관측 단위", "label"),
            paper.p(guide["background"]),
            paper.p(guide["unit"] + " " + guide["decision"]),
            paper.p(
                "제공 자료: data/"
                + sec.get("data", "")
                + (" | " + sec["level"] if sec.get("level") else ""),
                "small",
            ),
            Spacer(1, 6),
            paper.ruled_table(
                [["열", "의미·역할"]] + guide["fields"],
                [paper.WIDTH * 0.31, paper.WIDTH * 0.69],
            ),
            Spacer(1, 12),
            paper.p("고정 분석 조건", "label"),
            paper.p(sec.get("context", ""), "small"),
        ]
        if sec.get("table"):
            rows = sec["table"]
            story += [
                Spacer(1, 6),
                paper.ruled_table(rows, [paper.WIDTH / len(rows[0])] * len(rows[0])),
            ]
        # 자료 설명과 수행 과제를 분리해 학습자가 오가며 읽도록 합니다.
        story.append(PageBreak())
        story += [paper.section_heading(sec["title"] + " | 수행 과제"), Spacer(1, 12)]
        for q in sec["questions"]:
            label = q["id"] + (
                "  [" + str(q["points"]) + "점]" if "points" in q else ""
            )
            text = "\n".join(sentences(q["text"]))
            story.append(
                KeepTogether(
                    [paper.p(label, "label"), paper.p(text, "body"), Spacer(1, 10)]
                )
            )

    def furniture(c, doc):
        c.saveState()
        c.setFont("ExamGothic", 8)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(paper.MARGIN, 810, "ADP 학습용 모의 | " + folder.name)
        c.drawRightString(595 - paper.MARGIN, 810, "학습 자료 / 공식 시험지 아님")
        c.line(paper.MARGIN, 802, 595 - paper.MARGIN, 802)
        c.line(paper.MARGIN, 48, 595 - paper.MARGIN, 48)
        c.drawCentredString(297.5, 34, "- %d -" % doc.page)
        c.restoreState()

    path = next(folder.glob("*문제지.pdf"))
    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=paper.MARGIN,
        rightMargin=paper.MARGIN,
        topMargin=64,
        bottomMargin=65,
        title=title,
        author="ADP Study",
        invariant=1,
    ).build(
        story,
        onFirstPage=furniture,
        onLaterPages=furniture,
    )


def brief(value):
    if value is None:
        return "미정의/미제공"
    if isinstance(value, bool):
        return "참" if value else "거짓"
    if isinstance(value, float):
        return format(value, ".6g")
    text = str(value).replace("\n", " ").replace("|", "／")
    return text if len(text) < 180 else text[:170] + "…(전체 결과 참조)"


def table_markdown(data):
    if not hasattr(data, "columns"):
        return str(data)
    sample = data.head(16)
    return (
        "| "
        + " | ".join(map(str, sample.columns))
        + " |\n| "
        + " | ".join(["---"] * len(sample.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(brief(v) for v in row) + " |"
            for row in sample.itertuples(index=False, name=None)
        )
        + (
            "\n\n총 %d행 중 앞 16행입니다. 전체 정밀값은 수치 답안을 확인하십시오."
            % len(data)
            if len(data) > 16
            else ""
        )
    )


def enrich_answers(folder, title, sections, number, book):
    lines = [
        "# " + title + " 해설",
        "",
        "독자 기준: 분석 전공 지식을 전제로 하되 도메인 배경은 아래에서 설명합니다. 수치는 실행 결과를 읽어 표시한 값이며 상세 정밀값은 실행결과의 JSON과 노트북 출력에 있습니다.",
    ]
    values = {}
    for name in ["답안수치.json", "가설검정_답안.json"]:
        path = folder / "실행결과" / name
        if path.exists():
            for row in json.loads(path.read_text()):
                values.setdefault(row["question"], []).append(row)
    cells = {
        c.source.splitlines()[0][4:].strip(): c
        for c in book.cells
        if c.cell_type == "markdown" and c.source.startswith("### ")
    }
    if number:
        common = ROOT / "응용10회/공통"
        sys.path.insert(0, str(common))
        from presentation import tables, READING, conclusion
    for sec in sections:
        lines += ["", "## " + sec["title"], "", markdown(reading(folder, sec, number))]
        for q in sec["questions"]:
            cell = cells[q["id"]]
            lines += [
                "",
                "### " + q["id"],
                "",
                "**수행 과제**",
                "",
                task_list(q),
                "",
                "**풀이 근거와 순서**",
                "",
                cell.metadata.get(
                    "readability_reason",
                    "문항의 조건에 맞춰 실행 결과와 가정을 연결합니다.",
                ),
            ]
            rows = values.get(q["id"], [])
            if number and rows:
                for label, data in tables(q["task"], rows[0]["result"]):
                    lines += ["", "**" + label + "**", "", table_markdown(data)]
                lines += [
                    "",
                    "**결과를 읽는 기준**",
                    "",
                    conclusion(q["task"], rows[0]["result"]),
                ]
            elif rows:
                for row in rows:
                    if "result" in row:
                        payload = row["result"]
                        # 생산/생존은 record(question, **values) 방식도 지원합니다.
                    else:
                        payload = {k: v for k, v in row.items() if k != "question"}
                    import pandas as pd

                    scalars = {
                        k: v
                        for k, v in payload.items()
                        if v is None or isinstance(v, (str, bool, int, float))
                    }
                    if scalars:
                        lines += [
                            "",
                            "**실제 실행 수치**",
                            "",
                            table_markdown(
                                pd.DataFrame(
                                    [{"항목": k, "값": v} for k, v in scalars.items()]
                                )
                            ),
                        ]
                    for key, val in payload.items():
                        if isinstance(val, list) and val and isinstance(val[0], dict):
                            lines += [
                                "",
                                "**" + key + "**",
                                "",
                                table_markdown(pd.DataFrame(val)),
                            ]
                        elif isinstance(val, dict) and {
                            "data",
                            "index",
                            "columns",
                        } <= set(val):
                            lines += [
                                "",
                                "**" + key + "**",
                                "",
                                table_markdown(
                                    pd.DataFrame(
                                        val["data"],
                                        columns=val["columns"],
                                        index=val["index"],
                                    )
                                ),
                            ]
                        elif isinstance(val, dict):
                            lines += [
                                "",
                                "**" + key + "**",
                                "",
                                table_markdown(
                                    pd.DataFrame(
                                        [
                                            {"항목": k, "값": brief(v)}
                                            for k, v in val.items()
                                        ]
                                    )
                                ),
                            ]
                        elif isinstance(val, list):
                            lines += [
                                "",
                                "**" + key + "**",
                                "",
                                table_markdown(pd.DataFrame({"값": val})),
                            ]

            else:
                # JSON 기록이 없는 탐색/군집/문서 문항도 수치를 생략하지 않고 해당 셀의 출력으로 연결합니다.
                index = book.cells.index(cell)
                code = book.cells[index + 1] if index + 1 < len(book.cells) else None
                outputs = []
                if code and code.cell_type == "code":
                    for out in code.outputs:
                        text = out.get("text") or out.get("data", {}).get(
                            "text/plain", ""
                        )
                        if text and "Figure size" not in text:
                            outputs.append(text)
                if outputs:
                    lines += [
                        "",
                        "**실행 결과 읽기**",
                        "",
                        "```text",
                        "\n".join(outputs)[:5000],
                        "```",
                    ]
            lines += [
                "",
                "**답안 확인**",
                "",
                "관측 단위·변수 뜻을 유지했는지, 실제 수치가 위 설명을 뒷받침하는지, 가정과 자료 범위를 넘어선 결론이 없는지 확인합니다.",
            ]
    (folder / "해설.md").write_text("\n".join(lines) + "\n")


def improve_bundle(folder, render_pdf=True, answers_only=False):
    folder = Path(folder)
    title, sections, number = specification(folder)
    path = next(
        iter(list(folder.glob("*풀이.ipynb")) + list(folder.glob("*해설.ipynb")))
    )
    book = nbformat.read(path, 4)
    if not answers_only:
        book.cells[0].source = (
            "# "
            + title
            + " 풀이\n\n자료 이해 → 고정 분석 조건 → 문항별 풀이 근거 → 실행 코드 → 실제 결과의 순서로 읽으십시오. 전체를 위에서 아래로 실행합니다. 변수의 산업적 의미는 각 절의 사전을 따르며, 공식 정답이나 실제 현장 성능을 뜻하지 않습니다.\n\n실제 결과를 모은 [해설](해설.md)과 [자료 사전](자료안내.md)을 함께 사용하십시오."
        )
        section_iter = iter(sections)
        by_id = {q["id"]: q for sec in sections for q in sec["questions"]}
        guides = [
            "# " + title + " 자료 안내",
            "",
            "도메인 용어를 익힌 뒤 통계·모형 분석에 들어가기 위한 안내입니다. 원시 자료를 해석하는 사전이며 분석 결과나 정답을 뜻하지 않습니다.",
        ]
        for sec in sections:
            guides += [
                "",
                "## " + sec["title"],
                "",
                markdown(reading(folder, sec, number)),
            ]
        (folder / "자료안내.md").write_text("\n".join(guides) + "\n")
        for cell in book.cells:
            if cell.cell_type == "code":
                cell.source = format_code(cell.source)
            elif cell.source.startswith("## ") and not cell.source.startswith(
                "## 제출"
            ):
                sec = next(section_iter)
                cell.source = (
                    "## "
                    + sec["title"]
                    + "\n\n"
                    + markdown(reading(folder, sec, number))
                    + "\n\n**고정 분석 조건**\n\n"
                    + sec.get("context", "")
                )
            elif cell.source.startswith("### "):
                qid = cell.source.splitlines()[0][4:].strip()
                q = by_id[qid]
                if "readability_reason" not in cell.metadata:
                    source = cell.source
                    for marker in [
                        "**선택 근거·가정·해석**",
                        "**판단 및 풀이 근거**",
                        "**풀이 및 해석**",
                    ]:
                        if marker in source:
                            source = (
                                source.split(marker, 1)[1]
                                .split("**실행 답안**")[0]
                                .strip()
                            )
                            break
                    cell.metadata.readability_reason = source
                reason = cell.metadata.readability_reason
                cell.source = (
                    "### "
                    + qid
                    + "\n\n**문제"
                    + (" (" + str(q["points"]) + "점)" if "points" in q else "")
                    + "**\n\n"
                    + q["text"]
                    + (
                        "\n\n**핵심 산출물**\n\n" + DELIVERABLES[q["task"]]
                        if number
                        else ""
                    )
                    + "\n\n**풀이 근거와 읽는 순서**\n\n"
                    + "\n\n".join(sentences(reason))
                    + "\n\n**실행·해석**\n\n다음 셀은 해당 결과를 계산합니다. 수치와 그림을 위 가정·판단 기준에 연결하고 과제 범위를 넘는 인과·현장 결론은 피합니다."
                )
        book.metadata.adp["reader_standard"] = (
            "analysis major without domain background"
        )
        nbformat.write(book, path)
        wrapper = '"""이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이."""\nfrom pathlib import Path\nimport os\nos.chdir(Path(__file__).resolve().parent)\n\n'
        chunks = []
        for cell in book.cells:
            if cell.cell_type == "markdown" and cell.source.startswith("### "):
                chunks.append("# 문항 " + cell.source.splitlines()[0][4:])
            elif cell.cell_type == "code":
                chunks.append(cell.source)
        (folder / "분석코드.py").write_text(wrapper + "\n\n".join(chunks) + "\n")
        readme = folder / "README.md"
        text = readme.read_text() if readme.exists() else "# " + title + "\n"
        marker = "\n<!-- reader-guide -->"
        text = (
            text.split(marker)[0].rstrip()
            + marker
            + "\n\n도메인 배경 없이 시작하려면 [자료 안내](자료안내.md) → 문제지 → 실행 노트북 → [수치 해설](해설.md) 순서로 읽으십시오. 분석코드.py는 같은 코드를 일반 Python에서 실행합니다.\n"
        )
        if folder.name in ("1회", "2회", "3회", "4회"):
            text += "\n출처와 원 조건: [추가 문항 명세](../../추가_문항명세.json), [상황형 명세](../../상황형_문항명세.json). 공식 시험 원문·정답으로 표시하지 않습니다.\n"
        readme.write_text(text)
        if render_pdf:
            render(folder, title, sections, number)
    enrich_answers(folder, title, sections, number, book)
    print("readability:", folder.relative_to(REPO))


def improve_all(answers_only=False):
    for folder in folders():
        improve_bundle(folder, answers_only=answers_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-only", action="store_true")
    args = parser.parse_args()
    improve_all(args.answers_only)
