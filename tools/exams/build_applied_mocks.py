"""기존 공문서형 서식의 신규 응용 모의10회·150문항·코드·해설을 생성합니다."""

from pathlib import Path
import sys, json
import nbformat
from reportlab.platypus import SimpleDocTemplate, Spacer, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from applied_mock_specs import specification
from applied_mock_data import generate
import render_question_papers as paper

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "기출문제/모의고사"
SETUP = """from pathlib import Path
import sys
FOLDER=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0,str(FOLDER.parent/'공통'))
from analysis import Context
ctx=Context(FOLDER)
"""


def render(folder, spec):
    paper.ST = paper.styles()
    title = "모의고사 " + str(spec["exam_number"]) + "회"
    story = [
        paper.p("분석전문가(ADP) 실기", "title"),
        paper.p(title + " | " + spec["title"], "subtitle"),
        paper.ruled_table(
            [["성명", "작성일", "배점"], ["", "____년 __월 __일", "100점 / 15문항"]],
            [paper.WIDTH / 3] * 3,
        ),
        Spacer(1, 12),
        paper.section_heading("응시 안내"),
        Spacer(1, 10),
        paper.p(
            "기존 공문서형·상황형 학습 자료와 같은 구성의 창작 종합 모의입니다. 공식 기출 원문·공식 배점·정답이 아닙니다. 권장시간180분, 회귀36점·분류40점·시계열24점입니다. 제공 CSV와 실행설정.json의 후보·분할·시드를 따르십시오. 데이터의 생성 원리를 분석 전에 정답으로 가정하지 마십시오.",
            "body",
        ),
        paper.p(
            "검정은 방법 선정 근거·가정·가설·통계량·자유도·p값·판정·상황 결론을 구분합니다. 전처리·변수선택·샘플링·튜닝은 학습/CV 안에서만 수행하고 최종평가는 봉인합니다. 정보량·R²·p값·예측구간의 의미를 혼동하지 마십시오. 개선이 없거나 가정이 남아도 실제 결과를 보고합니다.",
            "body",
        ),
        paper.p(
            "자료: 고정 합성 관측이며 실제 기업·고객 데이터가 아닙니다. 기법의 기술 근거·범위는 공통/응용_출처및설계.md와 저장소 명세를 참고하십시오. 각 대문항은 별도 자료입니다.",
            "small",
        ),
    ]
    for sec in spec["sections"]:
        story += [
            PageBreak(),
            paper.section_heading(sec["title"]),
            Spacer(1, 10),
            paper.p("자료: data/" + sec["data"], "small"),
            paper.p(sec["context"], "small"),
            Spacer(1, 8),
        ]
        for q in sec["questions"]:
            story.append(
                paper.p(
                    q["id"] + ". [" + str(q["points"]) + "점] " + q["text"], "question"
                )
            )

    def furniture(c, d):
        c.saveState()
        c.setFont("ExamGothic", 8)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(paper.MARGIN, 810, "ADP 학습용 · " + title)
        c.drawRightString(595 - paper.MARGIN, 810, "창작 학습 자료 / 공식 시험지 아님")
        c.line(paper.MARGIN, 802, 595 - paper.MARGIN, 802)
        c.line(paper.MARGIN, 48, 595 - paper.MARGIN, 48)
        c.drawCentredString(297.5, 34, "- %d -" % d.page)
        c.restoreState()

    SimpleDocTemplate(
        str(folder / ("모의고사_%d회_문제지.pdf" % spec["exam_number"])),
        pagesize=A4,
        leftMargin=paper.MARGIN,
        rightMargin=paper.MARGIN,
        topMargin=65,
        bottomMargin=65,
        title=title,
        author="ADP Study",
    ).build(story, onFirstPage=furniture, onLaterPages=furniture)


def build():
    for number in range(1, 11):
        spec = specification(number)
        folder = ROOT / ("%d회" % spec["exam_number"])
        generate(folder, spec["plan"])
        (folder / "문항명세.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
        )
        (folder / "실행설정.json").write_text(
            json.dumps(spec["plan"], ensure_ascii=False, indent=2) + "\n"
        )
        cells = [
            nbformat.v4.new_markdown_cell(
                "# 모의고사 "
                + str(spec["exam_number"])
                + "회 풀이\n\n"
                + spec["title"]
                + "\n\n공식 정답이 아닌 실제 실행 예시 답안입니다. PDF와 지문이 동일합니다. 각 문항의 가정·가설·실제 수치·판단을 함께 읽고 src 모듈 및 공통 analysis.py를 참조하십시오."
            ),
            nbformat.v4.new_code_cell(SETUP),
        ]
        codes = [SETUP]
        explanations = [
            "# 모의고사 " + str(spec["exam_number"]) + "회 해설",
            "",
            spec["title"],
            "",
            "실제 수치는 실행 노트북과 실행결과/답안수치.json에 있습니다. 정해진 가정을 충족하는 결과를 만들기 위해 관측·시드·검정을 사후 선택하지 않습니다.",
        ]
        for sec in spec["sections"]:
            cells.append(
                nbformat.v4.new_markdown_cell(
                    "## " + sec["title"] + "\n\n" + sec["context"]
                )
            )
            for q in sec["questions"]:
                code = "ctx.run(" + repr(q["id"]) + "," + repr(q["task"]) + ")"
                cells.extend(
                    [
                        nbformat.v4.new_markdown_cell(
                            "### "
                            + q["id"]
                            + "\n\n**문제 ("
                            + str(q["points"])
                            + "점)** "
                            + q["text"]
                            + "\n\n**선택 근거·가정·해석**\n\n"
                            + q["answer"]
                        ),
                        nbformat.v4.new_code_cell(code),
                    ]
                )
                codes.append("# " + q["id"] + "\n" + code)
                explanations += ["", "## " + q["id"], "", q["text"], "", q["answer"]]
        ids = [q["id"] for s in spec["sections"] for q in s["questions"]]
        book = nbformat.v4.new_notebook(
            cells=cells,
            metadata=dict(
                adp=dict(type="applied_mock", subquestions=ids),
                kernelspec=dict(
                    name="python3", display_name="Python 3", language="python"
                ),
            ),
        )
        nbformat.write(
            book, folder / ("모의고사_%d회_풀이.ipynb" % spec["exam_number"])
        )
        (folder / "분석코드.py").write_text(
            '"""해당 파일 폴더 또는 임의 cwd에서 실행 가능한 실제 모의 풀이."""\n'
            + "\n\n".join(codes)
            + "\n"
        )
        (folder / "해설.md").write_text("\n".join(explanations) + "\n")
        (folder / "README.md").write_text(
            "# 모의고사 "
            + str(spec["exam_number"])
            + "회\n\n"
            + spec["title"]
            + "\n\n15문항·100점·권장180분. 회귀36/분류40/시계열24점. 문제지 PDF·실행 풀이·분석코드·해설·세 CSV·실행 설정·수치 답안·검증을 포함합니다.\n\n이 폴더와 상위 공통 코드·저장소 src를 함께 사용합니다. 의존성·출처·생성 원리·실행법은 [상위 안내](../README.md)를 참고하세요. 이 회차의 특별 회귀 조건은 `"
            + spec["plan"]["regression_kind"]
            + "`이며 샘플링은 `"
            + spec["plan"]["sampler"]
            + "`, 분류 탐색은 `"
            + spec["plan"]["classification_search"]
            + "`입니다. 최종평가로 후보를 바꾸지 않습니다.\n"
        )
        render(folder, spec)
        from improve_mock_readability import improve_bundle

        improve_bundle(folder)
        print("built", number, flush=True)


if __name__ == "__main__":
    build()
