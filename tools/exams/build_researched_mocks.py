"""종합 모의16~20회: 명세·고정 자료·문제지·풀이·해설을 함께 생성합니다."""

from pathlib import Path
import json
import nbformat
from researched_mock_specs import specification, SOURCES
from researched_mock_data import generate
from improve_mock_readability import improve_bundle

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "기출문제/모의고사"
SETUP = """from pathlib import Path
import sys
FOLDER=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0,str(FOLDER.parent/'공통'))
from researched_core import ResearchContext
ctx=ResearchContext(FOLDER)
"""


def build():
    for number in range(16, 21):
        spec = specification(number)
        folder = ROOT / ("%d회" % number)
        generate(folder, spec)
        (folder / "문항명세.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
        )
        (folder / "실행설정.json").write_text(
            json.dumps(
                dict(
                    seed=spec["seed"],
                    sections=[
                        dict(
                            family=s["family"],
                            settings=s["settings"],
                            context=s["context"],
                        )
                        for s in spec["sections"]
                    ],
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        cells = [
            nbformat.v4.new_markdown_cell("# 모의고사 %d회" % number),
            nbformat.v4.new_code_cell(SETUP),
        ]
        for sec in spec["sections"]:
            cells.append(
                nbformat.v4.new_markdown_cell(
                    "## " + sec["title"] + "\n\n" + sec["context"]
                )
            )
            for q in sec["questions"]:
                cells += [
                    nbformat.v4.new_markdown_cell(
                        "### "
                        + q["id"]
                        + "\n\n**문제** "
                        + q["text"]
                        + "\n\n**선택 근거·가정·해석**\n\n"
                        + q["answer"]
                    ),
                    nbformat.v4.new_code_cell("ctx.run(%r,%r)" % (q["id"], q["task"])),
                ]
        book = nbformat.v4.new_notebook(
            cells=cells,
            metadata=dict(
                adp=dict(
                    type="researched_mock",
                    subquestions=[
                        q["id"] for s in spec["sections"] for q in s["questions"]
                    ],
                ),
                kernelspec=dict(
                    name="python3", display_name="Python 3", language="python"
                ),
            ),
        )
        # Stable cell ids reduce unrelated changes after regeneration.
        for i, cell in enumerate(book.cells):
            cell.id = "mock%d-cell%02d" % (number, i)
        nbformat.write(book, folder / ("모의고사_%d회_풀이.ipynb" % number))
        (folder / ("모의고사_%d회_문제지.pdf" % number)).touch()
        links = ["| 출처 | 확인된 유형 |", "| --- | --- |"]
        for key in spec["source_ids"]:
            source = SOURCES[key]
            links.append(
                "| [%s·실제 기출%d회](%s) | %s |"
                % (key, source["exam"], source["url"], source["verified"])
            )
        (folder / "README.md").write_text(
            "# 모의고사 %d회 — %s\n\n12문항·100점·권장240분. [자료 안내](자료안내.md) → [문제지](모의고사_%d회_문제지.pdf) → [풀이](모의고사_%d회_풀이.ipynb) → [수치 해설](해설.md) 순서로 학습합니다.\n\n문항·배점·합성 자료·평가 조건은 창작이며 실제 기출 원문·정답·기업 데이터가 아닙니다. 회차 번호는 이 저장소의 모의 번호입니다. [10~20회 조사 및 대응표](../10~20회_출처조사.md)에 확인 범위·확장 조건을 기록했습니다.\n\n%s\n\n각 대문항의 source_ids는 유형의 근거입니다. 출처 없는 연관규칙, PCA·Welch, 구간 설계·튜닝·시간 간격 등 구체 조건은 응용 확장입니다. 영어 품사는 제공된 통제 어휘 사전으로만 처리합니다.\n\n저장소 루트에서 다음과 같이 실행합니다.\n\n```bash\npython -m pip install -r 기출문제/모의고사/공통/requirements-py312.txt\npython 기출문제/모의고사/%d회/분석코드.py\n```\n\n공통 코드는 `../공통/researched_*.py`에 분야별로 나누었으며 src의 기존 모듈을 조합합니다. 전체 정밀 수치·원행·확률·검증 구간은 실행결과/답안수치.json에 보존합니다.\n"
            % (number, spec["title"], number, number, "\n".join(links), number)
        )
        improve_bundle(folder)
        print("built", number, flush=True)


if __name__ == "__main__":
    build()
