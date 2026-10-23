"""등록부·레시피·30개 학습 묶음의 단일 집계 및 문항-모듈 정적 대응표."""

from pathlib import Path
import ast
import json
import re
import sys
import nbformat

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "기출문제"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def registry_counts():
    """선택 라이브러리를 import하지 않고 등록부의 실제 항목을 집계합니다."""
    module = ast.parse((REPO / "src/Recipes/loader.py").read_text())
    registry = next(
        ast.literal_eval(n.value)
        for n in module.body
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "_REGISTRY" for x in n.targets)
    )
    return dict(
        modules=len(registry),
        recipe_files=len(list((REPO / "src/Recipes").glob("*.json"))),
        default_recipe_combinations=sum(len(v[3]) for v in registry.values()),
    )


def source_graph(paths):
    """단순 함수 호출의 정적 연결을 추적합니다. 동적 dispatch는 완료로 추정하지 않습니다."""
    functions = {}
    imports = {}
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("src.")
            ):
                for alias in node.names:
                    imports[alias.asname or alias.name] = node.module
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node

    def resolve(code):
        pending = [
            n.func.id
            for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        found = set()
        visited = set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            if name in imports:
                found.add(imports[name])
            if name in functions:
                pending.extend(
                    n.func.id
                    for n in ast.walk(functions[name])
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                )
        return sorted(found)

    return resolve


def build():
    bundles = []
    scenario = read(ROOT / "상황형_문항명세.json")
    entries = [(folder, "scenario_practice", spec) for folder, spec in scenario.items()]
    for number in range(30, 37):
        folder = "유형연습/" + str(number) + "회"
        entries.append(
            (folder, "recent_type_practice", read(ROOT / folder / "문항명세.json"))
        )
    folder = "모의고사/생존분석_통계"
    entries.append(
        (folder, "survival_original_mock", read(ROOT / folder / "문항명세.json"))
    )
    folder = "모의고사/5회"
    entries.append(
        (folder, "production_original_mock", read(ROOT / folder / "문항명세.json"))
    )
    for number in range(1, 11):
        folder = "모의고사/%d회" % (number + 5)
        entries.append(
            (folder, "applied_original_mock", read(ROOT / folder / "문항명세.json"))
        )
    for folder, kind, spec in entries:
        dest = ROOT / folder
        notebooks = list(dest.glob("*풀이.ipynb")) + list(dest.glob("*해설.ipynb"))
        assert len(notebooks) == 1, (folder, notebooks)
        path = notebooks[0]
        book = nbformat.read(path, 4)
        code_cells = [c.source for c in book.cells if c.cell_type == "code"]
        setup = ast.parse(code_cells[0])
        imports = {}
        # Common helpers are traversed only when that question calls the helper.
        common = [ROOT / "공통/reconstruction.py", ROOT / "유형연습/공통/analysis.py"]
        resolver = source_graph(common)
        records = []
        codes = iter(code_cells[1:])
        for section in spec["sections"]:
            for q in section["questions"]:
                if isinstance(q, dict):
                    qid, text = q["id"], q["text"]
                else:
                    qid, text = q[0], q[1]
                code = next(codes)
                if kind == "applied_original_mock":
                    applied = source_graph(
                        sorted((ROOT / "모의고사/공통").glob("*.py"))
                    )
                    modules = applied(
                        q["task"]
                        + "(ctx)"
                        + (
                            "\n_reg_candidates(ctx)"
                            if q["task"] == "reg_tuning"
                            else ""
                        )
                    )
                elif kind == "recent_type_practice":
                    task = q["task"]
                    code = task + "(ctx)"
                if kind in ("survival_original_mock", "production_original_mock"):
                    # setup definitions and imports are part of the same notebook namespace.
                    temp = ast.parse(code_cells[0] + "\n" + code)
                    direct = {
                        a.asname or a.name: n.module
                        for n in ast.walk(temp)
                        if isinstance(n, ast.ImportFrom)
                        and n.module
                        and n.module.startswith("src.")
                        for a in n.names
                    }
                    calls = {
                        n.func.id
                        for n in ast.walk(ast.parse(code))
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    }
                    modules = sorted({direct[n] for n in calls if n in direct})
                elif kind != "applied_original_mock":
                    modules = resolver(code)
                    for n in ast.walk(setup):
                        if (
                            isinstance(n, ast.ImportFrom)
                            and n.module
                            and n.module.startswith("src.")
                        ):
                            referenced = {
                                v.id
                                for v in ast.walk(ast.parse(code))
                                if isinstance(v, ast.Name)
                            }
                            if any((a.asname or a.name) in referenced for a in n.names):
                                modules.append(n.module)
                    modules = sorted(set(modules))
                # Empty means orchestration/reporting or a remaining direct library calculation.
                records.append(
                    dict(
                        id=qid,
                        text=text,
                        module_paths=modules,
                        mapping=(
                            "static_reference"
                            if modules
                            else "exam_specific_or_direct_library"
                        ),
                        note="정적 참조 목록이며 모듈 재사용 완결성이나 실행 커버리지를 의미하지 않습니다.",
                    )
                )
        bundle = dict(
            folder=folder,
            kind=kind,
            questions=records,
            question_count=len(records),
            notebook=str(path.relative_to(REPO)),
            question_paper=str(next(dest.glob("*문제지.pdf")).relative_to(REPO)),
            execution=str((dest / "실행검증.json").relative_to(REPO)),
            code_cells=len(code_cells),
            figures=sum(
                "image/png" in o.get("data", {})
                for c in book.cells
                for o in c.get("outputs", [])
            ),
            verified_python=read(dest / "실행검증.json")["python"],
        )
        if kind == "recent_type_practice":
            bundle.update(sources=spec["sources"], scope=spec["scope"])
        else:
            bundle["scope"] = (
                "reconstructed_or_original_learning_material; see source audits"
            )
        bundles.append(bundle)
    # Keep comprehensive mock exams in numerical order, followed by the supplement.
    bundles.sort(
        key=lambda b: (
            (1, int(b["folder"].split("/")[-1][:-1]))
            if b["folder"].startswith("모의고사/")
            and b["folder"].split("/")[-1][:-1].isdigit()
            else ((2, 0) if b["folder"].startswith("모의고사/") else (0, 0))
        )
    )
    summary = dict(
        bundles=len(bundles),
        questions=sum(b["question_count"] for b in bundles),
        code_cells=sum(b["code_cells"] for b in bundles),
        figures=sum(b["figures"] for b in bundles),
    )
    assert summary["bundles"] == 41 and summary["questions"] == 393
    payload = dict(
        schema_version=1,
        analysis=registry_counts(),
        summary=summary,
        mapping_method="conservative static references; dynamic calls and remaining direct library code are not asserted as migrated",
        bundles=bundles,
    )
    (ROOT / "전체목록.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 전체 학습 자료 목록",
        "",
        "생성: `python tools/exams/build_catalog.py`. 등록부·JSON·회차 명세를 직접 집계합니다.",
        "",
        "**{modules}개 분석 모듈 · {recipe_files}개 JSON · {default_recipe_combinations}개 기본 로드 조합**".format(
            **payload["analysis"]
        ),
        "",
        "**{bundles}묶음 · {questions}문항 · {code_cells}코드 셀 · {figures}그림**".format(
            **summary
        ),
        "",
        "공식 기출 전체 복원이 아닙니다. 원자료·후기 유형·창작 보완의 구분은 각 묶음 출처 문서를 따릅니다. 32회는 통계, 35회는 공간 전처리의 부분 연습입니다.",
        "",
        "| 묶음 | 문항 | 코드 셀 | 실행 환경 |",
        "| --- | ---: | ---: | --- |",
    ]
    for b in bundles:
        lines.append(
            "| [{folder}]({folder}/README.md) | {question_count} | {code_cells} | Python {verified_python} |".format(
                **b
            )
        )
    lines += [
        "",
        "문항별 모듈 참조는 [전체목록.json](전체목록.json)에 있습니다. 정적 연결이 확인되지 않은 문항은 `exam_specific_or_direct_library`로 표시합니다. 이 표시는 누락을 감추거나 모든 풀이의 공통화를 완료한 것으로 집계하지 않기 위한 것입니다.",
    ]
    (ROOT / "전체목록.md").write_text("\n".join(lines) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(build()["summary"], ensure_ascii=False))
