"""
================================================================================
test_module_specifications.py - 전체 Python 파일 헤더와 생성 시 명세 보존을 검증합니다.
================================================================================
설명:
    - 전체 Python 파일 헤더와 생성 시 명세 보존을 검증합니다.
분석 가정:
    - 테스트의 고정 입력·합성 자료와 명시한 허용오차를 사용합니다. 시험 데이터의 성능 보장이 아닙니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

import ast
from pathlib import Path
import tempfile
import unittest
from tools.exams.module_spec import format_python, files, inspect_files, write_python


def executable_tree(source):
    tree = ast.parse(source)
    if ast.get_docstring(tree, clean=False) is not None:
        tree.body = tree.body[1:]
    return ast.dump(tree, include_attributes=False)


class ModuleSpecificationTest(unittest.TestCase):
    def test_every_python_file_has_requested_header(self):
        report = inspect_files(files())
        self.assertEqual(report["failures"], [])
        self.assertGreaterEqual(report["counts"]["python"], 680)

    def test_header_preserves_executable_ast_and_future_shebang(self):
        source = '#!/usr/bin/env python\n# coding: utf-8\nfrom __future__ import annotations\nx = "자료"\n'
        formatted = format_python(source, Path("sample.py"))
        self.assertTrue(
            formatted.startswith("#!/usr/bin/env python\n# coding: utf-8\n")
        )
        self.assertEqual(executable_tree(source), executable_tree(formatted))
        self.assertEqual(formatted, format_python(formatted, Path("sample.py")))
        compile(formatted, "sample.py", "exec")

    def test_existing_notes_and_history_remain_in_header(self):
        source = '"""추론 helper.\n\n반복 자료의 조건과 실행 경로를 확인합니다.\n"""\nvalue=1\n'
        formatted = format_python(source, Path("helper.py"))
        self.assertIn("반복 자료의 조건과 실행 경로를 확인합니다.", formatted)
        self.assertEqual(executable_tree(source), executable_tree(formatted))
        self.assertEqual(formatted, format_python(formatted, Path("helper.py")))

    def test_generated_wrapper_keeps_standard_header_on_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "분석코드.py"
            write_python(
                path, '"""원 자료를 분석합니다."""\nvalue=1\n', encoding="utf-8"
            )
            before = ast.get_docstring(ast.parse(path.read_text()))
            write_python(
                path, '"""원 자료를 분석합니다."""\nvalue=2\n', encoding="utf-8"
            )
            self.assertEqual(ast.get_docstring(ast.parse(path.read_text())), before)
            self.assertEqual(
                executable_tree(path.read_text()), executable_tree("value=2")
            )


if __name__ == "__main__":
    unittest.main()
