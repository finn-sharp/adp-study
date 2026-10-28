"""
================================================================================
test_visualization_gallery.py - PNG 전수 대응·실제 모듈 CLI·import 무저장 계약을 검증합니다.
================================================================================
설명:
    - PNG 전수 대응·실제 모듈 CLI·import 무저장 계약을 검증합니다.
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
import json
import os
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
import tempfile
import unittest
from examples.visualization_gallery import ROOT, check_gallery, registered_plots


class VisualizationGalleryTest(unittest.TestCase):
    def test_all_registered_modules_and_packaged_pngs(self):
        summary = check_gallery()
        self.assertEqual(summary, {"module_count": 61, "image_count": 80})
        manifest = json.loads(
            (ROOT / "docs/images/visualization/manifest.json").read_text()
        )
        self.assertEqual(
            set(manifest["dashboards"]), {p.split(".")[0] for p in registered_plots()}
        )
        variants = manifest["modules"]["regression.glm_residuals"]["variants"]
        self.assertEqual(
            set(variants),
            {
                "logistic_pearson",
                "logistic_deviance",
                "poisson_deviance",
                "gamma_pearson",
                "gamma_deviance",
            },
        )
        self.assertEqual(
            manifest["modules"]["regression.observed_predicted"]["parameters"][
                "sample_role"
            ],
            "train",
        )

    def test_each_module_main_passes_its_own_plot_function(self):
        for path in registered_plots():
            with self.subTest(path=path):
                source = ROOT / ("src/Visualization/" + path.replace(".", "/") + ".py")
                tree = ast.parse(source.read_text())
                guards = [
                    n
                    for n in tree.body
                    if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Compare)
                    and isinstance(n.test.left, ast.Name)
                    and n.test.left.id == "__name__"
                ]
                self.assertEqual(len(guards), 1)
                calls = [
                    n
                    for n in ast.walk(guards[0])
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "save_example"
                ]
                self.assertEqual(len(calls), 1)
                self.assertEqual(ast.literal_eval(calls[0].args[0]), path)
                self.assertEqual(calls[0].args[1].id, "plot_" + path.split(".")[-1])

    def test_simple_examples_do_not_fit_unrelated_models(self):
        from examples.visualization_gallery_inputs import example_cases

        example_cases.cache_clear()
        with patch(
            "src.stats.regression.ols.run_ols",
            side_effect=AssertionError("unrelated fit"),
        ), patch(
            "src.Optimization.genetic_algorithm.run_genetic_algorithm",
            side_effect=AssertionError("unrelated optimization"),
        ):
            self.assertIn("numeric.histogram", example_cases("numeric"))
            self.assertIn("qq.gamma", example_cases("qq"))
        example_cases.cache_clear()

    def test_import_all_modules_never_calls_image_export(self):
        script = """from unittest.mock import patch
import importlib
from examples.visualization_gallery import registered_plots
with patch('examples.visualization_gallery.save_example',side_effect=AssertionError('import must not save')):
    for path in registered_plots():
        importlib.import_module('src.Visualization.'+path)
"""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )

    def test_direct_module_execution_exports_images_outside_repo(self):
        with tempfile.TemporaryDirectory() as folder:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            env["MPLCONFIGDIR"] = str(Path(folder) / "mpl-cache")
            env["MPLBACKEND"] = "Agg"
            out = Path(folder) / "png"
            for name in ("numeric.histogram", "qq.gamma", "regression.glm_residuals"):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.Visualization." + name,
                        "--output-dir",
                        str(out),
                    ],
                    cwd=folder,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((out / (name.replace(".", "/") + ".png")).is_file())
            self.assertEqual(len(list(out.rglob("*.png"))), 8)


if __name__ == "__main__":
    unittest.main()
