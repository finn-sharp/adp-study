"""실제 학습, CV, 누수 방지, 결과 계약 및 명세 형식을 검증합니다."""
import ast
from dataclasses import fields
import importlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_regression


CASES = [
    ("Trees.xgboost", "XGBoost", "xgboost", {"n_estimators": 8}, "n_estimators", [5, 8]),
    ("Trees.catboost", "CatBoost", "catboost", {"iterations": 8, "depth": 3}, "iterations", [5, 8]),
    ("Trees.lightgbm", "LightGBM", "lightgbm", {"n_estimators": 8, "min_child_samples": 5}, "n_estimators", [5, 8]),
    ("Trees.adaboost", "AdaBoost", "adaboost", {"n_estimators": 8}, "n_estimators", [5, 8]),
    ("NeuralNetworks.mlp", "MLP", "mlp", {"solver": "lbfgs", "max_iter": 2000}, "alpha", [0.001, 0.01]),
    ("NeuralNetworks.perceptron", "Perceptron", "perceptron", {}, "eta0", [0.1, 1.0]),
]


class StandardModelsTest(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)

    def load_case(self, case):
        module = importlib.import_module("src.MachineLearning." + case[0])
        return module, getattr(module, case[1] + "Profile"), getattr(module, "run_" + case[2])

    def check_output(self, result, X, y, task, name):
        self.assertEqual(type(result).__name__, name + "OutputDTO")
        self.assertEqual(result.task, task)
        self.assertEqual(result.predictions.shape, result.y_test.shape)
        np.testing.assert_array_equal(result.y_test, y[result.test_indices])
        np.testing.assert_allclose(result.model.predict(X[result.test_indices]), result.predictions)
        self.assertFalse(set(result.train_indices) & set(result.test_indices))
        self.assertEqual(len(result.train_indices) + len(result.test_indices), len(X))
        payload = json.loads(result.to_json())
        self.assertNotIn("model", payload)
        self.assertIn("predictions", payload)
        self.assertIs(result.to_dict(include_runtime=True)["model"], result.model)
        self.assertTrue(all(np.isfinite(v) for v in result.test_metrics.values()))
        if task == "regression":
            self.assertIsNone(result.classes)
            self.assertIsNone(result.probabilities)
        elif name == "Perceptron":
            self.assertIsNone(result.probabilities)
        else:
            self.assertEqual(result.probabilities.shape, (len(result.y_test), len(result.classes)))
            np.testing.assert_allclose(result.probabilities.sum(axis=1), 1, atol=1e-5)

    def test_classification_and_regression(self):
        for case in CASES:
            _, Profile, run = self.load_case(case)
            for task in (["classification"] if case[1] == "Perceptron" else ["classification", "regression"]):
                with self.subTest(model=case[1], task=task):
                    X, y = (self.X, self.y) if task == "classification" else make_regression(
                        n_samples=90, n_features=4, noise=1, random_state=1234)
                    if task == "regression":
                        y = y / 100
                    result = run(X, y, Profile(task=task, **case[3]))
                    self.check_output(result, X, y, task, case[1])

    def test_grid_search_and_training_only_scaler(self):
        for case in CASES:
            _, Profile, run = self.load_case(case)
            with self.subTest(model=case[1]):
                X = self.X.copy()
                X[:10] *= 100
                result = run(X, self.y, Profile(scaling="standard", cv=2,
                             param_grid={case[4]: case[5]}, **case[3]))
                self.assertIn("model__" + case[4], result.best_params)
                self.assertEqual(len(result.cv_results["params"]), 2)
                np.testing.assert_allclose(result.model.named_steps["scaler"].mean_,
                                           X[result.train_indices].mean(axis=0))
                self.assertFalse(np.allclose(result.model.named_steps["scaler"].mean_, X.mean(axis=0)))
                self.check_output(result, X, self.y, "classification", case[1])

    def test_catboost_string_labels_and_regression_cv(self):
        _, Profile, run = self.load_case(CASES[1])
        labels = np.array(["class_" + str(i) for i in self.y])
        result = run(pd.DataFrame(self.X, columns=list("abcd")), labels,
                     Profile(iterations=5, depth=3, cv=2, param_grid={"depth": [2, 3]}))
        self.assertEqual(result.feature_names, list("abcd"))
        self.assertEqual(result.model.predict(self.X[:2]).shape, (2,))
        self.assertEqual(result.classes.tolist(), ["class_0", "class_1", "class_2"])
        X, y = make_regression(n_samples=60, n_features=4, random_state=1)
        result = run(X, y, Profile(task="regression", iterations=5, cv=2, param_grid={"depth": [2, 3]}))
        self.check_output(result, X, y, "regression", "CatBoost")

    def test_invalid_inputs(self):
        for case in CASES:
            _, Profile, run = self.load_case(case)
            with self.subTest(model=case[1]):
                for config in ({"n_jobs": 0}, {"cv": True}, {"scaling": "bad"}, {"stratify": 1}):
                    with self.assertRaises(ValueError):
                        run(self.X, self.y, Profile(**config))
                with self.assertRaises(TypeError):
                    run(self.X, self.y, object())
                with self.assertRaises(ValueError):
                    run(np.full_like(self.X, np.nan), self.y, Profile())
        _, Profile, run = self.load_case(CASES[0])
        with self.assertRaises(ValueError):
            run(self.X, self.y + 1, Profile())
        _, Profile, run = self.load_case(CASES[-1])
        with self.assertRaises(ValueError):
            run(self.X, self.y, Profile(task="regression"))

    def test_complete_uniform_specification(self):
        section_names = ["1. Imports", "2. Input Profile", "3. Output DTO", "4. Input Validation",
                         "5. Individual Model Runners", "6. Public Runner", "7. Example Execution"]
        for case in CASES:
            module, Profile, _ = self.load_case(case)
            with self.subTest(model=case[1]):
                source = Path(module.__file__).read_text(encoding="utf-8")
                positions = [source.index(name) for name in section_names]
                self.assertEqual(positions, sorted(positions))
                tree = ast.parse(source)
                header = ast.get_docstring(tree)
                for name in ("설명:", "분석 가정:", "작성 정보:", "변경 이력 (Revision History):"):
                    self.assertIn(name, header)
                for name in ("__author__", "__email__", "__version__", "__status__"):
                    self.assertTrue(getattr(module, name))
                for cls in (Profile, getattr(module, case[1] + "OutputDTO")):
                    for field in fields(cls):
                        self.assertIn(field.name + " (", cls.__doc__)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        for name in ("Args:", "Returns:", "Raises:"):
                            self.assertIn(name, ast.get_docstring(node))


if __name__ == "__main__":
    unittest.main()
