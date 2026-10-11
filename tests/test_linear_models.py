"""선형회귀 계수 복원과 ADALINE 수학·분류·CV·결과 계약 검증."""
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from src.MachineLearning.LinearModels.linear_regression import LinearRegressionProfile, run_linear_regression
from src.MachineLearning.NeuralNetworks.adaline import AdalineClassifier, AdalineProfile, run_adaline


class LinearModelsTest(unittest.TestCase):
    def test_simple_and_multiple_regression(self):
        rng = np.random.RandomState(12)
        for n_features in (1, 3):
            with self.subTest(n_features=n_features):
                X = rng.normal(size=(100, n_features))
                coef = np.arange(1, n_features + 1, dtype=float)
                y = X.dot(coef) + 4
                result = run_linear_regression(pd.DataFrame(X), y)
                np.testing.assert_allclose(result.coefficients, coef, atol=1e-10)
                self.assertAlmostEqual(result.intercept, 4)
                self.assertEqual(result.rank, n_features)
                np.testing.assert_allclose(result.residuals, 0, atol=1e-10)
                self.assertAlmostEqual(result.test_metrics["r2"], 1)
                self.assertEqual(result.feature_names, [str(i) for i in range(n_features)])
                self.assertIsNone(result.classes)
                self.assertIsNone(result.probabilities)
                self.assertNotIn("model", json.loads(result.to_json()))

    def test_regression_cv_scaling_and_positive_constraint(self):
        rng = np.random.RandomState(3)
        X = rng.uniform(1, 10, size=(80, 2))
        y = X.dot([2, 3])
        result = run_linear_regression(X, y, LinearRegressionProfile(
            scaling="standard", cv=2, param_grid={"fit_intercept": [True, False]}))
        scaler = result.model.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X[result.train_indices].mean(axis=0))
        np.testing.assert_allclose(result.coefficients, [2, 3] * scaler.scale_, atol=1e-10)
        self.assertIn("model__fit_intercept", result.best_params)
        np.testing.assert_allclose(result.residuals, result.y_test - result.predictions)
        result = run_linear_regression(X, y, LinearRegressionProfile(positive=True, fit_intercept=False))
        np.testing.assert_allclose(result.coefficients, [2, 3], atol=1e-10)
        self.assertEqual(result.intercept, 0)
        self.assertIsNone(result.rank)
        self.assertIsNone(result.singular_values)

    def test_adaline_gradient_is_mean_squared_linear_output(self):
        X = np.array([[1., 2.], [3., 1.], [2., -1.], [-1., -2.]])
        y = np.array([1, 1, 0, 0])
        encoded = np.array([1., 1., -1., -1.])
        model = AdalineClassifier(max_iter=1, eta0=0.1, tol=None).fit(X, y)
        expected_w = np.array([0.075, 0.15])
        np.testing.assert_allclose(model.coef_[0], expected_w)
        self.assertEqual(model.intercept_[0], 0)
        expected_loss = 0.5 * np.mean((encoded - X.dot(expected_w)) ** 2)
        self.assertAlmostEqual(model.loss_curve_[0], expected_loss)
        self.assertEqual(model.n_iter_, 1)
        self.assertFalse(model.converged_)
        self.assertTrue(is_classifier(model))
        self.assertEqual(clone(model).get_params(), model.get_params())
        model = AdalineClassifier(max_iter=1, eta0=0.1, tol=None).fit(X, [1, 1, 1, 0])
        self.assertAlmostEqual(model.intercept_[0], 0.05)
        model = AdalineClassifier(max_iter=1, eta0=0.1, tol=None, fit_intercept=False).fit(X, [1, 1, 1, 0])
        self.assertEqual(model.intercept_[0], 0)

    def test_adaline_binary_labels_cv_and_dto(self):
        X, y = make_blobs(n_samples=120, centers=[[-2, -2], [2, 2]], cluster_std=0.3, random_state=2)
        labels = np.where(y == 0, "negative", "positive")
        result = run_adaline(X, labels, AdalineProfile(cv=2, param_grid={"eta0": [0.01, 0.05]}))
        self.assertEqual(result.test_metrics["accuracy"], 1)
        self.assertTrue(result.converged)
        self.assertLess(result.loss_curve[-1], result.loss_curve[0])
        self.assertEqual(len(result.loss_curve), result.n_iter)
        self.assertIsNone(result.probabilities)
        self.assertNotIn("roc_auc", result.test_metrics)
        np.testing.assert_array_equal(result.classes, ["negative", "positive"])
        np.testing.assert_array_equal(result.predictions, result.classes[(result.activation_scores >= 0).astype(int)])
        np.testing.assert_array_equal(result.model.predict(X[result.test_indices]), result.predictions)
        np.testing.assert_allclose(result.activation_scores, result.model.decision_function(X[result.test_indices]))
        np.testing.assert_allclose(result.model.named_steps["scaler"].mean_, X[result.train_indices].mean(axis=0))
        self.assertFalse(set(result.train_indices) & set(result.test_indices))
        self.assertIn("model__eta0", result.best_params)
        self.assertNotIn("model", json.loads(result.to_json()))

    def test_adaline_convergence_and_invalid_input(self):
        X = np.array([[-2.], [-1.], [1.], [2.], [-3.], [3.]])
        y = np.array([0, 0, 1, 1, 0, 1])
        for kwargs in ({"eta0": 0}, {"max_iter": True}, {"tol": 0}, {"fit_intercept": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdalineClassifier(**kwargs).fit(X, y)
        with self.assertRaises(NotFittedError):
            AdalineClassifier().predict(X)
        with self.assertRaises(ValueError):
            run_adaline(X, [0, 1, 2, 0, 1, 2])
        with self.assertRaises(ValueError):
            run_adaline(X, y, AdalineProfile(task="regression"))
        with self.assertRaises(ValueError):
            AdalineClassifier().fit(X, [0, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            AdalineClassifier(eta0=1e200, max_iter=2).fit(X, y)
        with self.assertWarns(ConvergenceWarning):
            model = AdalineClassifier(max_iter=1, tol=1e-12).fit(X, y)
        self.assertFalse(model.converged_)
        with self.assertRaises(ValueError):
            model.predict(np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            run_linear_regression(X, y, LinearRegressionProfile(task="classification"))
        with self.assertRaises(ValueError):
            run_linear_regression(X, y, LinearRegressionProfile(positive=1))


if __name__ == "__main__":
    unittest.main()
