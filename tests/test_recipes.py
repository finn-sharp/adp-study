"""레시피 파일, 실제 실행 연결, 설정 오류 및 조기 종료 분리를 검증합니다."""
from dataclasses import fields
import json
import importlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.datasets import load_iris, make_regression

from src.Recipes import load_recipe, load_profile
from src.Recipes.loader import _REGISTRY
from src.MachineLearning.Trees.xgboost import run_xgboost
from src.MachineLearning.Trees.adaboost import AdaBoostProfile, run_adaboost
from src.stats.tests.parametric.Average.Multivariate.OneWay import run_oneway_anova


class RecipesTest(unittest.TestCase):
    def test_all_model_and_task_recipes(self):
        count = 0
        for name, (_, cls, _, tasks) in _REGISTRY.items():
            for task in tasks:
                with self.subTest(model=name, task=task):
                    profile = load_profile(name, task=task)
                    self.assertEqual(type(profile).__name__, cls)
                    if task is not None:
                        self.assertEqual(profile.task, task)
                    keys = set(load_recipe(name, task=task))
                    aliases = {'alpha', 'lambda'} if name == 'xgboost' else set()
                    self.assertFalse(keys - {f.name for f in fields(profile)} - aliases)
                    count += 1
        self.assertEqual(count, 34)
        self.assertIsInstance(load_profile('mlp').hidden_layer_sizes, tuple)

    def test_all_machine_learning_runners_use_default_recipes(self):
        X_class, y_class = load_iris(return_X_y=True)
        X_reg, y_reg = make_regression(n_samples=90, n_features=4, random_state=2)
        y_reg = y_reg / 100
        for name, (module_name, _, _, tasks) in _REGISTRY.items():
            if name not in ('xgboost','catboost','lightgbm','adaboost','mlp','perceptron','adaline','linear-regression'):
                continue
            run = getattr(importlib.import_module(module_name), 'run_' + name.replace('-', '_'))
            for task in tasks:
                with self.subTest(model=name, task=task):
                    X, y = (X_reg, y_reg) if task == 'regression' else (X_class, y_class)
                    if name == 'adaline':
                        X, y = X[y < 2], y[y < 2]
                    result = run(X, y, task=task)
                    self.assertEqual(result.task, task)
                    np.testing.assert_allclose(result.model.predict(X[result.test_indices]), result.predictions)
                    fitted = result.model.named_steps['model']
                    profile = load_profile(name, task=task)
                    for key in ('n_estimators', 'max_iter', 'iterations', 'fit_intercept'):
                        if hasattr(profile, key):
                            self.assertEqual(fitted.get_params()[key], getattr(profile, key))

    def test_original_xgboost_recipe_and_overrides(self):
        recipe = load_recipe('xgboost', task='regression')
        self.assertEqual(recipe['alpha'], 0)
        self.assertEqual(recipe['lambda'], 1)
        self.assertEqual(recipe['early_stopping_rounds'], 10)
        profile = load_profile('xgboost', task='regression')
        self.assertEqual(profile.reg_alpha, recipe['alpha'])
        self.assertEqual(profile.reg_lambda, recipe['lambda'])
        self.assertEqual(profile.min_child_weight, 10)
        self.assertEqual(profile.eval_metric, 'rmse')
        updated = load_profile('xgboost', task='regression', overrides={'reg_alpha': 0.7})
        self.assertEqual(updated.reg_alpha, 0.7)
        self.assertEqual(load_recipe('xgboost', task='regression'), recipe)

    def test_runner_reads_changed_recipe_without_cache(self):
        X, y = load_iris(return_X_y=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.json'
            for trees in (3, 7):
                path.write_text(json.dumps({'n_estimators': trees, 'random_state': 88}))
                result = run_adaboost(X, y, recipe_path=str(path))
                self.assertEqual(result.model.named_steps['model'].n_estimators, trees)
            result = run_adaboost(X, y, recipe_path=str(path), overrides={'n_estimators': 5})
            self.assertEqual(result.model.named_steps['model'].n_estimators, 5)
            with self.assertRaises(ValueError):
                run_adaboost(X, y, AdaBoostProfile(), recipe_path=str(path))
            with self.assertRaises(ValueError):
                run_adaboost(X, y, AdaBoostProfile(), overrides={})

    def test_invalid_recipe_is_not_silently_ignored(self):
        invalid = ['[]', '{"n_estimators": 2, "n_estimators": 3}',
                   '{"n_estimators": NaN}', '{"typo": 2}',
                   '{"task": "regression"}']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            for content in invalid:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        load_profile('adaboost', recipe_path=str(path))
            path.write_text('{"alpha": 1, "reg_alpha": 2}')
            with self.assertRaises(ValueError):
                load_profile('xgboost', recipe_path=str(path))
            with self.assertRaises(FileNotFoundError):
                load_profile('adaboost', recipe_path=str(path.parent / 'missing.json'))
        with self.assertRaises(ValueError):
            load_profile('perceptron', task='regression')
        with self.assertRaises(TypeError):
            load_profile('adaboost', overrides=[])
        with self.assertRaises(ValueError):
            load_profile('adaboost', overrides={'unknown': 1})

    def test_default_path_is_independent_of_working_directory(self):
        original = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                self.assertEqual(load_profile('xgboost', task='regression').early_stopping_rounds, 10)
        finally:
            os.chdir(original)

    def test_xgboost_early_stopping_inside_training_and_cv(self):
        X, y = make_regression(n_samples=100, n_features=4, random_state=7)
        result = run_xgboost(X, y, task='regression', overrides={
            'n_estimators': 20, 'cv': 2, 'param_grid': {'max_depth': [2, 3]}})
        train = set(result.early_stopping_train_indices)
        validation = set(result.early_stopping_validation_indices)
        self.assertFalse(train & validation)
        self.assertFalse((train | validation) & set(result.test_indices))
        self.assertEqual(train | validation, set(result.train_indices))
        self.assertIsInstance(result.best_iteration, int)
        self.assertTrue(0 <= result.best_iteration < 20)
        self.assertEqual(len(result.cv_results['params']), 2)
        fitted = result.model.named_steps['model']
        self.assertEqual(fitted.early_stopping_rounds, 10)
        self.assertEqual(fitted.native_model_.get_params()['min_child_weight'], 10)
        np.testing.assert_allclose(result.model.predict(X[result.test_indices]), result.predictions)
        X, y = load_iris(return_X_y=True)
        classification = run_xgboost(X, y, overrides={
            'early_stopping_rounds': 3, 'n_estimators': 10, 'cv': 2})
        np.testing.assert_allclose(classification.probabilities.sum(axis=1), 1, atol=1e-5)

    def test_statistical_runner_recipe_override(self):
        groups = [np.array([1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]),
                  np.array([4, 5, 6, 7, 8])]
        result = run_oneway_anova(groups, overrides={'equal_var': False, 'alpha': 0.01})
        self.assertIn('Welch', result.test_type)
        self.assertEqual(result.reject_h0, result.p_value < 0.01)


if __name__ == '__main__':
    unittest.main()
