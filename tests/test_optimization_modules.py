"""점수 방향·fold 독립 적합·그룹/시간 누출·Optuna 재현성과 오류 계약 검증."""
from dataclasses import replace
import json
import unittest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from src.MachineLearning.Optimization.grid_search import GridSearchProfile, run_grid_search
from src.MachineLearning.Optimization.random_search import run_random_search
from src.MachineLearning.Optimization.optuna_search import run_optuna_search
from src.Recipes import load_profile


class RecordFit(BaseEstimator, TransformerMixin):
    rows = []

    def fit(self, X, y=None):
        self.rows.append(np.asarray(X)[:, 0].astype(int).tolist())
        return self

    def transform(self, X):
        return X


class OptimizationTest(unittest.TestCase):
    def test_pipeline_is_cloned_and_fold_fit_excludes_validation(self):
        X, y = load_iris(return_X_y=True)
        X = np.column_stack([np.arange(len(X)), X])
        RecordFit.rows = []
        estimator = Pipeline([('prep', RecordFit()), ('model', DecisionTreeClassifier(random_state=2))])
        result = run_grid_search(X, y, estimator, overrides={
            'search_space': {'model__max_depth': [1, 2]}, 'refit': False})
        self.assertFalse(hasattr(estimator.named_steps['model'], 'tree_'))
        self.assertIsNone(result.model)
        expected = [set(f['train_indices']) for f in result.folds]
        self.assertEqual(len(RecordFit.rows), 6)
        for rows in RecordFit.rows:
            self.assertIn(set(rows), expected)
        self.assertTrue(all(len(r) < len(X) for r in RecordFit.rows))

    def test_negative_loss_direction_and_refit_prediction(self):
        X, y = make_regression(n_samples=60, n_features=3, random_state=6)
        result = run_grid_search(X, y, DecisionTreeRegressor(random_state=2), task='regression',
                                overrides={'search_space': {'max_depth': [1, 3]}, 'scoring': 'neg_mean_absolute_error'})
        self.assertLessEqual(result.best_score, 0)
        self.assertAlmostEqual(result.best_score, result.cv_results.mean_score.max())
        self.assertEqual(result.model.max_depth, result.best_params['max_depth'])
        self.assertEqual(result.model.predict(X).shape, y.shape)
        data = json.loads(result.to_json())
        self.assertNotIn('model', data); self.assertNotIn('study', data)

    def test_group_split_has_no_shared_groups(self):
        X, y = make_regression(n_samples=60, n_features=3, random_state=2)
        groups = np.repeat(np.arange(12), 5)
        result = run_grid_search(X, y, DecisionTreeRegressor(), groups=groups, task='regression',
            overrides={'split': 'group', 'shuffle': False, 'search_space': {'max_depth': [2]}})
        for f in result.folds:
            self.assertFalse(set(groups[f['train_indices']]) & set(groups[f['validation_indices']]))

    def test_time_split_never_fits_future_rows(self):
        X = np.arange(60).reshape(-1, 1); y = np.sin(X[:, 0])
        result = run_grid_search(X, y, DecisionTreeRegressor(), task='regression',
            overrides={'split': 'time', 'shuffle': False, 'search_space': {'max_depth': [2]}})
        for f in result.folds:
            self.assertLess(max(f['train_indices']), min(f['validation_indices']))

    def test_random_candidates_reproducible(self):
        X, y = load_iris(return_X_y=True)
        options = {'search_space': {'max_depth': [1, 2, 3, 4, 5]}, 'n_iter': 3}
        a = run_random_search(X, y, DecisionTreeClassifier(random_state=4), overrides=options)
        b = run_random_search(X, y, DecisionTreeClassifier(random_state=4), overrides=options)
        self.assertEqual(a.cv_results.params.tolist(), b.cv_results.params.tolist())
        np.testing.assert_array_equal(a.cv_results.mean_score, b.cv_results.mean_score)

    def test_optuna_json_space_and_scorer_direction_reproducible(self):
        try:
            import optuna
        except ImportError:
            self.skipTest('optional Optuna unavailable')
        X, y = make_regression(n_samples=50, n_features=2, random_state=7)
        kwargs = {'search_space': {'max_depth': {'type': 'int', 'low': 1, 'high': 4}}, 'n_iter': 4}
        a = run_optuna_search(X, y, DecisionTreeRegressor(random_state=3), task='regression', overrides=kwargs)
        b = run_optuna_search(X, y, DecisionTreeRegressor(random_state=3), task='regression', overrides=kwargs)
        self.assertEqual(a.best_params, b.best_params)
        self.assertLessEqual(a.best_score, 0)
        self.assertAlmostEqual(a.best_score, a.cv_results.mean_score.max())
        self.assertEqual(set(a.cv_results.state), {'COMPLETE'})
        json.loads(a.to_json())

    def test_bad_configuration_fails_before_fit(self):
        X, y = load_iris(return_X_y=True)
        est = DecisionTreeClassifier()
        for change in [{'cv': True}, {'n_jobs': 0}, {'split': 'time'},
                       {'scoring': 'typo'}, {'search_space': {'unknown': [1]}},
                       {'search_space': {}}, {'timeout': -1}, {'refit': 'yes'}]:
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                run_grid_search(X, y, est, overrides=change)
        for spec in [{'type': 'float', 'low': -1, 'high': 1, 'log': True},
                     {'type': 'int', 'low': 1.5, 'high': 4},
                     {'type': 'float', 'low': 1, 'high': 2, 'step': .3},
                     {'type': 'categorical', 'choices': []}]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                run_optuna_search(X, y, est, overrides={'search_space': {'max_depth': spec}})

    def test_recipe_conflict(self):
        X, y = load_iris(return_X_y=True)
        profile = load_profile('optimization-grid-search')
        with self.assertRaises(ValueError):
            run_grid_search(X, y, DecisionTreeClassifier(), profile, overrides={'cv': 2})
