"""수치 정확성, 데이터 누수, 입력 검증 및 모델 실행을 검증합니다.

실행: python -m unittest discover -s tests -v
외부 데이터/토큰/코퍼스 다운로드 없이 sklearn 내장 및 합성 데이터만 사용합니다.
"""
import ast
import importlib
import importlib.util
import pathlib
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_diabetes, load_iris, make_blobs
from sklearn.linear_model import Ridge

from src.MachineLearning.Linear.logistic import LogisticProfile, run_logistic
from src.MachineLearning.Linear.numerical import NumericalProfile, run_numerical_regression
from src.MachineLearning.Decomposition.pca import PCAClassificationProfile, run_pca_classification, run_decomposition
from src.MachineLearning.Decomposition.manifold import EmbeddingProfile, run_embedding
from src.MachineLearning.Clustering.clustering import ClusteringProfile, distance_matrix, make_linkage_matrix, run_clustering
from src.MachineLearning.feature_selection import FeatureSelectionProfile, run_feature_selection, run_permutation_importance
from src.stats.analysis import StatisticalProfile, run_descriptive, run_statistical_test
from src.stats.tests.parametric.Average.Univariate import TestProfile, run_one_sample_test
from src.stats.tests.parametric.Average.Bivariate import TwoSampleTestProfile, run_two_sample_test
from src.stats.tests.parametric.Average.Multivariate.OneWay import OneWayANOVAProfile, run_oneway_anova
from src.stats.tests.parametric.Average.Multivariate.TwoWay import TwoWayANOVAProfile, run_twoway_anova
from src.stats.tests.nonParametric.onesample import run_one_sample_rank_test
from src.stats.tests.nonParametric.twosample import run_two_sample_rank_test
from src.TextMining.text import TextProfile, TopicProfile, run_vectorization, run_text_classification, run_topic_model


class SupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_iris(return_X_y=True)
        cls.R, cls.t = load_diabetes(return_X_y=True)

    def test_all_model_families_and_tasks(self):
        cases = [
            ('Linear.least_squares', 'LeastSquares', 'least_squares', ['regression']),
            ('Linear.logistic', 'Logistic', 'logistic', ['classification']),
            ('Linear.lasso', 'Lasso', 'lasso', ['regression']),
            ('Linear.ridge', 'Ridge', 'ridge', ['regression']),
            ('Trees.decision_tree', 'DecisionTree', 'decision_tree', ['classification', 'regression']),
            ('Trees.random_forest', 'RandomForest', 'random_forest', ['classification', 'regression']),
            ('Trees.gradient_boosting', 'GradientBoosting', 'gradient_boosting', ['classification', 'regression']),
            ('Trees.adaboost', 'AdaBoost', 'adaboost', ['classification', 'regression']),
            ('Neighbors.knn', 'KNN', 'knn', ['classification', 'regression']),
            ('SVM.svm', 'SVM', 'svm', ['classification', 'regression']),
            ('NeuralNetworks.mlp', 'MLP', 'mlp', ['classification', 'regression']),
            ('NaiveBayes.naive_bayes', 'NaiveBayes', 'naive_bayes', ['classification'])]
        for module_name, name, func, tasks in cases:
            module = importlib.import_module('src.MachineLearning.' + module_name)
            for task in tasks:
                with self.subTest(model=name, task=task):
                    kwargs = {'task': task}
                    if name in ('RandomForest', 'GradientBoosting', 'AdaBoost'):
                        kwargs['n_estimators'] = 15
                    if name == 'MLP':
                        kwargs.update(solver='lbfgs', hidden_layer_sizes=(10,), max_iter=4000)
                    profile = getattr(module, name + 'Profile')(**kwargs)
                    X, y = (self.X, self.y) if task == 'classification' else (self.R[:100], self.t[:100])
                    result = getattr(module, 'run_' + func)(X, y, profile)
                    self.assertTrue(all(np.isfinite(list(result.test_metrics.values()))))
                    self.assertEqual(len(result.predictions), len(result.y_test))
                    self.assertEqual(set(result.train_indices) & set(result.test_indices), set())
                    if task == 'regression':
                        self.assertIsNone(result.classes)
                        self.assertIsNone(result.confusion)

    def test_scaler_only_uses_train_and_split_is_reproducible(self):
        profile = LogisticProfile()
        a, b = run_logistic(self.X, self.y, profile), run_logistic(self.X, self.y, profile)
        np.testing.assert_array_equal(a.test_indices, b.test_indices)
        np.testing.assert_allclose(a.model.named_steps['scaler'].mean_, self.X[a.train_indices].mean(axis=0))
        np.testing.assert_array_equal(a.predictions, b.predictions)

    def test_cv_and_pca_do_not_use_holdout(self):
        profile = PCAClassificationProfile(param_grid={'pca__n_components': [2, 3], 'C': [0.1, 1]}, cv=3)
        result = run_pca_classification(self.X, self.y, profile)
        self.assertIn('model__C', result.best_params)
        scaler = result.model.named_steps['scaler']
        np.testing.assert_allclose(scaler.mean_, self.X[result.train_indices].mean(axis=0))
        pca = result.model.named_steps['pca']
        self.assertEqual(pca.n_samples_, len(result.train_indices))
        self.assertTrue(np.isfinite(result.best_cv_score))

    def test_binary_auc_and_probability_class_order(self):
        X, y = self.X[self.y != 2], self.y[self.y != 2]
        result = run_logistic(X, y)
        self.assertIn('roc_auc', result.test_metrics)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1)
        self.assertEqual(result.confusion.shape, (2, 2))

    def test_elasticnet_and_naive_bayes_variants(self):
        from src.MachineLearning.NaiveBayes.naive_bayes import NaiveBayesProfile, run_naive_bayes
        result = run_logistic(self.X, self.y, LogisticProfile(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=3000))
        self.assertTrue(np.isfinite(result.test_metrics['accuracy']))
        for variant in ['gaussian', 'multinomial', 'bernoulli']:
            with self.subTest(variant=variant):
                result = run_naive_bayes(self.X, self.y, NaiveBayesProfile(variant=variant))
                self.assertTrue(np.isfinite(result.test_metrics['accuracy']))

    def test_invalid_features_profile_and_target(self):
        for X, y, p in [(self.X.copy()*np.nan, self.y, LogisticProfile()),
                        (self.X, self.y, LogisticProfile(test_size=1)),
                        (self.X, self.y, LogisticProfile(cv=1)),
                        (self.X, self.y.reshape(-1, 1), LogisticProfile()),
                        (self.X, self.y, LogisticProfile(param_grid={})),
                        (self.X, np.ones(len(self.y)), LogisticProfile())]:
            with self.subTest(profile=p), self.assertRaises(ValueError):
                run_logistic(X, y, p)


class NumericalAndUnsupervisedTests(unittest.TestCase):
    def test_svd_singular_design_matches_lstsq(self):
        X = np.column_stack([np.arange(10), np.arange(10)])
        y = 3 + 2*np.arange(10)
        result = run_numerical_regression(X, y)
        expected = np.linalg.lstsq(np.column_stack([np.ones(10), X]), y, rcond=1e-12)[0]
        np.testing.assert_allclose(np.r_[result.intercept, result.coefficients], expected)
        self.assertEqual(result.rank, 2)

    def test_ridge_closed_form_matches_sklearn(self):
        X = np.random.RandomState(1).normal(size=(30, 3)); y = np.arange(30)
        result = run_numerical_regression(X, y, NumericalProfile(method='ridge', alpha=2))
        expected = Ridge(alpha=2, solver='cholesky').fit(X, y)
        np.testing.assert_allclose(result.coefficients, expected.coef_)
        np.testing.assert_allclose(result.predictions, expected.predict(X))

    def test_normal_and_tsvd(self):
        X = np.random.RandomState(3).normal(size=(20, 4)); y = np.arange(20)
        normal = run_numerical_regression(X, y, NumericalProfile(method='normal'))
        svd = run_numerical_regression(X, y)
        np.testing.assert_allclose(normal.predictions, svd.predictions)
        truncated = run_numerical_regression(X, y, NumericalProfile(method='tsvd', n_components=2))
        self.assertEqual(truncated.predictions.shape, y.shape)

    def test_clustering_families_and_noise(self):
        X, y = make_blobs(n_samples=60, centers=3, random_state=12)
        for method in ['kmeans', 'hierarchical', 'dbscan', 'gaussian_mixture']:
            with self.subTest(method=method):
                result = run_clustering(X, ClusteringProfile(method=method, n_init=2), labels_true=y)
                self.assertEqual(result.labels.shape, y.shape)
                self.assertIn('adjusted_rand', result.metrics)
        noise = run_clustering(X, ClusteringProfile(method='dbscan', eps=0.00001))
        self.assertEqual(noise.n_noise, len(X))
        self.assertNotIn('silhouette', noise.metrics)

    def test_hierarchical_threshold_distance_and_linkage(self):
        X = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
        result = run_clustering(X, ClusteringProfile(method='hierarchical', n_clusters=None, distance_threshold=2, scaling='none'))
        self.assertEqual(result.n_clusters, 2)
        np.testing.assert_allclose(np.diag(distance_matrix(X)), 0)
        self.assertEqual(make_linkage_matrix(X).shape, (3, 4))
        with self.assertRaises(ValueError):
            run_clustering(X, ClusteringProfile(method='hierarchical', metric='cosine'))

    def test_pca_factor_analysis_and_feature_selection(self):
        from src.MachineLearning.Decomposition.pca import DecompositionProfile
        X, y = load_iris(return_X_y=True)
        pca = run_decomposition(X)
        self.assertEqual(pca.transformed.shape, (150, 2))
        self.assertLessEqual(pca.cumulative_variance_ratio[-1], 1)
        factor = run_decomposition(X, DecompositionProfile(method='factor_analysis'))
        self.assertIsNone(factor.explained_variance_ratio)
        selected = run_feature_selection(X, y, FeatureSelectionProfile(n_features=2))
        self.assertEqual(selected.transformed.shape[1], 2)
        result = run_logistic(X, y)
        pi = run_permutation_importance(result.model, X[result.test_indices], result.y_test, n_repeats=2)
        self.assertEqual(pi.importances.shape, (4, 2))
        with self.assertRaises(ValueError):
            run_feature_selection(X-10, y, FeatureSelectionProfile())

    def test_manifold_methods(self):
        X, _ = make_blobs(n_samples=30, centers=2, random_state=5)
        for method in ['mds', 'isomap', 'lle', 'tsne']:
            with self.subTest(method=method):
                result = run_embedding(X, EmbeddingProfile(method=method, perplexity=5))
                self.assertEqual(result.coordinates.shape, (30, 2))
                self.assertTrue(np.isfinite(result.coordinates).all())


class StatisticalTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([1, 2, 4, 5, 6.5, 8])
        self.b = np.array([2, 3, 3, 6, 7, 10])

    def test_one_and_two_sample_tests_match_scipy(self):
        for direction in ['two-sided', 'less', 'greater']:
            one = run_one_sample_test(self.a, TestProfile(mu_0=3, alternative=direction))
            expected = stats.ttest_1samp(self.a, 3, alternative=direction)
            self.assertAlmostEqual(one.p_value, expected.pvalue)
            for equal_var in [False, True]:
                two = run_two_sample_test(self.a, self.b, TwoSampleTestProfile(diff_0=1, equal_var=equal_var, alternative=direction))
                expected = stats.ttest_ind(self.a-1, self.b, equal_var=equal_var, alternative=direction)
                self.assertAlmostEqual(two.statistic, expected.statistic)
                self.assertAlmostEqual(two.p_value, expected.pvalue)

    def test_z_tests(self):
        one = run_one_sample_test(self.a, TestProfile(mu_0=3, known_var=True, sigma=2))
        z = (self.a.mean()-3)/(2/np.sqrt(len(self.a)))
        self.assertAlmostEqual(one.statistic, z)
        self.assertAlmostEqual(one.p_value, 2*stats.norm.sf(abs(z)))
        two = run_two_sample_test(self.a, self.b, TwoSampleTestProfile(known_var=True, sigma1=2, sigma2=3))
        self.assertAlmostEqual(two.statistic, (self.a.mean()-self.b.mean())/np.sqrt(4/len(self.a)+9/len(self.b)))

    def test_classic_and_welch_anova(self):
        from statsmodels.stats.oneway import anova_oneway
        groups = [self.a, self.b, np.array([9, 10, 11, 12])]
        for equal_var in [True, False]:
            result = run_oneway_anova(groups, OneWayANOVAProfile(equal_var=equal_var))
            expected = anova_oneway(groups, use_var='equal' if equal_var else 'unequal')
            self.assertAlmostEqual(result.statistic, expected.statistic)
            self.assertAlmostEqual(result.p_value, expected.pvalue)

    def test_type_three_anova_uses_named_terms(self):
        y = [12.5,13.1,11.8,14,16.2,15.8,17.1,16.5,10.1,11,10.5,9.8,12,12.8,13.2,12.1]
        for interaction in [True, False]:
            for ss_type in [1, 2, 3]:
                result = run_twoway_anova(y, ['A']*8+['B']*8, (['Low']*4+['High']*4)*2,
                                           TwoWayANOVAProfile(interaction=interaction, ss_type=ss_type))
                table = result.anova_table
                self.assertAlmostEqual(result.f_factor1, table.loc['C(Q("Factor1"), Sum)', 'F'])
                self.assertAlmostEqual(result.f_factor2, table.loc['C(Q("Factor2"), Sum)', 'F'])
                if interaction:
                    self.assertAlmostEqual(result.f_interaction, table.loc['C(Q("Factor1"), Sum):C(Q("Factor2"), Sum)', 'F'])

    def test_distribution_and_nonparametric_tests(self):
        for test in ['paired_t', 'wilcoxon', 'mannwhitney', 'pearson', 'spearman', 'kendall', 'variance_f']:
            with self.subTest(test=test):
                result = run_statistical_test(self.a, StatisticalProfile(test=test), other=self.b)
                self.assertTrue(np.isfinite(result.statistic))
                self.assertTrue(0 <= result.p_value <= 1)
        for test in ['levene', 'bartlett', 'kruskal', 'friedman']:
            result = run_statistical_test([self.a, self.b, self.a+4], StatisticalProfile(test=test))
            self.assertTrue(0 <= result.p_value <= 1)
        for test in ['shapiro', 'normaltest', 'variance_chi2']:
            self.assertTrue(np.isfinite(run_statistical_test(np.arange(20), StatisticalProfile(test=test)).p_value))
        one = run_one_sample_rank_test(self.a)
        self.assertAlmostEqual(one.p_value, stats.wilcoxon(self.a).pvalue)
        two = run_two_sample_rank_test(self.a, self.b)
        self.assertAlmostEqual(two.p_value, stats.mannwhitneyu(self.a, self.b, alternative='two-sided').pvalue)

    def test_chi_square_and_quartile_deviation(self):
        table = [[10, 20], [30, 40]]
        result = run_statistical_test(table, StatisticalProfile(test='chi2_independence'))
        self.assertAlmostEqual(result.p_value, stats.chi2_contingency(table)[1])
        goodness = run_statistical_test([10, 20, 30], StatisticalProfile(test='chi2_goodness'), expected=[20, 20, 20])
        self.assertAlmostEqual(goodness.p_value, stats.chisquare([10, 20, 30], [20, 20, 20]).pvalue)
        desc = run_descriptive(self.a)
        self.assertEqual(desc['quartile_deviation'], desc['iqr']/2)

    def test_invalid_statistical_inputs(self):
        with self.assertRaises(ValueError):
            run_oneway_anova([[1, np.nan], [2, 3]], OneWayANOVAProfile())
        with self.assertRaises(ValueError):
            run_oneway_anova([[1, 1], [2, 2]], OneWayANOVAProfile())
        with self.assertRaises(ValueError):
            run_one_sample_test(self.a, TestProfile(mu_0=0, alpha=3))
        with self.assertRaises(ValueError):
            run_statistical_test([1, 1, 1])

    def test_regression_inference_selection_and_repeated_anova(self):
        from src.stats.regression import RegressionProfile, SelectionProfile, run_regression_inference, run_variable_selection
        from src.stats.repeated_measures import RepeatedANOVAProfile, run_repeated_anova
        rng = np.random.RandomState(7); X = rng.normal(size=(80, 3)); y = 3*X[:, 0]+rng.normal(size=80)
        result = run_regression_inference(X, y)
        np.testing.assert_allclose(result.model.params.values, np.linalg.lstsq(np.column_stack([np.ones(80), X]), y, rcond=None)[0])
        self.assertIn('breusch_pagan_p', result.diagnostics)
        for method in ['forward', 'backward', 'stepwise']:
            selection = run_variable_selection(X, y, SelectionProfile(method=method))
            self.assertIn('x0', selection.selected_features)
            self.assertTrue(selection.converged)
        binary = (X[:, 0] + rng.normal(size=80) > 0).astype(int)
        self.assertIn('pseudo_r2', run_regression_inference(X, binary, RegressionProfile(method='logit')).diagnostics)
        frame = pd.DataFrame({'subject': np.repeat(np.arange(10), 3), 'time': np.tile(np.arange(3), 10),
                               'response': rng.normal(size=30)+np.tile(np.arange(3), 10)})
        repeated = run_repeated_anova(frame, RepeatedANOVAProfile('response', 'subject', ['time']))
        self.assertIn('reject_h0', repeated.anova_table)


class TextAssociationAndSeriesTests(unittest.TestCase):
    def test_text_vocabulary_uses_training_only(self):
        result = run_vectorization(['alpha beta', 'beta gamma'], test_documents=['testonlytoken beta'])
        self.assertNotIn('testonlytoken', result.feature_names)
        self.assertEqual(result.test_matrix.shape, (1, 3))

    def test_text_classification_cv_and_topics(self):
        docs = [('good excellent happy' if i%2 else 'bad awful sad')+' unique'+str(i) for i in range(40)]
        y = np.arange(40)%2
        result = run_text_classification(docs, y, TextProfile(param_grid={'model__C': [0.1, 1]}, cv=3))
        vocab = result.model.named_steps['vectorizer'].vocabulary_
        for i in result.test_indices:
            self.assertNotIn('unique'+str(i), vocab)
        topics = run_topic_model(docs, TopicProfile(max_iter=3))
        np.testing.assert_allclose(topics.document_topics.sum(axis=1), 1)
        self.assertEqual(len(topics.top_words), 2)

    def test_association_duplicate_items_and_no_rules(self):
        from src.Association.apriori import AssociationProfile, run_association
        result = run_association([['Milk', 'Milk', 'Eggs'], ['Eggs'], ['Milk']])
        milk = result.frequent_itemsets[result.frequent_itemsets.itemsets == frozenset(['Milk'])]
        self.assertAlmostEqual(float(milk.support.iloc[0]), 2/3)
        empty = run_association([['Milk'], ['Eggs']], AssociationProfile(min_support=1))
        self.assertTrue(empty.rules.empty)

    def test_forecast_time_split_and_diagnostics(self):
        from src.TimeSeries.forecasting import ForecastProfile, SeriesDiagnosticsProfile, run_forecast, run_series_diagnostics
        rng = np.random.RandomState(4)
        series = pd.Series(10+np.cumsum(rng.normal(size=80)), index=pd.date_range('2020-01-01', periods=80, freq='D'))
        result = run_forecast(series, ForecastProfile(holdout=8, order_candidates=[(0, 1, 0), (1, 1, 0)]))
        self.assertEqual(result.model.nobs, 72)
        self.assertEqual(result.train.index[-1], series.index[-9])
        np.testing.assert_array_equal(result.predictions.index, result.test.index)
        self.assertTrue((result.intervals.lower <= result.intervals.upper).all())
        diagnostics = run_series_diagnostics(series, SeriesDiagnosticsProfile(difference_order=2))
        np.testing.assert_allclose(diagnostics['transformed'].values, series.diff().diff().dropna().values)
        seasonal = run_forecast(series, ForecastProfile(method='sarimax', order=(0, 1, 0), holdout=8))
        self.assertTrue(np.isfinite(seasonal.metrics['rmse']))

    def test_varmax_and_irregular_series(self):
        from src.TimeSeries.forecasting import MultivariateForecastProfile, validate_series, run_multivariate_forecast
        rng = np.random.RandomState(8); X = rng.normal(size=(100, 2))
        for i in range(1, 100): X[i] += 0.4*X[i-1]
        frame = pd.DataFrame(X, columns=['a', 'b'])
        result = run_multivariate_forecast(frame, MultivariateForecastProfile(holdout=5))
        self.assertEqual(result['predictions'].shape, (5, 2))
        with self.assertRaises(ValueError):
            validate_series(pd.Series(range(10), index=[0, 2, 1, 3, 4, 5, 6, 7, 8, 9]))

    def test_all_plot_kinds_preserve_rcparams(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.Visualization.plots import PlotProfile, run_plot, run_pairplot
        before = plt.rcParams['font.size']
        for kind in ['line', 'scatter', 'bar', 'hist', 'box', 'heatmap']:
            X, y = (np.eye(3), None) if kind == 'heatmap' else (np.arange(6), np.arange(6)**2)
            plot = run_plot(X, y, PlotProfile(kind=kind, font_size=20))
            self.assertEqual(plt.rcParams['font.size'], before)
            self.assertTrue(len(plot.axes.get_children()) > 0)
            plt.close(plot.figure)
        grid = run_pairplot(pd.DataFrame({'x': range(10), 'y': np.arange(10)**2}))
        plt.close(grid.fig)


class OptionalBoostingTests(unittest.TestCase):
    def test_xgboost_and_lightgbm_when_installed(self):
        X, y = load_iris(return_X_y=True)
        R, t = load_diabetes(return_X_y=True)
        ran = 0
        for dependency, module_name, name, func in [('xgboost', 'xgboost', 'XGBoost', 'run_xgboost'),
                                                   ('lightgbm', 'lightgbm', 'LightGBM', 'run_lightgbm')]:
            if importlib.util.find_spec(dependency) is None:
                continue
            module = importlib.import_module('src.MachineLearning.Trees.' + module_name)
            for task in ['classification', 'regression']:
                with self.subTest(dependency=dependency, task=task):
                    result = getattr(module, func)(X if task == 'classification' else R,
                           y if task == 'classification' else t,
                           getattr(module, name+'Profile')(task=task, n_estimators=5))
                    self.assertTrue(all(np.isfinite(list(result.test_metrics.values()))))
                    ran += 1
        if not ran:
            self.skipTest('선택적 xgboost/lightgbm 패키지가 설치되지 않음')


class DocumentationAndCompatibilityTests(unittest.TestCase):
    def test_python37_syntax_in_new_modules(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        folders = ['src/MachineLearning', 'src/Association', 'src/TimeSeries', 'src/TextMining', 'src/Visualization']
        for folder in folders:
            for path in (root/folder).rglob('*.py'):
                if path.name == '__init__.py': continue
                with self.subTest(path=str(path)):
                    tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 7)) if hasattr(ast, 'unparse') else ast.parse(path.read_text(encoding='utf-8'))
                    self.assertIsNotNone(ast.get_docstring(tree))
                    for node in tree.body:
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith('_'):
                            self.assertIsNotNone(ast.get_docstring(node), node.name)


if __name__ == '__main__':
    unittest.main()
