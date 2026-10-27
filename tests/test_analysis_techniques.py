"""
================================================================================
test_analysis_techniques.py - 새 분석 기법의 수학, 결과 계약, 누수 방지, 레시피와 명세 검증.
================================================================================
설명:
    - 새 분석 기법의 수학, 결과 계약, 누수 방지, 레시피와 명세 검증.
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
from dataclasses import fields
import importlib
from itertools import combinations
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import adjusted_rand_score

from src.Recipes import load_profile
from src.MachineLearning.Clustering.kmeans import run_kmeans
from src.MachineLearning.Clustering.dbscan import run_dbscan
from src.MachineLearning.Clustering.hdbscan import run_hdbscan
from src.MachineLearning.Clustering.k_distance import run_k_distance
from src.MachineLearning.Decomposition.pca import run_pca
from src.MachineLearning.Decomposition.lda import run_lda
from src.MachineLearning.Decomposition.umap import run_umap
from src.MachineLearning.Decomposition.tsne import run_tsne
from src.MachineLearning.Decomposition.mds import run_mds
from src.MachineLearning.Decomposition.som import run_som, SOMTransformer
from src.MachineLearning.NaiveBayes.naive_bayes import run_naive_bayes
from src.MachineLearning.FeatureSelection.wrapper import run_wrapper_selection, WrapperSelector
from src.MachineLearning.FeatureSelection.regularized import run_regularized_selection
from src.Association.apriori import run_apriori
from src.Networks.sna import run_sna

NEW_KEYS = ['kmeans','dbscan','hdbscan','k-distance','pca','lda','umap','tsne','mds','som',
            'naive-bayes','wrapper-selection','regularized-selection','apriori','sna']


class AnalysisTechniquesTest(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.blobs, self.labels = make_blobs(n_samples=90, centers=3, cluster_std=0.12, random_state=1234)

    def test_kmeans_and_density_labels_and_noise(self):
        for run in (run_kmeans, run_dbscan, run_hdbscan):
            result = run(self.blobs)
            self.assertEqual(result.n_clusters, 3)
            self.assertGreater(adjusted_rand_score(self.labels, result.labels), 0.95)
            self.assertTrue(0 <= result.silhouette <= 1)
            self.assertNotIn('model', json.loads(result.to_json()))
        result = run_dbscan(self.X, overrides={'eps': 1e-12, 'min_samples': 10})
        self.assertEqual(result.n_clusters, 0)
        self.assertTrue(result.noise_mask.all())
        self.assertIsNone(result.silhouette)
        result = run_dbscan(self.blobs, overrides={'eps': 20})
        self.assertEqual(result.n_clusters, 1)
        self.assertIsNone(result.silhouette)

    def test_k_distance_includes_self_and_preserves_rows(self):
        X = np.array([[0.], [1.], [3.], [6.]])
        result = run_k_distance(X, overrides={'scaling': 'none', 'k': 2})
        np.testing.assert_allclose(result.distances, [1, 1, 2, 3])
        np.testing.assert_allclose(result.sorted_distances, result.distances[result.sorted_indices])
        duplicate = run_k_distance([[0.],[0.],[2.]], overrides={'scaling': 'none', 'k': 2})
        np.testing.assert_allclose(duplicate.distances, [0,0,2])

    def test_pca_reference_and_new_transform(self):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        result = run_pca(self.X)
        scaled = StandardScaler().fit_transform(self.X)
        reference = PCA(n_components=2, svd_solver='full').fit(scaled)
        np.testing.assert_allclose(result.explained_variance_ratio, reference.explained_variance_ratio_)
        np.testing.assert_allclose(result.embedding, reference.transform(scaled))
        error = np.mean((scaled-reference.inverse_transform(result.embedding))**2)
        self.assertAlmostEqual(result.reconstruction_error, error)
        self.assertEqual(result.model.transform(self.X[:3]).shape, (3,2))

    def test_lda_supervision_and_rank_bound(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        result = run_lda(self.X, self.y)
        reference = LinearDiscriminantAnalysis(n_components=2).fit(StandardScaler().fit_transform(self.X), self.y)
        np.testing.assert_allclose(result.explained_variance_ratio, reference.explained_variance_ratio_)
        with self.assertRaises(ValueError):
            run_lda(self.X[self.y < 2], self.y[self.y < 2])
        self.assertEqual(run_lda(self.X[self.y < 2], self.y[self.y < 2], overrides={'n_components':1}).embedding.shape[1], 1)
        with self.assertRaises(ValueError):
            run_lda(self.X, np.zeros(len(self.y)))

    def test_umap_tsne_mds_transform_contract_and_metrics(self):
        X = self.X[:45]
        umap = run_umap(X, overrides={'n_neighbors':5, 'n_epochs':40})
        self.assertTrue(umap.supports_transform)
        self.assertEqual(umap.model.transform(self.X[-3:]).shape, (3,2))
        tsne = run_tsne(X, overrides={'perplexity':10, 'max_iter':500})
        self.assertFalse(tsne.supports_transform)
        self.assertFalse(hasattr(tsne.model, 'transform'))
        self.assertTrue(np.isfinite(tsne.kl_divergence))
        mds = run_mds(X, overrides={'n_init':1, 'max_iter':50})
        self.assertFalse(mds.supports_transform)
        scaled = mds.model.named_steps['scaler'].transform(X)
        stress = np.sum((pdist(scaled)-pdist(mds.embedding))**2)
        self.assertAlmostEqual(mds.distance_reconstruction_error, stress, places=5)
        self.assertAlmostEqual(mds.stress, mds.model.named_steps["model"].stress_)
        for result in (umap,tsne,mds):
            self.assertEqual(result.embedding.shape, (45,2))
            self.assertTrue(np.isfinite(result.embedding).all())
            json.loads(result.to_json())

    def test_som_codebook_and_quantization_math(self):
        result = run_som(self.X, overrides={'rows':3,'columns':2,'n_iter':200})
        scaled = result.model.named_steps['scaler'].transform(self.X)
        distances = cdist(scaled, result.weights.reshape(6,4))
        self.assertAlmostEqual(result.quantization_error, distances.min(axis=1).mean())
        coords = np.array([(r,c) for r in range(3) for c in range(2)])
        np.testing.assert_array_equal(result.embedding, coords[distances.argmin(axis=1)])
        self.assertTrue(0 <= result.topographic_error <= 1)
        self.assertEqual(result.model.transform(self.X[:1]).shape,(1,2))
        som = SOMTransformer(rows=1,columns=1,n_iter=10)
        np.testing.assert_allclose(clone(som).fit(self.X).weights_, som.fit(self.X).weights_)
        self.assertEqual(som.topographic_error(self.X),0)

    def test_naive_bayes_variants_and_nonnegative_contract(self):
        from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
        for variant, Cls in [('gaussian',GaussianNB),('multinomial',MultinomialNB),('bernoulli',BernoulliNB),('complement',ComplementNB)]:
            with self.subTest(variant=variant):
                result = run_naive_bayes(self.X, self.y, overrides={'variant':variant})
                reference = Cls().fit(self.X[result.train_indices],self.y[result.train_indices])
                np.testing.assert_allclose(result.probabilities, reference.predict_proba(self.X[result.test_indices]))
                np.testing.assert_allclose(result.class_counts, reference.class_count_)
        with self.assertRaises(ValueError):
            run_naive_bayes(self.X-10,self.y,overrides={'variant':'multinomial'})
        with self.assertRaises(ValueError):
            run_naive_bayes(self.X,self.y,overrides={'variant':'bernoulli','binarize':None})

    def test_wrapper_directions_and_cv_training_only(self):
        rng = np.random.RandomState(9)
        X = rng.normal(size=(100,5))
        y = 7 * X[:,0]
        for method in ('forward','backward','stepwise'):
            with self.subTest(method=method):
                result = run_wrapper_selection(pd.DataFrame(X,columns=list('abcde')),y,task='regression',
                    overrides={'method':method,'n_features_to_select':2})
                self.assertIn('a',result.selected_features)
                self.assertEqual(len(result.selected_features),1 if method=='stepwise' else 2)
                np.testing.assert_allclose(result.transformed_test,X[result.test_indices][:,result.selected_indices])
                final_scaler=result.model.named_steps['selected_scaler']
                np.testing.assert_allclose(final_scaler.mean_,result.transformed_train.mean(axis=0))
                # 선택기에 보류 자료가 전달되지 않았음을 동일 학습 행 재적합으로 확인.
                reference=clone(result.selector).fit(X[result.train_indices],y[result.train_indices])
                np.testing.assert_array_equal(reference.get_support(),result.support_mask)
                self.assertEqual(reference.history_,result.history)
                np.testing.assert_allclose(result.model.predict(X[result.test_indices]),result.predictions)
        result=run_wrapper_selection(X,y,task='regression',overrides={'n_features_to_select':None})
        self.assertEqual(result.selected_indices.tolist(),[0])
        self.assertEqual(result.stopped_reason,'no_improvement')
        self.assertEqual(clone(WrapperSelector()).get_params(),WrapperSelector().get_params())

    def test_l1_l2_coefficients_threshold_and_holdout(self):
        rng=np.random.RandomState(2)
        X=rng.normal(size=(100,5));y=5*X[:,0]
        for penalty in ('l1','l2'):
            result=run_regularized_selection(X,y,task='regression',overrides={'penalty':penalty})
            expected=(result.importances>=result.applied_threshold)&(result.importances>0)
            np.testing.assert_array_equal(result.support_mask,expected)
            self.assertIn(0,result.selected_indices)
            scaler=result.selector.selection_model_.named_steps['scaler']
            np.testing.assert_allclose(scaler.mean_,X[result.train_indices].mean(axis=0))
            self.assertAlmostEqual(result.test_metrics['r2'],1)
            if penalty=='l1':self.assertEqual(result.selected_indices.tolist(),[0])
            else:self.assertAlmostEqual(result.applied_threshold,result.importances.mean())
            classification=run_regularized_selection(self.X,self.y,overrides={'penalty':penalty})
            self.assertEqual(classification.coefficients.shape[1],4)
        with self.assertRaises(ValueError):
            run_regularized_selection(X,y,task='regression',overrides={'threshold':1e8})

    def test_apriori_exact_support_and_rule_arithmetic(self):
        transactions=[['A','A','B'],['A','B','C'],['A','C'],[],['B','C']]
        result=run_apriori(transactions,overrides={'min_support':0.2,'min_confidence':0,'min_lift':0})
        supports={tuple(row.items):row.support for row in result.itemsets.itertuples(index=False)}
        for size in (1,2,3):
            for subset in combinations('ABC',size):
                expected=sum(set(subset).issubset(set(t)) for t in transactions)/5
                self.assertAlmostEqual(supports[subset],expected)
        for row in result.rules.itertuples(index=False):
            self.assertAlmostEqual(row.confidence,row.support/supports[row.antecedents])
            self.assertAlmostEqual(row.lift,row.confidence/supports[row.consequents])
        self.assertEqual(result.n_transactions,5)
        self.assertEqual(len(result.rules),12)
        with self.assertRaises(ValueError):
            run_apriori(transactions,overrides={'max_candidates':2})
        with self.assertRaises(ValueError):
            run_apriori(transactions,overrides={'max_rules':1,'min_confidence':0,'min_lift':0})
        empty=run_apriori([[],[]])
        self.assertTrue(empty.itemsets.empty)
        self.assertTrue(empty.rules.empty)
        json.loads(empty.to_json())

    def test_sna_known_graph_weight_and_isolates(self):
        result=run_sna([('A','B'),('B','C')],overrides={'nodes':['A','B','C','isolated']})
        self.assertEqual(result.n_components,2)
        self.assertGreater(result.node_metrics.loc['B','betweenness'],result.node_metrics.loc['A','betweenness'])
        self.assertAlmostEqual(result.node_metrics.pagerank.sum(),1)
        self.assertAlmostEqual(result.density,2/6)
        weighted=run_sna([('A','B',2),('B','A',3)])
        self.assertEqual(weighted.model['A']['B']['weight'],5)
        self.assertEqual(weighted.model['A']['B']['distance'],0.2)
        directed=run_sna([('A','B'),('B','A')],overrides={'directed':True})
        self.assertEqual(directed.density,1)
        isolated=run_sna([],overrides={'nodes':['A','B']})
        self.assertEqual(isolated.modularity,0)
        self.assertEqual(isolated.n_components,2)
        json.loads(result.to_json())
        with self.assertRaises(ValueError):run_sna([('A','A')])

    def test_uniform_full_specification_and_json_profiles(self):
        from src.Recipes.loader import _REGISTRY
        for key in NEW_KEYS:
            with self.subTest(key=key):
                module_name,cls,default,tasks=_REGISTRY[key]
                module=importlib.import_module(module_name)
                source=Path(module.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
                sections=['1. Imports','2. Input Profile','3. Output DTO','4. Input Validation','5. Individual Model Runners','6. Public Runner','7. Example Execution']
                positions=[source.index(s) for s in sections]
                self.assertEqual(positions,sorted(positions))
                for label in ('설명:','분석 가정:','작성 정보:','변경 이력 (Revision History):'):
                    self.assertIn(label,ast.get_docstring(tree))
                for task in tasks:
                    profile=load_profile(key,task=task)
                    for f in fields(profile):self.assertIn(f.name+' (',profile.__doc__)
                DTO=getattr(module,cls.replace('Profile','OutputDTO'))
                for f in fields(DTO):self.assertIn(f.name+' (',DTO.__doc__)
                for node in ast.walk(tree):
                    if isinstance(node,ast.FunctionDef):
                        for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))

    def test_existing_z_and_t_tests_reference(self):
        from scipy import stats
        from src.stats.tests.parametric.Average import Univariate as one, Bivariate as two
        x=np.array([1.,2.,4.,5.,7.]);z=np.array([2.,3.,4.,4.,6.,8.])
        for alternative in ('two-sided','less','greater'):
            p=one.TestProfile(mu_0=3,alternative=alternative)
            result=one._run_t_test(x,p,x.mean(),x.std(ddof=1))
            self.assertAlmostEqual(result.p_value,stats.ttest_1samp(x,3,alternative=alternative).pvalue)
            p=one.TestProfile(mu_0=3,sigma=2,known_var=True,alternative=alternative)
            result=one._run_z_test(x,p,x.mean(),x.std(ddof=1))
            statistic=(x.mean()-3)/(2/np.sqrt(len(x)))
            expected=stats.norm.sf(abs(statistic))*2 if alternative=='two-sided' else stats.norm.sf(statistic) if alternative=='greater' else stats.norm.cdf(statistic)
            self.assertAlmostEqual(result.p_value,expected)
            for equal_var in (True,False):
                p=two.TwoSampleTestProfile(diff_0=1,equal_var=equal_var,alternative=alternative)
                result=two._run_two_sample_t_test(x,z,p,x.mean(),z.mean(),x.std(ddof=1),z.std(ddof=1))
                self.assertAlmostEqual(result.p_value,stats.ttest_ind(x-1,z,equal_var=equal_var,alternative=alternative).pvalue)
            p=two.TwoSampleTestProfile(sigma1=2,sigma2=3,known_var=True,alternative=alternative)
            result=two._run_two_sample_z_test(x,z,p,x.mean(),z.mean(),x.std(ddof=1),z.std(ddof=1))
            statistic=(x.mean()-z.mean())/np.sqrt(4/len(x)+9/len(z))
            expected=stats.norm.sf(abs(statistic))*2 if alternative=='two-sided' else stats.norm.sf(statistic) if alternative=='greater' else stats.norm.cdf(statistic)
            self.assertAlmostEqual(result.p_value,expected)

    def test_invalid_inputs_and_recipe_ranges(self):
        for run in (run_kmeans,run_dbscan,run_hdbscan,run_k_distance,run_pca,run_umap,run_tsne,run_mds,run_som):
            with self.subTest(run=run.__name__):
                with self.assertRaises(ValueError):run([[1,np.nan],[2,3]])
                with self.assertRaises(TypeError):run(self.X,profile=object())
        for run,config in [(run_kmeans,{'n_clusters':True}), (run_dbscan,{'eps':0}),
                           (run_hdbscan,{'min_samples':150}), (run_k_distance,{'k':151}),
                           (run_pca,{'n_components':5}), (run_umap,{'n_neighbors':150}),
                           (run_tsne,{'perplexity':150}), (run_mds,{'eps':0}),
                           (run_som,{'sigma':0})]:
            with self.subTest(config=config),self.assertRaises(ValueError):run(self.X,overrides=config)
        with self.assertRaises(ValueError):run_wrapper_selection(self.X,self.y,overrides={'method':'unknown'})
        with self.assertRaises(ValueError):run_regularized_selection(self.X,self.y,overrides={'penalty':'l3'})
        with self.assertRaises(ValueError):run_apriori([['A']],overrides={'min_support':2})


if __name__=='__main__':
    unittest.main()
