"""공식 참조값·손계산·확률 순서·평가 누수·경계 및 skeleton 계약 검증."""
import ast
import importlib
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn import metrics as sk
from src.Recipes import load_profile
from src.Recipes.loader import _REGISTRY
from src.Evaluation._common import example_inputs, UndefinedMetric
from src.Evaluation.classification.metrics import run_classification_metrics
from src.Evaluation.regression.metrics import run_regression_metrics
from src.Evaluation.clustering.metrics import run_clustering_metrics
from src.Evaluation.time_series.metrics import run_time_series_metrics
from src.Evaluation.comparison import run_comparison
from src.Evaluation.cv_summary import run_cv_summary
from src.Evaluation.skeleton._common import prepare


def run(group,name,**kwargs):
    module=importlib.import_module('src.Evaluation.'+group+'.'+name)
    return getattr(module,'run_'+name)(**kwargs)


class EvaluationMetricsTest(unittest.TestCase):
    def test_classification_reference(self):
        data=example_inputs('classification');r=run_classification_metrics(**data)
        y,p,s=data['y_true'],data['y_pred'],data['y_score']
        for name,reference in [('accuracy',sk.accuracy_score(y,p)),('precision',sk.precision_score(y,p,average='weighted')),('recall',sk.recall_score(y,p,average='weighted')),('f1',sk.f1_score(y,p,average='weighted')),('mcc',sk.matthews_corrcoef(y,p)),('cohen_kappa',sk.cohen_kappa_score(y,p)),('roc_auc',sk.roc_auc_score(y,s)),('average_precision',sk.average_precision_score(y,s)),('log_loss',sk.log_loss(y,s)),('brier_score',sk.brier_score_loss(y,s))]:
            self.assertAlmostEqual(r.value[name],reference)
        precision,recall,_=sk.precision_recall_curve(y,s)
        self.assertAlmostEqual(r.value['pr_auc'],sk.auc(recall,precision))
        self.assertAlmostEqual(r.value['gini'],2*r.value['roc_auc']-1)

    def test_lift_gain_hand_and_ties(self):
        args=dict(y_true=[1,0,1,0,0],y_score=[.9,.9,.7,.2,.1],overrides={'top_fractions':[.2,.4,1.]})
        lift=run('classification','lift',**args);gain=run('classification','gain',**args)
        np.testing.assert_allclose(lift.value,[2.5,1.25,1.])
        np.testing.assert_allclose(gain.value,[.5,.5,1.])
        np.testing.assert_array_equal(lift.details['top']['count'],[1,2,5])
        curve=run('classification','gain_curve',y_true=args['y_true'],y_score=args['y_score'])
        self.assertEqual(curve.details['curve'].iloc[0]['gain'],0)
        self.assertEqual(curve.details['curve'].iloc[-1]['gain'],1)
        # ceil realized fraction, never divide by requested fraction.
        small=run('classification','lift',y_true=[1,0,0],y_score=[.9,.5,.1],overrides={'top_fractions':[.1]})
        self.assertAlmostEqual(small.value[0],3.)

    def test_reversed_string_probability_order(self):
        truth=np.array(['yes','no','yes','no']);score=np.array([[.8,.2],[.1,.9],[.7,.3],[.3,.7]])
        opts={'labels':['yes','no'],'positive_label':'yes'}
        result=run_classification_metrics(truth,truth,y_score=score,overrides=opts)
        self.assertAlmostEqual(result.value['roc_auc'],1.)
        self.assertAlmostEqual(result.value['log_loss'],-np.mean(np.log([.8,.9,.7,.7])))
        self.assertAlmostEqual(result.value['brier_score'],np.mean(np.array([.2,.1,.3,.3])**2))

    def test_multiclass_column_mapping(self):
        y=np.array(['b','a','c','b','a','c']);pred=y.copy();proba=np.array([[.7,.2,.1],[.1,.8,.1],[.1,.2,.7],[.8,.1,.1],[.2,.7,.1],[.1,.1,.8]])
        opts={'labels':['b','a','c']}
        r=run_classification_metrics(y,pred,y_score=proba,overrides=opts)
        mapped=np.array([0,1,2,0,1,2])
        self.assertAlmostEqual(r.value['roc_auc'],sk.roc_auc_score(mapped,proba,multi_class='ovr',average='weighted'))
        self.assertAlmostEqual(r.value['brier_score'],np.mean(np.sum((proba-np.eye(3)[mapped])**2,axis=1)))
        self.assertNotIn('lift',r.value)
        with self.assertRaises(ValueError): run_classification_metrics(y,pred,y_score=proba)

    def test_class_averages_specificity(self):
        y=[0,0,1,1];pred=[0,1,1,1]
        self.assertAlmostEqual(run('classification','specificity',y_true=y,y_pred=pred,overrides={'average':'binary'}).value,.5)
        np.testing.assert_allclose(run('classification','specificity',y_true=y,y_pred=pred,overrides={'average':None}).value,[1.,.5])
        self.assertAlmostEqual(run('classification','fbeta',y_true=y,y_pred=pred,overrides={'average':'binary','beta':2.}).value,sk.fbeta_score(y,pred,beta=2))

    def test_one_class_and_invalid_probabilities(self):
        with self.assertRaises(UndefinedMetric): run('classification','roc_auc',y_true=[0,0],y_score=[.1,.2],overrides={'labels':[0,1]})
        r=run_classification_metrics([0,0],[0,0],y_score=[.1,.2],overrides={'labels':[0,1]})
        self.assertTrue(np.isnan(r.value['roc_auc']));self.assertTrue(r.warnings)
        self.assertIsNone(json.loads(r.to_json())['value']['roc_auc'])
        invalid=[[[.1,.1],[.1,.9]],[[1.1,-.1],[.1,.9]],[[.1,.9]],[[np.nan,.9],[.1,.9]]]
        for score in invalid:
            with self.assertRaises(ValueError):run_classification_metrics([0,1],[0,1],y_score=score,overrides={'labels':[0,1]})
        with self.assertRaises(ValueError):run('classification','log_loss',y_true=[0,1],y_score=[-1,2])

    def test_float32_softmax_rounding_and_invalid_totals(self):
        probabilities = np.array([[.25, .75000012], [.75, .25000012]], dtype=np.float32)
        original = probabilities.copy()
        result = run_classification_metrics([1, 0], [1, 0], y_score=probabilities,
                                            overrides={'labels': [0, 1]})
        normalized = probabilities.astype(float)
        normalized /= normalized.sum(axis=1, keepdims=True)
        self.assertAlmostEqual(result.value['log_loss'], -np.log(normalized[[0, 1], [1, 0]]).mean())
        self.assertAlmostEqual(result.value['brier_score'], np.mean((normalized[:, 1] - [1, 0])**2))
        np.testing.assert_array_equal(probabilities, original)
        # Identical float64 values exceed their tighter rounding tolerance.
        with self.assertRaises(ValueError):
            run_classification_metrics([1, 0], [1, 0], y_score=original.astype(float),
                                       overrides={'labels': [0, 1]})
        for invalid in (np.array([[.25, .75001], [.75, .25]], dtype=np.float32),
                        np.array([[-1e-8, 1.0], [.75, .25]], dtype=np.float32),
                        np.array([[.25, .76], [.75, .25]], dtype=np.float16)):
            with self.assertRaises(ValueError):
                run_classification_metrics([1, 0], [1, 0], y_score=invalid,
                                           overrides={'labels': [0, 1]})

    def test_calibration_and_ks(self):
        r=run('classification','calibration',y_true=[0,1,0,1],y_score=[0,1,.2,.8],overrides={'n_bins':2})
        self.assertAlmostEqual(r.value,.1);self.assertEqual(r.details['bins']['count'].sum(),4)
        r=run('classification','ks',y_true=[0,0,1,1],y_score=[.1,.2,.8,.9])
        self.assertAlmostEqual(r.value,1.)

    def test_regression_reference_multioutput(self):
        a=np.array([[1.,10.],[2.,20.],[4.,40.],[8.,80.]])
        b=a+np.array([[1.,2.],[-1.,-2.],[2.,4.],[-2.,-4.]])
        r=run_regression_metrics(a,b,overrides={'multioutput':'raw_values','n_features':1})
        np.testing.assert_allclose(r.value['mae'],sk.mean_absolute_error(a,b,multioutput='raw_values'))
        np.testing.assert_allclose(r.value['mse'],sk.mean_squared_error(a,b,multioutput='raw_values'))
        np.testing.assert_allclose(r.value['rmse'],np.sqrt(sk.mean_squared_error(a,b,multioutput='raw_values')))
        np.testing.assert_allclose(r.value['r2'],sk.r2_score(a,b,multioutput='raw_values'))
        np.testing.assert_allclose(r.value['adjusted_r2'],1-(1-r.value['r2'])*3/2)
        np.testing.assert_allclose(r.value['explained_variance'],sk.explained_variance_score(a,b,multioutput='raw_values'))
        np.testing.assert_allclose(r.value['pinball_loss'],sk.mean_pinball_loss(a,b,multioutput='raw_values'))

    def test_percentage_definitions_zero_and_constant(self):
        args=dict(y_true=[1.,2.],y_pred=[2.,1.])
        self.assertAlmostEqual(run('regression','mape',**args).value,75.)
        self.assertAlmostEqual(run('regression','smape',**args).value,200/3)
        self.assertAlmostEqual(run('regression','wape',**args).value,200/3)
        with self.assertRaises(UndefinedMetric):run('regression','mape',y_true=[0,2],y_pred=[1,1])
        self.assertEqual(run('regression','mape',y_true=[0,2],y_pred=[1,1],overrides={'zero_policy':'omit'}).value,50.)
        self.assertEqual(run('regression','smape',y_true=[0],y_pred=[0]).value,0.)
        with self.assertRaises(UndefinedMetric):run('regression','r2',y_true=[1,1],y_pred=[1,1])
        with self.assertRaises(UndefinedMetric):run('regression','msle',y_true=[1,2],y_pred=[-1,2])
        with self.assertRaises(ValueError):run_regression_metrics([1,2],[[1],[2]])

    def test_cluster_reference_noise_mask(self):
        X=np.array([[0,0],[0,1],[5,5],[5,6],[100,100]])
        y=np.array([0,0,1,1,-1]);truth=np.array(['a','a','b','b','c'])
        r=run_clustering_metrics(truth,y,X=X)
        self.assertEqual(r.n_samples,4)
        self.assertAlmostEqual(r.value['silhouette'],sk.silhouette_score(X[:4],y[:4]))
        self.assertAlmostEqual(r.value['davies_bouldin'],sk.davies_bouldin_score(X[:4],y[:4]))
        self.assertAlmostEqual(r.value['calinski_harabasz'],sk.calinski_harabasz_score(X[:4],y[:4]))
        self.assertEqual(r.value['ari'],1.)
        r=run_clustering_metrics(truth,y,X=X,overrides={'exclude_noise':False});self.assertEqual(r.n_samples,5)
        with self.assertRaises(UndefinedMetric):run('clustering','silhouette',y_pred=[0,0],X=[[0],[1]])
        invalid=run_clustering_metrics(y_pred=[-1,-1],X=[[0],[1]])
        self.assertEqual(invalid.n_samples,0);self.assertTrue(invalid.warnings)

    def test_seasonal_mase_rmsse_horizon(self):
        a=np.array([[10.,12.],[11.,13.]]);b=a+2
        train=np.array([1.,3.,5.,7.]);r=run_time_series_metrics(a,b,y_train=train)
        self.assertEqual(r.value['mase'],1.);self.assertEqual(r.value['rmsse'],1.)
        np.testing.assert_allclose(r.details['metrics']['horizon_errors']['rmse_by_horizon_target'],[[2.],[2.]])
        seasonal=run('time_series','mase',y_true=a,y_pred=b,y_train=train,overrides={'seasonal_period':2})
        self.assertEqual(seasonal.value,.5)
        with self.assertRaises(UndefinedMetric):run('time_series','mase',y_true=a,y_pred=b,y_train=[1,1,1])
        with self.assertRaises(ValueError):run('time_series','mase',y_true=a,y_pred=b,y_train=[1])
        r=run_time_series_metrics(a[:,:,None],b[:,:,None],y_train=train[:,None]);self.assertEqual(r.value['mase'],1.)

    def test_comparison_cv(self):
        r=run_comparison([1.,2.,4.],{'perfect':[1.,2.,4.],'bad':[2.,3.,5.]},overrides={'evaluation_task':'regression'})
        self.assertEqual(r.table.loc['perfect','rmse'],0.);self.assertEqual(r.directions['rmse'],'lower')
        with self.assertRaises(ValueError):run_comparison([0,1],{'bad':[0]})
        cv=run_cv_summary({'rmse':[1.,2.,3.]})
        self.assertEqual(cv.table.iloc[0]['mean'],2.);self.assertEqual(cv.table.iloc[0]['std'],1.)
        with self.assertRaises(ValueError):run_cv_summary({'rmse':[1,np.nan]})
        self.assertEqual(run_cv_summary({'rmse':[1,np.nan]},overrides={'nan_policy':'omit'}).table.iloc[0]['omitted'],1)

    def test_every_module_recipe_example_and_python37(self):
        count=0
        for key,(module_name,cls,_,_) in _REGISTRY.items():
            if not key.startswith('metric-'):continue
            module=importlib.import_module(module_name);p=load_profile(key)
            self.assertEqual(type(p).__name__,cls)
            if key in ('metric-comparison','metric-cv-summary'):continue
            group,name=module_name.split('.')[-2:]
            runner=getattr(module,'run_'+(group+'_metrics' if name=='metrics' else name))
            result=runner(**example_inputs(group));json.loads(result.to_json());count+=1
            with self.assertRaises(ValueError):runner(**example_inputs(group),profile=p,overrides={})
        self.assertEqual(count,54)
        for path in Path('src/Evaluation').rglob('*.py'):
            if sys.version_info >= (3,8):
                ast.parse(path.read_text(),feature_version=(3,7))
            else:
                ast.parse(path.read_text())

    def test_all_metric_skeletons_offline_no_import_side_effect(self):
        from sklearn.datasets import load_iris,make_regression
        iris=load_iris();frame=pd.DataFrame(iris.data,columns=['a','b','c','d']);frame['target']=iris.target
        binary=frame[frame.target<2].reset_index(drop=True)
        X,y=make_regression(n_samples=80,n_features=3,random_state=1);reg=pd.DataFrame(X,columns=['a','b','c']);reg['target']=y+1000
        series=pd.DataFrame({'cnt':np.arange(1.,61.)})
        data={'classification':binary,'regression':reg,'clustering':frame,'time_series':series}
        prepared={group:prepare(group,values) for group,values in data.items()}
        for key,(module_name,_,_,_) in _REGISTRY.items():
            if not key.startswith('metric-') or key in ('metric-comparison','metric-cv-summary'):continue
            group,name=module_name.split('.')[-2:]
            skeleton=importlib.import_module('src.Evaluation.skeleton.'+group+'.'+name)
            with patch('src.Evaluation.skeleton.'+group+'.'+name+'.prepare',return_value=prepared[group]):
                result=skeleton.build_pipeline()(data[group]);json.loads(result.to_json())
        with patch('src.datasets.uci.dataset_bytes',side_effect=AssertionError('import must not download')):
            importlib.reload(importlib.import_module('src.Evaluation.skeleton.classification.lift'))

    def test_existing_skeletons_and_summary_skeletons(self):
        from sklearn.datasets import load_iris
        from src.MachineLearning.skeleton.linear_regression import build_pipeline
        from src.MachineLearning.skeleton._common import fit_evaluate
        from src.MachineLearning.skeleton.kmeans import main as cluster_main
        from src.Evaluation.skeleton.comparison import build_pipeline as compare_pipeline
        from src.Evaluation.skeleton.cv_summary import build_pipeline as cv_pipeline
        X = np.arange(1., 81.).reshape(-1, 1)
        frame = pd.DataFrame({'x': X[:, 0], 'target': 2*X[:, 0]+1})
        result = fit_evaluate(frame, build_pipeline(['x']), 'linear')
        self.assertLess(result.metrics['rmse'], 1e-10)
        self.assertIn('wape', result.metrics)
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=['a','b','c','d']); data['target'] = iris.target
        with patch('src.MachineLearning.skeleton.kmeans.load_data', return_value=data):
            cluster = cluster_main()
        self.assertIn('ari', cluster.metrics)
        self.assertIn('davies_bouldin', cluster.metrics)
        compare = compare_pipeline()(data[data.target < 2])
        self.assertIn('lift@10%', compare.table.columns)
        self.assertEqual(cv_pipeline()(data).table.iloc[0]['count'], 3)
        json.loads(compare.to_json())

    def test_inputs_unchanged_and_bad_settings(self):
        y=np.array([0,0,1,1]);score=np.array([.1,.2,.8,.9]);snapshot=score.copy()
        run('classification','lift',y_true=y,y_score=score)
        np.testing.assert_array_equal(score,snapshot)
        for options in [{'beta':0},{'top_fractions':[0]},{'n_bins':True},{'labels':[0,0]}]:
            with self.assertRaises(ValueError):run_classification_metrics(y,y,y_score=score,overrides=options)
        with self.assertRaises(ValueError):run_regression_metrics([1,2],[1,2],overrides={'n_features':True})


if __name__=='__main__':unittest.main()
