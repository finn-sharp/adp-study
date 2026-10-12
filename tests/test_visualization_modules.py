"""Axes 所有권·표시 수치·결정 지표·정의역·명세·subplot 조합 검증."""
import ast
from dataclasses import fields
import importlib,json,sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,silhouette_samples,confusion_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist,squareform
from examples.eda_visualization_inputs import cases
from src.Recipes import load_profile
from src.Recipes.loader import _REGISTRY


def draw(path,args,ax,**kwargs):
    module=importlib.import_module('src.Visualization.'+path)
    return getattr(module,'plot_'+path.split('.')[-1])(*args,ax=ax,**kwargs)

class VisualizationModulesTest(unittest.TestCase):
    def tearDown(self):plt.close('all')

    def test_every_module_reuses_ax_and_keeps_other_panels_and_json_data(self):
        for path,args in cases().items():
            with self.subTest(path=path):
                fig,axes=plt.subplots(1,2);axes[1].plot([1,2],[2,1]);before=tuple(axes[1].get_children());nfigs=len(plt.get_fignums())
                with patch.object(plt,'show',side_effect=AssertionError('No show')),patch.object(plt,'subplots',side_effect=AssertionError('No figure creation')):
                    result=draw(path,args,axes[0])
                self.assertIs(result.axes,axes[0]);self.assertIs(result.figure,fig)
                self.assertEqual(tuple(axes[1].get_children()),before);self.assertEqual(len(fig.axes),2);self.assertEqual(len(plt.get_fignums()),nfigs)
                self.assertTrue(result.artists);self.assertIsInstance(result.data,pd.DataFrame)
                payload=json.loads(result.to_json());self.assertFalse(set(payload)&{'axes','figure','artists'})
                fig.canvas.draw();plt.close(fig)

    def test_explicit_colorbar_stays_in_existing_subplot(self):
        fig,axes=plt.subplots(1,3);matrix=pd.DataFrame([[1.,.5],[.5,1.]],index=['a','b'],columns=['a','b'])
        result=draw('relationships.correlation',(matrix,),axes[0],cax=axes[1]);self.assertEqual(len(fig.axes),3);self.assertIs(result.axes,axes[0]);self.assertTrue(axes[1].collections)
        other,_=plt.subplots()
        with self.assertRaises(ValueError):draw('relationships.correlation',(matrix,),axes[0],cax=other.axes[0])
        with self.assertRaises(ValueError):draw('relationships.correlation',(matrix,),axes[0],cax=axes[0])

    def test_distribution_category_denominators_and_wilson_intervals(self):
        _,ax=plt.subplots();ecdf=draw('numeric.ecdf',([3,1,1,2],),ax)
        np.testing.assert_array_equal(ecdf.data.value,[1,2,3]);np.testing.assert_allclose(ecdf.data.cumulative_fraction,[.5,.75,1])
        count=draw('categorical.count',(['a']*5+['b']*3+['c']*2,),ax,overrides={'max_categories':1,'proportion':True})
        self.assertEqual(count.diagnostics['omitted_rows'],5);self.assertAlmostEqual(count.data.fraction.sum(),1.)
        pareto=draw('categorical.pareto',(['a']*5+['b']*3+['c']*2,),ax,overrides={'max_categories':1})
        self.assertEqual(pareto.diagnostics['categories_to_coverage'],2)
        rate=draw('categorical.category_rate',(['a']*10+['b']*10,[0]*10+[1]*10),ax)
        np.testing.assert_allclose(rate.data.rate,[0,1]);self.assertTrue((rate.data.lower>=0).all());self.assertTrue((rate.data.upper<=1+1e-12).all())
        from statsmodels.stats.proportion import proportion_confint
        low,high=proportion_confint(rate.data.positive_count.to_numpy(),rate.data.n.to_numpy(),alpha=.05,method='wilson')
        np.testing.assert_allclose(rate.data.lower,low,atol=1e-14);np.testing.assert_allclose(rate.data.upper,high,atol=1e-14)
        for normalization in ('count','row','column','all'):
            result=draw('categorical.contingency',(['a','a','b','b'],[0,1,1,1]),ax,overrides={'normalization':normalization})
            if normalization=='row':np.testing.assert_allclose(result.data.sum(axis=1),1)
            if normalization=='column':np.testing.assert_allclose(result.data.sum(axis=0),1)
            if normalization=='all':self.assertAlmostEqual(result.data.values.sum(),1)

    def test_missingness_correlations_and_heatmap_masks_never_zero_undefined(self):
        _,ax=plt.subplots();data=pd.DataFrame({'a':[None,1,None,2],'b':[None,1,None,2],'complete':[1,2,3,4]})
        result=draw('quality.missing_association',(data,),ax);self.assertAlmostEqual(result.data.loc['a','b'],1);self.assertTrue(np.isnan(result.data.loc['a','complete']))
        matrix=pd.DataFrame([[1,.8],[.8,1]],index=list('ab'),columns=list('ab'));p=pd.DataFrame([[np.nan,.2],[.2,np.nan]],index=list('ab'),columns=list('ab'))
        result=draw('relationships.correlation',(matrix,p),ax,overrides={'max_p_value':.05});self.assertTrue(result.data.isna().all().all())
        quality=draw('quality.missingness',(data,),ax);self.assertEqual(quality.diagnostics['complete_rows'],2)

    def test_hierarchy_distance_modes_and_silhouette_reference(self):
        _,ax=plt.subplots();X=np.array([[0.,0.],[0.,1.],[5.,5.],[5.,6.]])
        result=draw('clustering.dendrogram',(X,),ax,overrides={'truncate_last':0,'cut_height':2.})
        np.testing.assert_allclose(result.data.values,linkage(X,method='ward'))
        self.assertEqual(len(np.unique(result.diagnostics['cluster_labels'])),2)
        distance=squareform(pdist(X));result=draw('clustering.dendrogram',(distance,),ax,overrides={'input_kind':'distance','method':'average'})
        np.testing.assert_allclose(result.data.values,linkage(pdist(X),method='average'))
        labels=np.array([0,0,1,1]);result=draw('clustering.silhouette',(X,labels),ax)
        np.testing.assert_allclose(result.data.silhouette,silhouette_samples(X,labels))
        np.testing.assert_allclose(draw('clustering.silhouette',(distance,labels),ax,overrides={'metric':'precomputed'}).data.silhouette,result.data.silhouette)
        strings=draw('clustering.silhouette',(X,['left','left','right','right']),ax)
        np.testing.assert_allclose(strings.data.silhouette,result.data.silhouette)
        noise=draw('clustering.silhouette',(X,[-1,0,1,1]),ax)
        self.assertEqual(noise.diagnostics['noise_excluded'],1);self.assertEqual(noise.data.original_position.tolist(),[1,2,3])
        with self.assertRaises(ValueError):draw('clustering.silhouette',(pd.DataFrame(X),pd.Series(labels,index=[1,2,3,4])),ax)
        kd=draw('clustering.k_distance',(X,),ax,overrides={'k':2});np.testing.assert_allclose(kd.data.distance,1.)
        with self.assertRaises(ValueError):draw('clustering.dendrogram',(distance,),ax,overrides={'input_kind':'distance'})
        with self.assertRaises(ValueError):draw('clustering.silhouette',(X,[0]*4),ax)

    def test_classification_reference_and_all_thresholds_costs(self):
        _,ax=plt.subplots();y=np.array([0,0,1,1,0,1]);s=np.array([.1,.5,.5,.8,.2,.9])
        roc=draw('classification.roc',(y,s),ax);self.assertAlmostEqual(roc.diagnostics['roc_auc'],roc_auc_score(y,s))
        pr=draw('classification.precision_recall',(y,s),ax);self.assertAlmostEqual(pr.diagnostics['average_precision'],average_precision_score(y,s))
        threshold=draw('classification.threshold',(y,s),ax,overrides={'false_positive_cost':2.,'false_negative_cost':3.,'max_thresholds':2})
        for row in threshold.data.itertuples():
            prediction=(s>=row.threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
            self.assertEqual((row.tn,row.fp,row.fn,row.tp),(tn,fp,fn,tp));self.assertAlmostEqual(row.cost_per_sample,(2*fp+3*fn)/len(y))
        self.assertEqual(threshold.diagnostics['minimum_cost_per_sample'],threshold.data.cost_per_sample.min())
        calibration=draw('classification.calibration',(y,s),ax,overrides={'n_bins':4});self.assertAlmostEqual(calibration.diagnostics['brier_score'],np.mean((s-y)**2));self.assertEqual(calibration.data.n.sum(),len(y))
        with self.assertRaises(ValueError):draw('classification.calibration',(y,s+1),ax)
        with self.assertRaises(ValueError):draw('classification.confusion',(y,y,[0]),ax)

    def test_input_domains_paired_rows_and_profile_precedence(self):
        _,ax=plt.subplots()
        with self.assertRaises(ValueError):draw('numeric.kde',([1,1,1],),ax)
        with self.assertRaises(ValueError):draw('numeric.histogram',([1,np.inf],),ax)
        with self.assertRaises(ValueError):draw('relationships.scatter',(pd.Series([1,2],index=[0,1]),pd.Series([1,2],index=[1,2])),ax)
        with self.assertRaises(ValueError):draw('categorical.count',([1,'1'],),ax)
        with self.assertRaises(ValueError):draw('time_series.lag',([1,np.nan,2],),ax)
        with self.assertRaises(ValueError):draw('time_series.timeline',(['2020-01-02','2020-01-01'],[1,2]),ax)
        with self.assertRaises(ValueError):draw('decomposition.explained_variance',([.7,.5],),ax)
        with self.assertRaises(TypeError):draw('numeric.histogram',([1,2],),None)
        profile=load_profile('plot-numeric-histogram')
        with self.assertRaises(ValueError):draw('numeric.histogram',([1,2],),ax,profile=profile,overrides={})
        scatter=draw('relationships.scatter',([1,np.nan,3],[1,2,4]),ax);self.assertEqual(scatter.diagnostics['missing_removed'],1)
        original=pd.DataFrame({'a':[1.,2.,np.nan],'category':['a','b','a']});copy=original.copy(deep=True)
        draw('quality.missingness',(original,),ax);draw('categorical.grouped_distribution',(original.a,original.category),ax)
        pd.testing.assert_frame_equal(original,copy)

    def test_all_profiles_dtos_and_seven_sections_import_offline(self):
        count=0
        for key,(path,profile_name,_,_) in _REGISTRY.items():
            if not key.startswith('plot-'):continue
            count+=1;module=importlib.import_module(path);source=Path(module.__file__).read_text();profile=load_profile(key)
            self.assertIsInstance(profile,getattr(module,profile_name))
            for cls in (getattr(module,profile_name),getattr(module,profile_name.replace('Profile','OutputDTO'))):
                for field in fields(cls):self.assertIn(field.name+' (',cls.__doc__)
            for index,section in enumerate(('Imports','Input Profile','Output DTO','Input Validation','Individual Model Runners','Public Runner','Example Execution'),1):self.assertIn(str(index)+'. '+section,source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))
            for forbidden in ('plt.show','plt.subplots','plt.figure','plt.gca','plt.sca','savefig(','sns.'):
                self.assertNotIn(forbidden,source)
            if sys.version_info >= (3,8):ast.parse(source,feature_version=(3,7))
        self.assertEqual(count,35)

    def test_alternative_views_and_explicit_uncertainty(self):
        _,ax=plt.subplots();X=np.array([1.,2.,4.,5.,6.,9.]);categories=['a']*3+['b']*3
        for kind in ('violin','strip'):
            result=draw('categorical.grouped_distribution',(X,categories),ax,overrides={'kind':kind})
            np.testing.assert_array_equal(result.data.n,[3,3])
        timeline=draw('time_series.timeline',(pd.date_range('2020-01-01',periods=6),X),ax,overrides={'rolling_window':3})
        self.assertTrue(timeline.data.trailing_mean.iloc[:2].isna().all());self.assertAlmostEqual(timeline.data.trailing_mean.iloc[2],7/3)
        from statsmodels.tsa.stattools import pacf
        values=np.sin(np.arange(60))
        result=draw('time_series.autocorrelation',(values,),ax,overrides={'kind':'pacf','nlags':5})
        np.testing.assert_allclose(result.data.coefficient,pacf(values,nlags=5,method='ywm'))
        result=draw('selection.feature_importance',(['a','b'],[.1,-.2],[.01,.02]),ax)
        self.assertEqual(result.data.feature.tolist(),['a','b']);self.assertTrue((result.data.importance<0).any())
        curve=draw('selection.validation_curve',([1,2,10],[.9,.8,.8],[.5,.6,.7]),ax)
        self.assertEqual(curve.diagnostics['x_spacing'],'original numeric values')

    def test_skeleton_composition_imports_offline_and_reuses_supplied_axes(self):
        from src.datasets import uci,adult
        with patch.object(uci,'urlopen',side_effect=AssertionError('Import must be offline')),patch.object(adult,'urlopen',side_effect=AssertionError('Import must be offline')):
            for domain in ['numeric','categorical','quality','relationships','clustering','decomposition','classification','selection','association','networks','time_series']:
                module=importlib.import_module('src.Visualization.skeleton.'+domain)
                for name in ('load_data','load_recipe','build_pipeline','main'):self.assertTrue(callable(getattr(module,name)))
                for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
                    if isinstance(node,ast.FunctionDef):
                        for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))
        from src.Visualization.skeleton.numeric import main
        data=pd.DataFrame({'a':np.arange(1.,20.)})
        fig,axes=plt.subplots(2,3);before=tuple(axes[1,2].get_children())
        with patch('src.Visualization.skeleton.numeric.load_data',return_value=data),patch.object(plt,'subplots',side_effect=AssertionError('Supplied axes reused')):
            result=main(axes=axes)
        self.assertIs(result.axes,axes);self.assertEqual(len(fig.axes),6);self.assertEqual(tuple(axes[1,2].get_children()),before)
        self.assertEqual(len(result.panels),4);json.loads(result.to_json())

if __name__=='__main__':unittest.main()
