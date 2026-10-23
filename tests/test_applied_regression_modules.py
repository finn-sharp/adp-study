"""응용 모듈의 참조값·선택/CV·시간 미래 격리·분포 조건·Axes 계약."""
import ast,importlib,json,sys
from pathlib import Path
from dataclasses import fields
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from src.Recipes import load_profile
from src.stats.regression.information_selection import run_information_selection,InformationSelector
from src.MachineLearning.LinearModels._regression_estimator import RegressionEstimator
from src.MachineLearning.LinearModels.weighted_regression import run_weighted_regression
from src.TimeSeries.rolling_selection import run_rolling_selection

class AppliedModulesTest(unittest.TestCase):
    def setUp(self):
        r=np.random.RandomState(11);self.X=pd.DataFrame(r.normal(size=(120,4)),columns=list('abcd'));self.y=2+3*self.X.a+r.normal(size=120)*.2
    def test_information_selector_criterion_trace_and_clone(self):
        out=run_information_selection(self.X,self.y);self.assertIn('a',out.selected_features)
        expected=sm.OLS(self.y,sm.add_constant(self.X[out.selected_features])).fit();self.assertAlmostEqual(out.model.bic,expected.bic)
        scores=out.history.loc[out.history.accepted,'bic'];self.assertTrue((scores.diff().dropna()<0).all());self.assertLessEqual(len(out.selected_features),4)
        selector=clone(out.selector).fit(self.X,self.y);self.assertEqual(selector.transform(self.X[:2]).shape[1],len(out.selected_features))
        with self.assertRaises(ValueError):selector.transform(self.X[list('dcba')])
        model=clone(RegressionEstimator(information_criterion='bic')).fit(self.X,self.y);self.assertEqual(model.predict(self.X[:3]).shape,(3,))
        with self.assertRaises(ValueError):RegressionEstimator(information_criterion='aic',target_method='log').fit(self.X,np.exp(self.y))
    def test_glm_all_families_match_native_and_domains(self):
        r=np.random.RandomState(19)
        for name,y,family in [('logistic',r.binomial(1,.5,120),sm.families.Binomial()),('poisson',r.poisson(3,size=120),sm.families.Poisson()),('gamma',np.exp(self.y/5),sm.families.Gamma(link=getattr(sm.families.links,'Log',sm.families.links.log)()))]:
            module=importlib.import_module('src.MachineLearning.GeneralizedLinearModels.'+name);out=getattr(module,'run_'+name)(self.X,y,self.X[:3]);native=sm.GLM(y,sm.add_constant(self.X),family=family).fit();np.testing.assert_allclose(out.coefficients.coefficient,native.params,rtol=1e-8);self.assertAlmostEqual(out.fit_metrics['aic'],native.aic);np.testing.assert_allclose(out.predictions,native.predict(sm.add_constant(self.X[:3],has_constant='add')));json.loads(out.to_json());self.assertTrue((out.diagnostics.cooks_distance>=0).all())
        from src.MachineLearning.GeneralizedLinearModels.poisson import run_poisson
        from src.MachineLearning.GeneralizedLinearModels.logistic import run_logistic
        with self.assertRaises(ValueError):run_logistic(pd.DataFrame({'x':np.r_[np.arange(-20,0),np.arange(1,21)]}),np.r_[np.zeros(20),np.ones(20)])
        with self.assertRaises(ValueError):run_poisson(self.X,np.arange(120)+.2)
        offset=np.log(np.linspace(1,2,120));out=run_poisson(self.X,np.random.RandomState(5).poisson(3,size=120),self.X[:3],offset=offset,new_offset=offset[:3]);self.assertEqual(out.predictions.shape,(3,))
        with self.assertRaises(ValueError):run_poisson(self.X,np.arange(120),self.X[:3],offset=offset)
    def test_fgls_weights_and_predictions_are_training_only(self):
        out=run_weighted_regression(self.X,self.y,self.X[:4]);native=sm.WLS(self.y,out.model._matrix(self.X),weights=out.weights).fit(cov_type='HC3',use_t=True);np.testing.assert_allclose(out.coefficients.coefficient,native.params);self.assertLessEqual(out.weights.max()/out.weights.min(),100+1e-8);json.loads(out.to_json())
        original=out.model.imputer_.statistics_.copy();out.model.predict(self.X.iloc[:4]+10000);np.testing.assert_array_equal(out.model.imputer_.statistics_,original)
        self.assertEqual(clone(out.model).fit(self.X,self.y).predict(self.X[:2]).shape,(2,))
    def test_rolling_future_cannot_change_selection_or_fit(self):
        y=pd.Series(np.random.RandomState(9).normal(size=95),index=pd.date_range('2020-01-01',periods=95))
        settings={'candidate_orders':[[0,0,0],[1,0,0]],'folds':2,'horizon':5,'validation_size':5};a=run_rolling_selection(y,overrides=settings);y.iloc[-5:]+=100;b=run_rolling_selection(y,overrides=settings)
        self.assertEqual(a.selected_order,b.selected_order);np.testing.assert_array_equal(a.predictions,b.predictions);self.assertNotEqual(a.test_metrics,b.test_metrics)
        for fold in a.folds:self.assertLess(max(fold['train_indices']),min(fold['validation_indices']));self.assertLess(max(fold['validation_indices']),min(a.test_indices))
        with self.assertRaises(ValueError):run_rolling_selection(pd.Series(np.arange(40.),index=pd.date_range('2020-01-01',periods=41).delete(5)))
    def test_contracts_and_plot_ownership(self):
        keys=['regression-information-selection','regression-weighted','glm-logistic','glm-poisson','glm-gamma','timeseries-rolling-selection','plot-regression-model-comparison','plot-regression-glm-residuals']
        from src.Recipes.loader import _REGISTRY
        for key in keys:
            module=importlib.import_module(_REGISTRY[key][0]);p=load_profile(key);source=Path(module.__file__).read_text()
            for cls in (type(p),getattr(module,type(p).__name__.replace('Profile','OutputDTO'))):
                for f in fields(cls):self.assertIn(f.name+' (',cls.__doc__)
            for i in range(1,8):self.assertIn('# '+str(i)+'. ',source)
            if sys.version_info>=(3,8):ast.parse(source,feature_version=(3,7))
        import matplotlib;matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from src.Visualization.regression.model_comparison import plot_model_comparison
        fig,axes=plt.subplots(1,2);before=tuple(axes[1].get_children());result=plot_model_comparison(pd.DataFrame({'name':['base','new'],'cv_r2':[.3,.5]}),ax=axes[0]);self.assertIs(result.axes,axes[0]);self.assertEqual(tuple(axes[1].get_children()),before);json.loads(result.to_json())
        from src.Visualization.regression.glm_residuals import plot_glm_residuals
        from src.MachineLearning.GeneralizedLinearModels.logistic import run_logistic
        glm=run_logistic(self.X,np.random.RandomState(31).binomial(1,.5,len(self.X)));panel=plot_glm_residuals(glm,ax=axes[0]);np.testing.assert_allclose(panel.data.pearson_residual,glm.model.resid_pearson);self.assertIs(panel.axes,axes[0]);json.loads(panel.to_json());plt.close(fig)

if __name__=='__main__':unittest.main()
