"""분포 QQ 분위수·지지/추정·회귀 subplot 수치·새44개 명세 계약 검증."""
import ast,importlib,json,sys
from dataclasses import fields
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from src.Recipes import load_profile
from src.Recipes.loader import _REGISTRY
from src.stats.regression.ols import run_ols
from src.stats.regression.collinearity import run_collinearity
from src.stats.regression.influence import run_influence
from src.MachineLearning.LinearModels.regression_workflow import run_regression_workflow


def qq_cases():
    return dict(normal=stats.norm(),student_t=stats.t(df=5),exponential=stats.expon(),gamma=stats.gamma(a=2),lognormal=stats.lognorm(s=1),weibull=stats.weibull_min(c=1.5),beta=stats.beta(a=2,b=2),uniform=stats.uniform(),chi_square=stats.chi2(df=5),logistic=stats.logistic(),poisson=stats.poisson(mu=1),binomial=stats.binom(n=10,p=.5),bernoulli=stats.bernoulli(p=.5))


def plot_cases():
    rng=np.random.RandomState(15);X=pd.DataFrame(rng.normal(size=(80,3)),columns=list('abc'));y=2*X.a-X.b+rng.normal(size=80)*.5;result=run_ols(X,y);future=X.iloc[:12]
    return dict(residual_fitted=(result,),scale_location=(result,),residual_order=(result,),cooks_distance=(result,),leverage=(result,),coefficient_intervals=(result,),observed_predicted=(y,result.fitted_values),prediction_interval=(result,future,y.iloc[:12]),partial_residual=(result,'a'),vif=(run_collinearity(X),))

class RegressionDistributionVisualizationsTest(unittest.TestCase):
    def tearDown(self):plt.close('all')

    def test_all_thirteen_qq_match_named_ppf_and_keep_axes(self):
        rng=np.random.RandomState(7)
        for name,distribution in qq_cases().items():
            with self.subTest(name=name):
                values=distribution.rvs(size=80,random_state=rng);module=importlib.import_module('src.Visualization.qq.'+name);figure,axes=plt.subplots(1,2);axes[1].plot([0,1]);before=tuple(axes[1].get_children());count=len(plt.get_fignums())
                with patch.object(plt,'subplots',side_effect=AssertionError('Plot must reuse ax')),patch.object(plt,'show',side_effect=AssertionError('No show')):
                    result=getattr(module,'plot_'+name)(values,ax=axes[0])
                p=(np.arange(80)+.5)/80;np.testing.assert_allclose(result.data.theoretical,distribution.ppf(p));np.testing.assert_array_equal(result.data.observed,np.sort(values));self.assertIs(result.axes,axes[0]);self.assertEqual(len(figure.axes),2);self.assertEqual(len(plt.get_fignums()),count);self.assertEqual(tuple(axes[1].get_children()),before);json.loads(result.to_json());figure.canvas.draw();plt.close(figure)

    def test_parameter_fit_support_violations_and_discrete_ties(self):
        _,ax=plt.subplots();normal=importlib.import_module('src.Visualization.qq.normal');values=np.array([1.,2.,4.,5.,7.])
        result=normal.plot_normal(values,ax=ax,overrides={'fit_parameters':True});params=result.diagnostics['distribution_parameters'];self.assertAlmostEqual(params['loc'],values.mean());self.assertAlmostEqual(params['scale'],values.std())
        exponential=importlib.import_module('src.Visualization.qq.exponential');result=exponential.plot_exponential([-1.,0.,1.,2.],ax=ax);self.assertEqual(result.diagnostics['outside_support'],1)
        with self.assertRaises(ValueError):exponential.plot_exponential([-1.,0.,1.,2.],ax=ax,overrides={'fit_parameters':True})
        poisson=importlib.import_module('src.Visualization.qq.poisson');result=poisson.plot_poisson([0,1,1,2,3],ax=ax,overrides={'fit_parameters':True});self.assertAlmostEqual(result.diagnostics['distribution_parameters']['mu'],1.4)
        with self.assertRaises(ValueError):poisson.plot_poisson([0.,.5,1.],ax=ax)
        bernoulli=importlib.import_module('src.Visualization.qq.bernoulli');result=bernoulli.plot_bernoulli([0,0,0,0,1],ax=ax,overrides={'success_probability':.1,'reference':'quartile'});self.assertIsNone(result.diagnostics['reference_slope']);self.assertTrue(result.warnings)
        with self.assertRaises(ValueError):normal.plot_normal(values,ax=ax,overrides={'scale':0.})
        with self.assertRaises(ValueError):normal.plot_normal([1.,1.,1.],ax=ax,overrides={'fit_parameters':True})
        for name in ('gamma','lognormal','weibull','beta','student_t','logistic','uniform','chi_square'):
            distribution=qq_cases()[name];module=importlib.import_module('src.Visualization.qq.'+name);values=distribution.rvs(size=60,random_state=17);json.loads(getattr(module,'plot_'+name)(values,ax=ax,overrides={'fit_parameters':True}).to_json())

    def test_all_ten_regression_panels_keep_ax_and_return_actual_data(self):
        for name,args in plot_cases().items():
            with self.subTest(name=name):
                module=importlib.import_module('src.Visualization.regression.'+name);figure,axes=plt.subplots(1,2);before=tuple(axes[1].get_children())
                result=getattr(module,'plot_'+name)(*args,ax=axes[0]);self.assertIs(result.axes,axes[0]);self.assertTrue(result.artists);self.assertEqual(tuple(axes[1].get_children()),before);self.assertEqual(len(figure.axes),2);payload=json.loads(result.to_json());self.assertFalse(set(payload)&{'figure','axes','artists'});figure.canvas.draw();plt.close(figure)

    def test_regression_coefficient_interval_cook_and_prediction_reference(self):
        cases=plot_cases();result=cases['residual_fitted'][0];_,ax=plt.subplots()
        from src.Visualization.regression.coefficient_intervals import plot_coefficient_intervals
        from src.Visualization.regression.cooks_distance import plot_cooks_distance
        from src.Visualization.regression.prediction_interval import plot_prediction_interval
        from src.Visualization.regression.observed_predicted import plot_observed_predicted
        output=plot_coefficient_intervals(result,ax=ax);np.testing.assert_allclose(output.data[['lower','upper']],result.model.conf_int())
        cook=plot_cooks_distance(result,ax=ax);np.testing.assert_allclose(cook.data.cooks_distance,run_influence(result).table.cooks_distance);self.assertAlmostEqual(cook.diagnostics['threshold'],4/80)
        interval=plot_prediction_interval(*cases['prediction_interval'],ax=ax);self.assertTrue((interval.data.obs_ci_lower<=interval.data.mean_ci_lower).all());self.assertTrue((interval.data.obs_ci_upper>=interval.data.mean_ci_upper).all());np.testing.assert_allclose(interval.data['mean'],result.model.predict(result.design.iloc[:12]))
        observed=plot_observed_predicted(*cases['observed_predicted'],ax=ax,overrides={'sample_role':'train'});self.assertAlmostEqual(observed.diagnostics['r2'],result.fit_metrics['r2'])

    def test_new_profiles_dtos_seven_sections_and_all_function_contracts(self):
        keys=[key for key in _REGISTRY if key.startswith(('regression-','naive-bayes-','plot-qq-','plot-regression-'))]
        self.assertEqual(len(keys),48)
        for key in keys:
            path,profile_name,_,_=_REGISTRY[key];module=importlib.import_module(path);source=Path(module.__file__).read_text();profile=load_profile(key)
            self.assertIsInstance(profile,getattr(module,profile_name))
            for cls in (getattr(module,profile_name),getattr(module,profile_name.replace('Profile','OutputDTO'))):
                for field in fields(cls):self.assertIn(field.name+' (',cls.__doc__)
            for index,section in enumerate(('Imports','Input Profile','Output DTO','Input Validation','Individual Model Runners','Public Runner','Example Execution'),1):self.assertIn(str(index)+'. '+section,source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))
            if sys.version_info>=(3,8):ast.parse(source,feature_version=(3,7))

if __name__=='__main__':unittest.main()
