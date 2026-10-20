"""도메인 이관의 수치 참조·경계·설정·자료 분리 회귀 검증."""
import ast
import importlib
import json
from pathlib import Path
import unittest
import sys
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.stats.survival.kaplan_meier import run_kaplan_meier, KaplanMeierProfile
from src.stats.survival.rmst import run_rmst
from src.stats.survival.logrank import run_logrank
from src.MachineLearning.NonlinearModels.nonlinear_regression import run_nonlinear_regression
from src.TimeSeries.arima import run_arima
from src.TimeSeries.sarima import run_sarima

ROOT=Path(__file__).resolve().parents[1]


class DomainExtensionTest(unittest.TestCase):
    def test_contract_and_recipes(self):
        api=json.loads((ROOT/'docs/DOMAIN_EXTENSION_API.json').read_text())
        for key,item in api.items():
            with self.subTest(key=key):
                p=load_profile(key)
                self.assertEqual(type(p).__name__,item['profile'])
                source=(ROOT/(item['module'].replace('.','/')+'.py')).read_text()
                ast.parse(source,**({'feature_version':(3,7)} if sys.version_info>=(3,8) else {}))
                for n,title in enumerate(['Imports','Input Profile','Output DTO','Input Validation','Individual Model Runners','Public Runner','Example Execution'],1):
                    self.assertIn(str(n)+'. '+title,source)
                self.assertIn('Raises:',source)
        with self.assertRaises(ValueError): run_kaplan_meier([1],[1],KaplanMeierProfile(),overrides={})
        with self.assertRaises(ValueError): run_kaplan_meier([1],[1],overrides={'typo':1})

    def test_km_hand_and_degenerate_samples(self):
        r=run_kaplan_meier([1,2,2,3,4],[1,1,0,0,1],overrides={'evaluation_times':[4,0,2,2]})
        np.testing.assert_array_equal(r.at_risk,[5,4,2,1])
        np.testing.assert_allclose(r.survival,[.8,.6,.6,0])
        np.testing.assert_allclose(r.survival_at_times,[0,1,.6,.6])
        self.assertEqual(r.median_survival_time,4)
        self.assertTrue(np.all(r.confidence_lower<=r.survival))
        self.assertTrue(np.all(r.confidence_upper>=r.survival))
        z=run_kaplan_meier([0,1],[1,0],overrides={'evaluation_times':[0]})
        self.assertEqual(z.survival_at_times[0],.5)
        censored=run_kaplan_meier([1,2],[0,0])
        self.assertIsNone(censored.median_survival_time)
        self.assertIsNone(json.loads(censored.to_json())['median_survival_time'])
        for t,e in [([],[]),([1],[2]),([-1],[1]),([np.nan],[0]),([1,2],[1])]:
            with self.assertRaises((ValueError,TypeError)): run_kaplan_meier(t,e)
        with self.assertRaises(ValueError): run_kaplan_meier([1],[1],overrides={'evaluation_times':[2]})

    def test_rmst_hand(self):
        # S(t)=1 on [0,1), .8 on [1,2), .6 on [2,3).
        r=run_rmst([1,2,2,3,4],[1,1,0,0,1],overrides={'tau':3})
        self.assertAlmostEqual(r.estimate,2.4)
        self.assertEqual(run_rmst([0,2],[1,0],overrides={'tau':2}).estimate,1.)
        with self.assertRaises(ValueError):run_rmst([1],[0],overrides={'tau':2})

    def test_survival_reference(self):
        try: from lifelines import KaplanMeierFitter
        except ImportError: self.skipTest('optional lifelines not installed')
        from lifelines.statistics import multivariate_logrank_test
        root=ROOT/'기출문제/모의고사/생존분석_통계/data'
        d=pd.read_csv(root/'subscription.csv')
        for group,part in d.groupby('group'):
            r=run_kaplan_meier(part.time,part.event)
            fit=KaplanMeierFitter().fit(part.time,part.event)
            np.testing.assert_allclose(r.survival,fit.predict(r.times),atol=1e-12)
            ci=fit.confidence_interval_.loc[r.times].to_numpy()
            np.testing.assert_allclose(r.confidence_lower,ci[:,0],atol=1e-12)
            np.testing.assert_allclose(r.confidence_upper,ci[:,1],atol=1e-12)
        r=run_logrank(d.time,d.event,d.group,overrides={'pairwise':True})
        ref=multivariate_logrank_test(d.time,d.group,d.event)
        self.assertAlmostEqual(r.statistic,ref.test_statistic,places=9)
        self.assertAlmostEqual(r.p_value,ref.p_value,places=12)
        self.assertEqual(len(r.pairwise),3)
        for row in r.pairwise:self.assertGreaterEqual(row['adjusted_p_value'],row['p_value'])
        with self.assertRaises(ValueError):run_logrank([1,2],[0,0],['a','b'])

    def test_cox_reference_and_prediction(self):
        try: from lifelines import CoxPHFitter
        except ImportError: self.skipTest('optional lifelines not installed')
        from src.stats.survival.cox import run_cox
        d=pd.read_csv(ROOT/'기출문제/모의고사/생존분석_통계/data/equipment.csv').drop(columns='id')
        new=d.drop(columns=['time','event']).iloc[:2].copy()
        r=run_cox(d,new_data=new,overrides={'prediction_times':[12.,24.]})
        ref=CoxPHFitter().fit(d,duration_col='time',event_col='event')
        np.testing.assert_allclose(r.coefficients['coef'],ref.params_,rtol=1e-10)
        np.testing.assert_allclose(r.predictions,ref.predict_survival_function(new,times=[12,24]))
        self.assertNotIn('model',r.to_dict())
        self.assertTrue(np.all(r.ph_diagnostics.adjusted_p_value>=r.ph_diagnostics.p))
        with self.assertRaises(ValueError):run_cox(d,new_data=new)

    def test_nonlinear_exact_and_holdout(self):
        x=np.linspace(0,3,50); y=2*np.exp(.4*x)+1
        r=run_nonlinear_regression(x[:40],y[:40],overrides={'initial_parameters':[1,.2,0]},x_test=x[40:],y_test=y[40:])
        np.testing.assert_allclose(r.coefficients,[2,.4,1],atol=1e-7)
        self.assertLess(r.test_rmse,1e-8)
        changed=run_nonlinear_regression(x[:40],y[:40],overrides={'initial_parameters':[1,.2,0]},x_test=x[40:],y_test=y[40:]+100)
        np.testing.assert_array_equal(r.coefficients,changed.coefficients)
        for name,coef,fn in [('logistic',[4,2,1],lambda x:4/(1+np.exp(-2*(x-1)))),('michaelis_menten',[3,2],lambda x:3*x/(2+x))]:
            r=run_nonlinear_regression(x,fn(x),overrides={'model_name':name,'initial_parameters':coef})
            np.testing.assert_allclose(r.coefficients,coef,atol=1e-6)
        with self.assertRaises(ValueError):run_nonlinear_regression(x,y,overrides={'initial_parameters':[1,2]})

    def test_timeseries_reference_and_indices(self):
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y=pd.Series(np.random.RandomState(12).normal(size=90),index=pd.date_range('2020-01-01',periods=90,freq='D'))
        r=run_arima(y,overrides={'steps':4,'trend':'c','candidate_orders':[[1,0,0],[2,0,0]]})
        ref=ARIMA(y,order=tuple(r.selected_order),trend='c').fit(method_kwargs={'maxiter':100})
        np.testing.assert_allclose(r.predictions,ref.forecast(4),rtol=1e-8)
        self.assertEqual(r.forecast_index[0],y.index[-1]+pd.Timedelta(days=1))
        self.assertEqual(r.aic,min(row['aic'] for row in r.candidates if row['converged']))
        rs=run_sarima(y,overrides={'steps':3,'seasonal_order':[0,0,0,12]})
        refs=SARIMAX(y,order=(1,0,0),seasonal_order=(0,0,0,12)).fit(disp=False,maxiter=100)
        np.testing.assert_allclose(rs.predictions,refs.forecast(3),rtol=1e-8)
        self.assertEqual(len(rs.prediction_lower),3)
        exog=pd.DataFrame({'x':np.arange(90.)},index=y.index)
        future=pd.DataFrame({'x':np.arange(90.,94.)},index=pd.date_range(y.index[-1]+pd.Timedelta(days=1),periods=4,freq='D'))
        ex=run_arima(y,exog=exog,future_exog=future,overrides={'steps':4})
        self.assertEqual(len(ex.predictions),4)
        wrong=future.copy();wrong.index=wrong.index+pd.Timedelta(days=1)
        with self.assertRaises(ValueError):run_arima(y,exog=exog,future_exog=wrong,overrides={'steps':4})
        with self.assertRaises(ValueError):run_arima(y.drop(y.index[20]))
        with self.assertRaises(ValueError):run_arima(y,exog=np.ones((90,1)))
        with self.assertRaises(ValueError):run_arima(y,overrides={'order':[1.1,0,0]})

    def test_resamplers_and_pipeline(self):
        try: import imblearn
        except ImportError:self.skipTest('optional imbalanced-learn not installed')
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score,StratifiedKFold
        from imblearn.pipeline import Pipeline
        from src.Preprocessing.resampling._common import SPECS,load_resampler
        X,y=make_classification(n_samples=180,n_features=5,n_redundant=0,weights=[.75,.25],random_state=1234)
        for name,info in SPECS.items():
            a=X.copy()
            if info['kind']=='mixed_declared':a[:,0]=(a[:,0]>0).astype(int)
            if info['kind']=='categorical':a=np.where(a>0,'A','B')
            runner=getattr(importlib.import_module('src.Preprocessing.resampling.'+name),'run_'+name)
            with self.subTest(name=name):
                r=runner(a,y)
                self.assertEqual(sum(r.class_counts_after.values()),len(r.y_resampled))
                self.assertEqual(len(r.X_resampled),len(r.y_resampled))
                self.assertNotIn('sampler',r.to_dict())
                if r.sample_indices is not None:np.testing.assert_array_equal(np.asarray(r.X_resampled),a[r.sample_indices])
                json.loads(r.to_json())
        from src.Preprocessing.resampling.random_over import run_random_over
        mixed=pd.DataFrame({'category':['A']*8+['B']*4,'value':np.arange(12)})
        m=run_random_over(mixed,np.r_[np.zeros(8),np.ones(4)])
        self.assertEqual(set(m.X_resampled.category),{'A','B'})
        pipe=Pipeline([('sample',load_resampler('random_over')),('model',LogisticRegression())])
        scores=cross_val_score(pipe,X,y,cv=StratifiedKFold(3),scoring='f1')
        self.assertTrue(np.isfinite(scores).all())
        self.assertFalse(hasattr(pipe.named_steps['sample'],'sample_indices_'))
        from src.Preprocessing.resampling.smote import run_smote
        with self.assertRaises(ValueError):run_smote(mixed,np.r_[np.zeros(8),np.ones(4)])


if __name__=='__main__': unittest.main()
