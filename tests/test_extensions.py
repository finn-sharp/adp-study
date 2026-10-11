"""회귀 정확도·10종 리샘플링·유형 유지·CV 누수·시간 분할 검증."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.MachineLearning.Regression import (run_multiple_regression, run_polynomial_regression,
    run_nonlinear_regression, MultipleRegressionProfile, PolynomialRegressionProfile, NonlinearRegressionProfile)
from src.Preprocessing import (SamplingProfile, make_sampler, resample_training_data,
    build_sampling_pipeline, OVER_METHODS, UNDER_METHODS)
from src.TimeSeries.arima import ARIMAProfile, SARIMAProfile, run_arima, run_sarima


class ExtensionsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(18)
        self.X, self.y = make_classification(n_samples=180, n_features=4, n_redundant=0,
            weights=[.75,.25], class_sep=.8, random_state=18)

    def test_multiple_known_coefficients(self):
        X = self.rng.normal(size=(90,3)); y = 4+X.dot([2,-3,.5])
        r = run_multiple_regression(X,y)
        np.testing.assert_allclose(r.model.named_steps['model'].coef_,[2,-3,.5],atol=1e-9)
        self.assertLess(r.test_metrics['rmse'],1e-9)
        with self.assertRaises(ValueError):run_multiple_regression(X[:,:1],y)

    def test_polynomial_known_surface_and_cv(self):
        X = self.rng.normal(size=(90,2));y=1+2*X[:,0]**2+3*X[:,0]*X[:,1]
        p=PolynomialRegressionProfile(param_grid={'polynomial__degree':[1,2]})
        r=run_polynomial_regression(X,y,p)
        self.assertEqual(r.best_params['polynomial__degree'],2)
        self.assertLess(r.test_metrics['rmse'],1e-9)
        with self.assertRaises(ValueError):run_polynomial_regression(X,y,PolynomialRegressionProfile(degree=0))

    def test_nonlinear_known_parameters_and_clone(self):
        X=np.linspace(-1,1,90).reshape(-1,1)
        f=lambda z,a,b,c:a*np.exp(b*z[0])+c
        r=run_nonlinear_regression(X,f(X.T,2.,.7,.3),f,NonlinearRegressionProfile(initial_params=(1.,.5,0.)))
        np.testing.assert_allclose(r.model.named_steps['model'].parameters_,[2.,.7,.3],atol=1e-6)
        self.assertLess(r.test_metrics['rmse'],1e-7)
        clone(r.model)
        with self.assertRaises(ValueError):run_nonlinear_regression(X,X[:,0],f,NonlinearRegressionProfile())

    def test_nonlinear_invalid_output(self):
        X=np.arange(30.).reshape(-1,1)
        with self.assertRaises(ValueError):run_nonlinear_regression(X,X[:,0],lambda z,a:1.,NonlinearRegressionProfile(initial_params=(1.,)))

    def test_all_ten_samplers(self):
        self.assertEqual(len(OVER_METHODS),5); self.assertEqual(len(UNDER_METHODS),5)
        for method in OVER_METHODS+UNDER_METHODS:
            with self.subTest(method=method):
                X=self.X; kind='numeric';cats=None
                if method=='smotenc':
                    X=pd.DataFrame({'number':self.X[:,0],'category':np.where(self.X[:,1]>0,'A','B')});kind='mixed';cats=[1]
                elif method=='smoten':
                    X=np.where(self.X>0,'A','B');kind='categorical'
                r=resample_training_data(X,self.y,SamplingProfile(method=method,data_kind=kind,categorical_features=cats,neighbors=3))
                self.assertEqual(len(r.X),len(r.y));self.assertEqual(sum(r.counts_after.values()),len(r.y))
                if method in OVER_METHODS:self.assertGreaterEqual(len(r.y),len(self.y))
                else:self.assertLessEqual(len(r.y),len(self.y))
                if method=='smotenc':self.assertLessEqual(set(r.X['category']),{'A','B'})
                if method=='smoten':self.assertLessEqual(set(np.asarray(r.X).ravel()),{'A','B'})

    def test_random_sampling_categorical(self):
        X=pd.DataFrame({'category':np.where(self.X[:,0]>0,'low','high')})
        for method in ('random_over','random_under'):
            r=resample_training_data(X,self.y,SamplingProfile(method=method,data_kind='categorical'))
            self.assertEqual(len(set(r.counts_after.values())),1)
            self.assertLessEqual(set(r.X['category']),{'low','high'})

    def test_sampling_validation(self):
        for p in (SamplingProfile(method='smote',data_kind='categorical'),
                  SamplingProfile(method='smotenc',data_kind='numeric'),
                  SamplingProfile(method='smoten',data_kind='numeric'),
                  SamplingProfile(method='random_over',parameters={'random_state':1}),
                  SamplingProfile(neighbors=0)):
            with self.assertRaises(ValueError):make_sampler(p)
        with self.assertRaises(ValueError):resample_training_data(self.X,np.linspace(0,1,len(self.y)))
        with self.assertRaises(ValueError):resample_training_data(self.X,self.y,SamplingProfile(method='smotenc',data_kind='mixed',categorical_features=[7]))
        X=self.X.copy();X[0,0]=np.inf
        with self.assertRaises(ValueError):resample_training_data(X,self.y)
        with self.assertRaises(ValueError):resample_training_data(self.X,self.y,SamplingProfile(method='smote',neighbors=90))

    def test_pipeline_fold_local_and_inference_length(self):
        pipe=build_sampling_pipeline(LogisticRegression(max_iter=500),SamplingProfile(),StandardScaler())
        cv=StratifiedKFold(3,shuffle=True,random_state=1)
        for train,test in cv.split(self.X,self.y):
            fitted=clone(pipe).fit(self.X[train],self.y[train])
            np.testing.assert_allclose(fitted.named_steps['preprocess'].mean_,self.X[train].mean(axis=0))
            self.assertLess(fitted.named_steps['sampler'].sample_indices_.max(),len(train))
            self.assertEqual(len(fitted.predict(self.X[test])),len(test))
        search=GridSearchCV(pipe,{'model__C':[.1,1.]},cv=cv,error_score='raise').fit(self.X,self.y)
        self.assertTrue(np.isfinite(search.best_score_))

    def test_categorical_pipeline_after_encoding(self):
        X=np.where(self.X>0,'A','B')
        pipe=build_sampling_pipeline(LogisticRegression(max_iter=500),SamplingProfile(method='smoten',data_kind='categorical',neighbors=3),after_sampling=OneHotEncoder(handle_unknown='ignore'))
        pipe.fit(X[:140],self.y[:140])
        self.assertEqual(len(pipe.predict(X[140:])),40)

    def test_time_wrappers_and_holdout_invariance(self):
        data=pd.Series(np.sin(np.arange(72)*2*np.pi/6)+self.rng.normal(0,.1,72))
        p=SARIMAProfile(order=(0,0,0),seasonal_order=(1,0,0,6),holdout=6)
        r=run_sarima(data,p)
        changed=data.copy();changed.iloc[-6:]+=100
        other=run_sarima(changed,p)
        np.testing.assert_allclose(r.predictions,other.predictions)
        self.assertTrue(r.predictions.index.equals(data.iloc[-6:].index))
        self.assertTrue((r.intervals.lower<=r.intervals.upper).all())
        a=run_arima(data,ARIMAProfile(order=(1,0,0),holdout=6))
        self.assertEqual(len(a.predictions),6)
        with self.assertRaises(ValueError):run_sarima(data,SARIMAProfile(seasonal_order=(0,0,0,6)))
        with self.assertRaises(ValueError):run_sarima(data,SARIMAProfile(seasonal_order=(1,0,0,50)))
        with self.assertRaises(ValueError):run_arima(data,ARIMAProfile(method='sarimax'))

if __name__=='__main__':unittest.main()
