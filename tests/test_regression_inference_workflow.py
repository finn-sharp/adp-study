"""
================================================================================
test_regression_inference_workflow.py - 회귀 추론 참조값·rank/행 대응·전처리/CV 누수·명시 관측 처리 검증.
================================================================================
설명:
    - 회귀 추론 참조값·rank/행 대응·전처리/CV 누수·명시 관측 처리 검증.
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
import unittest,json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import linear_reset,het_breuschpagan,het_white,acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor,OLSInfluence
from sklearn.model_selection import KFold,cross_validate
from sklearn.base import clone
from src.stats.regression.ols import run_ols,OLSProfile
from src.stats.regression.joint_f import run_joint_f
from src.stats.regression.nested_f import run_nested_f
from src.stats.regression.linearity import run_linearity
from src.stats.regression.normality import run_normality
from src.stats.regression.heteroscedasticity import run_heteroscedasticity
from src.stats.regression.independence import run_independence
from src.stats.regression.collinearity import run_collinearity
from src.stats.regression.influence import run_influence
from src.Preprocessing.regression.polynomial import run_polynomial,build_polynomial_transformer
from src.Preprocessing.regression.target_transform import run_target_transform,build_target_transformer
from src.Preprocessing.regression.vif_selection import run_vif_selection,build_vif_selector
from src.Preprocessing.regression.influence_filter import run_influence_filter
from src.MachineLearning.LinearModels.regression_workflow import run_regression_workflow,build_regression_estimator

class RegressionInferenceWorkflowTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.RandomState(7);self.X=pd.DataFrame(rng.normal(size=(120,3)),columns=list('abc'));self.y=pd.Series(2*self.X.a-self.X.b+.5*self.X.c+rng.normal(size=120)*.3,index=self.X.index)

    def test_ols_coefficients_t_f_r2_intervals_and_robust_covariance(self):
        design=sm.add_constant(self.X)
        for cov in ('nonrobust','HC3'):
            result=run_ols(self.X,self.y,X_new=self.X.iloc[:5],overrides={'cov_type':cov});native=sm.OLS(self.y,design).fit(cov_type=cov,use_t=True)
            np.testing.assert_allclose(result.coefficients.coefficient,native.params);np.testing.assert_allclose(result.coefficients.t_statistic,native.tvalues);np.testing.assert_allclose(result.coefficients.p_value,native.pvalues);np.testing.assert_allclose(result.coefficients[['ci_lower','ci_upper']],native.conf_int())
            self.assertAlmostEqual(result.fit_metrics['f_statistic'],native.fvalue);self.assertAlmostEqual(result.fit_metrics['f_p_value'],native.f_pvalue);self.assertAlmostEqual(result.fit_metrics['r2'],native.rsquared);self.assertAlmostEqual(result.fit_metrics['adjusted_r2'],native.rsquared_adj)
            np.testing.assert_allclose(result.new_predictions,native.get_prediction(design.iloc[:5]).summary_frame())
            self.assertNotIn('model',json.loads(result.to_json()))

    def test_joint_f_t_squared_and_nested_models_require_same_response_and_rows(self):
        full=run_ols(self.X,self.y);R=np.array([[0.,1.,0.,0.]])
        joint=run_joint_f(full,R);self.assertAlmostEqual(joint.tests.statistic.iloc[0],full.coefficients.t_statistic.iloc[1]**2)
        small=run_ols(self.X[['a','b']],self.y);nested=run_nested_f(small,full);native=full.model.compare_f_test(small.model)
        self.assertAlmostEqual(nested.tests.statistic.iloc[0],native[0]);self.assertAlmostEqual(nested.tests.p_value.iloc[0],native[1])
        with self.assertRaises(ValueError):run_nested_f(small,run_ols(self.X,self.y+1))
        with self.assertRaises(ValueError):run_nested_f(full,small)
        with self.assertRaises(ValueError):run_nested_f(small,run_ols(self.X,self.y,overrides={'cov_type':'HC3'}))

    def test_linearity_normality_heteroscedasticity_and_ordered_independence(self):
        result=run_ols(self.X,self.y);model=result.model
        reset=run_linearity(result);native=linear_reset(model,power=3,test_type='fitted',use_f=True);self.assertAlmostEqual(reset.tests.p_value.iloc[0],float(native.pvalue))
        for method,reference in [('breusch_pagan',het_breuschpagan),('white',het_white)]:
            output=run_heteroscedasticity(result,overrides={'method':method});native=reference(model.resid,model.model.exog);self.assertAlmostEqual(output.tests.p_value.iloc[0],native[1]);self.assertAlmostEqual(output.tests.p_value.iloc[1],native[3])
        normal=run_normality(result,overrides={'method':'shapiro'});self.assertAlmostEqual(normal.tests.p_value.iloc[0],stats.shapiro(model.resid)[1])
        for method in ('jarque_bera','dagostino','anderson'):json.loads(run_normality(result,overrides={'method':method}).to_json())
        descriptive=run_independence(result);self.assertIsNone(descriptive.tests.p_value.iloc[0]);self.assertIsNone(descriptive.tests.reject_h0.iloc[0]);self.assertEqual(len(descriptive.tests),1)
        ordered=run_independence(result,overrides={'ordered':True,'nlags':2});native=acorr_breusch_godfrey(model,nlags=2);self.assertAlmostEqual(ordered.tests.p_value.iloc[1],native[1])
        self.assertEqual(len(run_independence(result,overrides={'ordered':True,'method':'ljung_box','nlags':2}).tests),2)

    def test_vif_and_influence_match_projection_and_leave_one_out_reference(self):
        col=run_collinearity(self.X);design=sm.add_constant(self.X).values
        expected=[variance_inflation_factor(design,i) for i in range(1,4)];np.testing.assert_allclose(col.table.vif,expected)
        result=run_ols(self.X,self.y);influence=run_influence(result);native=OLSInfluence(result.model)
        hat=np.diag(design@np.linalg.inv(design.T@design)@design.T);np.testing.assert_allclose(influence.table.leverage,hat)
        cook=result.residuals**2/(design.shape[1]*result.model.mse_resid)*hat/(1-hat)**2;np.testing.assert_allclose(influence.table.cooks_distance,cook)
        np.testing.assert_allclose(influence.table.studentized_external,native.resid_studentized_external);np.testing.assert_allclose(influence.dfbetas,native.dfbetas)
        duplicate=self.X.copy();duplicate['duplicate']=duplicate.a;self.assertTrue(np.isinf(run_collinearity(duplicate).table.vif.iloc[0]))

    def test_rank_constants_definition_domains_and_schema(self):
        duplicate=self.X.assign(duplicate=self.X.a)
        with self.assertRaises(ValueError):run_ols(duplicate,self.y)
        with self.assertRaises(ValueError):run_ols(self.X.assign(constant=1.),self.y)
        with self.assertRaises(ValueError):run_ols(self.X,self.y,X_new=self.X[['b','a','c']])
        with self.assertRaises(ValueError):run_ols(self.X,self.y.set_axis(np.arange(1,121)))
        with self.assertRaises(ValueError):run_influence(run_ols(self.X,2*self.X.a))
        no_intercept=run_ols(self.X,self.y,overrides={'fit_intercept':False});self.assertEqual(no_intercept.fit_metrics['r2_kind'],'uncentered')
        with self.assertRaises(ValueError):run_heteroscedasticity(no_intercept)

    def test_target_transform_inverse_poly_names_limits_and_clone(self):
        y=np.array([1.,2.,4.,8.,16.])
        for method in ('none','log','log1p','yeo-johnson','box-cox'):
            output=run_target_transform(y,y+1,overrides={'method':method});np.testing.assert_allclose(output.transformer.inverse_transform(output.transformed_target),y,rtol=1e-7)
            clone(output.transformer).fit(y)
        with self.assertRaises(ValueError):run_target_transform([0,1,2],overrides={'method':'log'})
        poly=run_polynomial(self.X,self.X.iloc[:2]);self.assertEqual(poly.preprocessed_feature.shape,(120,9));self.assertIn('a b',poly.output_features)
        with self.assertRaises(ValueError):build_polynomial_transformer(overrides={'max_features':2}).fit(self.X)

    def test_vif_selector_preserve_and_new_rows_never_change_selection(self):
        data=self.X.assign(duplicate=self.X.a*2,constant=1.)
        selected=run_vif_selection(data,data+100,overrides={'preserve_columns':['a']})
        self.assertIn('a',selected.output_features);self.assertNotIn('duplicate',selected.output_features);self.assertNotIn('constant',selected.output_features)
        self.assertEqual(selected.selection_history.reason.tolist(),['constant','vif'])
        np.testing.assert_allclose(selected.transformer.final_vif_,[variance_inflation_factor(sm.add_constant(self.X).values,i) for i in range(1,4)])
        with self.assertRaises(ValueError):build_vif_selector(overrides={'preserve_columns':['a','duplicate']}).fit(data)
        with self.assertRaises(ValueError):selected.transformer.transform(data[data.columns[::-1]])

    def test_cook_flag_drop_refit_and_removal_guards(self):
        X=self.X.copy();X.iloc[0]=[30.,30.,30.];y=self.y.copy();y.iloc[0]+=100
        flag=run_influence_filter(X,y,overrides={'cook_threshold':1.});self.assertEqual(len(flag.removed_positions),0);self.assertTrue(flag.review_table.flag.iloc[0])
        dropped=run_influence_filter(X,y,overrides={'action':'drop','cook_threshold':1.});self.assertIn(0,dropped.removed_positions);self.assertEqual(len(dropped.cleaned_features),119)
        with self.assertRaises(ValueError):run_influence_filter(X,y,overrides={'action':'drop','cook_threshold':1.,'max_remove_fraction':0.})
        estimator=build_regression_estimator(influence_action='drop',cook_threshold=1.).fit(X,y)
        imputer=estimator.preprocessor_.named_transformers_['numeric'].named_steps['impute'].estimator_;np.testing.assert_allclose(imputer.statistics_,np.median(X.iloc[estimator.kept_positions_],axis=0));self.assertEqual(estimator.n_training_samples_,119)
        self.assertEqual(len(estimator.predict(X)),120)

    def test_cv_training_only_and_workflow_selection_uses_original_unit_holdout(self):
        X=self.X.copy();X.loc[4,'a']=np.nan;split=KFold(3,shuffle=True,random_state=10)
        output=cross_validate(build_regression_estimator(),X,self.y,cv=split,return_estimator=True)
        for estimator,(train,test) in zip(output['estimator'],split.split(X)):
            np.testing.assert_allclose(estimator.preprocessor_.named_transformers_['numeric'].named_steps['impute'].estimator_.statistics_,np.nanmedian(X.iloc[train],axis=0));self.assertEqual(estimator.n_training_samples_,len(train))
        workflow=run_regression_workflow(X,self.y,overrides={'candidates':[{'name':'baseline'},{'name':'poly','degree':2},{'name':'log','target_method':'log'}]})
        self.assertEqual(workflow.selected_name,workflow.comparison.loc[workflow.comparison.cv_score.idxmax(),'name']);self.assertTrue(workflow.comparison.loc[workflow.comparison.name=='log','error'].notna().all());self.assertFalse(set(workflow.train_indices)&set(workflow.test_indices))
        np.testing.assert_allclose(workflow.model.predict(X.iloc[workflow.test_indices]),workflow.predictions);json.loads(workflow.to_json())
        np.testing.assert_allclose(workflow.model.preprocessor_.named_transformers_['numeric'].named_steps['impute'].estimator_.statistics_,np.nanmedian(X.iloc[workflow.train_indices],axis=0))

    def test_custom_feature_groups_and_explicit_target_missing(self):
        from src.Preprocessing.feature_pipeline import make_feature_spec,build_feature_preprocessor
        from src.Preprocessing.numeric.imputation import build_numeric_imputation_transformer
        from src.Preprocessing.categorical.encoding import build_categorical_encoding_transformer
        X=self.X[['a']].assign(category=np.where(self.X.b>0,'high','low'))
        feature=make_feature_spec('amount',['a'],[('impute',build_numeric_imputation_transformer())]);category=make_feature_spec('category',['category'],[('encode',build_categorical_encoding_transformer(overrides={'drop_first':True}))]);pre=build_feature_preprocessor([feature,category]);result=run_regression_workflow(X,self.y,preprocessor=pre)
        self.assertTrue(any('category' in name for name in result.model.output_features_));self.assertEqual(len(result.comparison),2)
        y=self.y.copy();y.iloc[1]=np.nan
        with self.assertRaises(ValueError):run_regression_workflow(self.X,y)
        dropped=run_regression_workflow(self.X,y,overrides={'target_missing':'drop','candidates':[{'name':'baseline'}]});self.assertEqual(dropped.excluded_target_positions.tolist(),[1]);self.assertNotIn(1,dropped.train_indices);self.assertNotIn(1,dropped.test_indices)
        with self.assertRaises(ValueError):run_regression_workflow(self.X,self.y,overrides={'cov_type':'HAC'})

if __name__=='__main__':unittest.main()
