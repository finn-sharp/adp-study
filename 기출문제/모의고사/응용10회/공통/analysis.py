"""응용 모의10회 문항 조합: 계산은 src 모듈, 문항 상태·출력·그림은 시험 계층."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,GammaRegressor,PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score,KFold,TimeSeriesSplit
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,roc_auc_score,average_precision_score,confusion_matrix,brier_score_loss
from matplotlib import pyplot as plt
from IPython.display import display
REPO=next(p for p in Path(__file__).parents if (p/'src/Recipes/loader.py').exists());sys.path.insert(0,str(REPO))
from src.output import _json_value
from src.MachineLearning.LinearModels._regression_estimator import RegressionEstimator
from src.MachineLearning.LinearModels.weighted_regression import build_weighted_regressor
from src.MachineLearning.LinearModels.weighted_regression import run_weighted_regression
from src.MachineLearning.NonlinearModels.nonlinear_regression import build_nonlinear_regressor
from src.MachineLearning.GeneralizedLinearModels.gamma import run_gamma
from src.MachineLearning.GeneralizedLinearModels.poisson import run_poisson
from src.MachineLearning.GeneralizedLinearModels.logistic import run_logistic
from src.MachineLearning.Optimization.grid_search import run_grid_search
from src.MachineLearning.Optimization.optuna_search import run_optuna_search
from src.Preprocessing.resampling._common import load_resampler
from src.Preprocessing.regression.vif_selection import build_vif_selector
from src.stats.regression.information_selection import run_information_selection
from src.stats.regression.normality import run_normality
from src.stats.regression.linearity import run_linearity
from src.stats.regression.heteroscedasticity import run_heteroscedasticity
from src.stats.regression.independence import run_independence
from src.stats.regression.collinearity import run_collinearity
from src.stats.regression.influence import run_influence
from src.TimeSeries.rolling_selection import run_rolling_selection
from src.Visualization.time_series.timeline import plot_timeline
from src.stats.tests.TimeSeries.adf import run_adf
from src.Visualization.qq.normal import plot_normal
from src.Visualization.regression.glm_residuals import plot_glm_residuals
from src.Visualization.regression.residual_fitted import plot_residual_fitted
from src.Visualization.regression.partial_residual import plot_partial_residual
from src.Visualization.regression.vif import plot_vif
from src.Visualization.regression.cooks_distance import plot_cooks_distance
from src.Visualization.regression.observed_predicted import plot_observed_predicted
from src.Visualization.regression.coefficient_intervals import plot_coefficient_intervals
from src.Visualization.regression.model_comparison import plot_model_comparison
from src.Visualization.classification.roc import plot_roc
from src.Visualization.classification.precision_recall import plot_precision_recall
from src.Visualization.classification.calibration import plot_calibration


def metrics(y,pred):
    return dict(r2=float(r2_score(y,pred)),rmse=float(np.sqrt(mean_squared_error(y,pred))),mae=float(mean_absolute_error(y,pred)))


class Context:
    """한 회차의 원행·학습/보류·문항 결과를 유지합니다. src는 시험 코드를 import하지 않습니다."""
    def __init__(self,folder):
        self.folder=Path(folder);self.plan=json.loads((self.folder/'실행설정.json').read_text());self.seed=self.plan['seed'];self.number=self.plan['number'];self.results=[];self.out=self.folder/'실행결과';self.out.mkdir(exist_ok=True)
        self.reg=pd.read_csv(self.folder/'data/regression.csv');self.cls=pd.read_csv(self.folder/'data/classification.csv');self.ts=pd.read_csv(self.folder/'data/timeseries.csv',parse_dates=['date']).set_index('date').target.asfreq('D')
    def run(self,qid,task):
        value=getattr(self,task)();value=_json_value(value);record=dict(question=qid,task=task,result=value);self.results.append(record);(self.out/'답안수치.json').write_text(json.dumps(self.results,ensure_ascii=False,indent=2,allow_nan=False)+'\n');display(value);return value
    def reg_design(self):
        self.X=self.reg.drop(columns=['id','target']);self.y=self.reg.target;ids=np.arange(len(self.y))
        if self.plan['ordered_regression']:
            cut=int(len(ids)*.7);self.rt,self.rv=ids[:cut],ids[cut:];self.cv=TimeSeriesSplit(3)
        else:self.rt,self.rv=train_test_split(ids,test_size=.3,random_state=self.seed);self.rt=np.sort(self.rt);self.rv=np.sort(self.rv);self.cv=KFold(3,shuffle=True,random_state=self.seed)
        return dict(n=len(self.reg),feature_names=self.X.columns.tolist(),missing=self.X.isna().sum().to_dict(),train_indices=self.rt,test_indices=self.rv,split='chronological' if self.plan['ordered_regression'] else 'iid',baseline='train-only median+scale+OLS; final test sealed')
    def reg_baseline(self):
        self.base=RegressionEstimator().fit(self.X.iloc[self.rt],self.y.iloc[self.rt]);self.base_cv=float(cross_val_score(RegressionEstimator(),self.X.iloc[self.rt],self.y.iloc[self.rt],cv=self.cv,scoring='r2').mean());inf=self.base.inference_
        return dict(train=metrics(self.y.iloc[self.rt],self.base.predict(self.X.iloc[self.rt])),cv_r2=self.base_cv,coefficients=inf.coefficients,fit_metrics=inf.fit_metrics,final_test='not evaluated yet')
    def reg_diagnostics(self):
        m=self.base.inference_;self.base_influence=run_influence(m);self.base_diagnostics=dict(normality=run_normality(m),linearity=run_linearity(m),heteroscedasticity=run_heteroscedasticity(m),independence=run_independence(m,overrides={'ordered':self.plan['ordered_regression'],'nlags':3}),collinearity=run_collinearity(m.design.drop(columns='Intercept')))
        fig,axes=plt.subplots(3,3,figsize=(16,12));plot_residual_fitted(m,ax=axes[0,0]);plot_normal(m.residuals,ax=axes[0,1],overrides={'fit_parameters':True});plot_cooks_distance(m,ax=axes[1,0]);plot_coefficient_intervals(m,ax=axes[1,1]);plot_vif(self.base_diagnostics['collinearity'],ax=axes[0,2])
        for feature,ax in zip(m.design.columns[1:],[axes[1,2],axes[2,0],axes[2,1],axes[2,2]]):plot_partial_residual(m,feature,ax=ax)
        plt.tight_layout();plt.show()
        return dict(tests=self.base_diagnostics,influence=self.base_influence.table,review_original_positions=self.rt[self.base_influence.table.cook_flag.to_numpy()] if 'cook_flag' in self.base_influence.table else [],interpretation='tests are conditional on fitted training model; nonrejection does not prove assumptions; Cook flags are review only')
    def reg_selection(self):
        design=self.base.transform_features(self.X.iloc[self.rt]);self.ic_results={criterion:run_information_selection(design,self.y.iloc[self.rt].to_numpy(),overrides={'criterion':criterion}) for criterion in ['aic','bic']}
        return dict(aic=self.ic_results['aic'],bic=self.ic_results['bic'],same_rows=self.rt,same_response='original y; never compare with log(y) AIC',note='variable selection is refitted inside each CV fold below; full-train table is descriptive')
    def _reg_candidates(self):
        full=self.X.columns.tolist();candidates=[('baseline',RegressionEstimator(),full,{'degree':[1]}),('vif',RegressionEstimator(vif_threshold=5.),full,{'degree':[1]}),('aic',RegressionEstimator(vif_threshold=5.,information_criterion='aic'),full,{'degree':[1]}),('bic',RegressionEstimator(vif_threshold=5.,information_criterion='bic'),full,{'degree':[1]})]
        kind=self.plan['regression_kind']
        if kind=='power':candidates.append(('feature_power',RegressionEstimator(vif_threshold=5.,power='yeo-johnson'),full,{'degree':[1]}))
        elif kind=='lognormal':candidates.append(('log_target',RegressionEstimator(target_method='log',smearing=True,vif_threshold=5.),full,{'degree':[1]}))
        elif kind=='quadratic':candidates.append(('polynomial',RegressionEstimator(vif_threshold=10.),full,{'degree':[1,2]}))
        elif kind=='heteroscedastic':candidates.append(('fgls',build_weighted_regressor(),full,{'iterations':[1,2]}))
        elif kind=='influence':candidates.append(('cook_drop',RegressionEstimator(vif_threshold=5.,influence_action='drop',max_remove_fraction=.2),full,{'degree':[1]}))
        elif kind=='ordered':candidates.append(('hac_inference',RegressionEstimator(vif_threshold=5.,cov_type='HAC',hac_maxlags=3),full,{'degree':[1]}))
        elif kind in ('exponential','growth'):
            settings=dict(model_name='exponential',initial_parameters=[2.,.5,1.],lower_bounds=[.1,.01,-5.],upper_bounds=[8.,2.,8.]) if kind=='exponential' else dict(model_name='logistic',initial_parameters=[10.,1.,1.5],lower_bounds=[1.,.01,-2.],upper_bounds=[20.,6.,6.])
            candidates.append(('nonlinear',Pipeline([('impute',SimpleImputer(strategy='median')),('model',build_nonlinear_regressor(overrides=settings))]),full[:1],{'model__maxfev':[10000]}))
        elif kind in ('gamma','poisson'):
            cls=GammaRegressor if kind=='gamma' else PoissonRegressor
            candidates.append((kind,Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',cls(max_iter=500))]),full,{'model__alpha':[0.,.1]}))
        return candidates
    def reg_tuning(self):
        self.reg_search={};self.reg_columns={};rows=[]
        for name,model,columns,space in self._reg_candidates():
            result=run_grid_search(self.X.iloc[self.rt][columns],self.y.iloc[self.rt],model,task='regression',overrides={'search_space':space,'scoring':'r2','split':'time' if self.plan['ordered_regression'] else 'iid','shuffle':not self.plan['ordered_regression'],'cv':3,'random_state':self.seed,'n_jobs':1})
            self.reg_search[name]=result;self.reg_columns[name]=columns;rows.append(dict(name=name,cv_r2=float(result.best_score),best_params=result.best_params))
        self.reg_comparison=pd.DataFrame(rows);self.reg_name=max(self.reg_search,key=lambda n:self.reg_search[n].best_score);fig,ax=plt.subplots(figsize=(7,4));plot_model_comparison(self.reg_comparison,ax=ax);plt.tight_layout();plt.show()
        return dict(comparison=self.reg_comparison,selected_name=self.reg_name,folds=self.reg_search[self.reg_name].folds,scope='train CV only; raw y R2 shared across all candidates; no final test used')
    def reg_final(self):
        result=self.reg_search[self.reg_name];model=result.model;cols=self.reg_columns[self.reg_name];prediction=model.predict(self.X.iloc[self.rv][cols]);baseline=self.base.predict(self.X.iloc[self.rv]);details={};kind=self.plan['regression_kind']
        if isinstance(model,RegressionEstimator):
            details.update(fitted_response_scale=model.target_method,model_inference=model.inference_,removed_original_positions=self.rt[model.removed_positions_],selected_features=model.output_features_)
            details['after_tests']=dict(normality=run_normality(model.inference_),linearity=run_linearity(model.inference_),heteroscedasticity=run_heteroscedasticity(model.inference_),independence=run_independence(model.inference_,overrides={'ordered':self.plan['ordered_regression'],'nlags':3}))
        if kind=='heteroscedastic':
            details['fgls']=run_weighted_regression(self.X.iloc[self.rt],self.y.iloc[self.rt],self.X.iloc[self.rv]);wres=details['fgls'].weighted_residuals;stat,p=stats.shapiro(wres);details['weighted_residual_check']=dict(statistic=float(stat),p_value=float(p),interpretation='descriptive conditional check after variance estimation; not an exact independent normality test')
        if kind in ('exponential','growth'):
            native=self.reg_search['nonlinear'].model.named_steps['model'].result_;details['nonlinear_candidate']=native;details['nonlinear_covariance_condition_number']=float(np.linalg.cond(native.covariance))
        if kind in ('gamma','poisson'):
            preprocessing=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]);train=pd.DataFrame(preprocessing.fit_transform(self.X.iloc[self.rt]),columns=cols);test=pd.DataFrame(preprocessing.transform(self.X.iloc[self.rv]),columns=cols);details['family_inference']=(run_gamma if kind=='gamma' else run_poisson)(train,self.y.iloc[self.rt].to_numpy(),test)
        selected=metrics(self.y.iloc[self.rv],prediction);base=metrics(self.y.iloc[self.rv],baseline)
        if kind in ('gamma','poisson','heteroscedastic'):
            fig,axes=plt.subplots(1,2,figsize=(11,4));plot_observed_predicted(self.y.iloc[self.rv],prediction,ax=axes[0])
            if kind=='heteroscedastic':plot_normal(details['fgls'].weighted_residuals,ax=axes[1],overrides={'fit_parameters':True,'title':'Estimated-weight residuals (conditional)'})
            else:plot_glm_residuals(details['family_inference'],ax=axes[1])
        else:
            fig,ax=plt.subplots(figsize=(6,4));plot_observed_predicted(self.y.iloc[self.rv],prediction,ax=ax)
        plt.tight_layout();plt.show()
        return dict(selected_name=self.reg_name,baseline_test=base,selected_test=selected,delta_r2=selected['r2']-base['r2'],test_indices=self.rv,truth=self.y.iloc[self.rv].to_numpy(),predictions=prediction,details=details,conclusion='CV selected before test. R2 improvement is empirical, not guaranteed; covariance correction alone does not change predictions; nonlinear/GLM do not inherit all OLS assumptions.')
    def cls_design(self):
        self.CX=self.cls.drop(columns=['id','target']);self.Cy=self.cls.target.to_numpy();self.ct,self.ce=train_test_split(np.arange(len(self.Cy)),test_size=.3,stratify=self.Cy,random_state=self.seed);self.ct=np.sort(self.ct);self.ce=np.sort(self.ce)
        return dict(train_indices=self.ct,test_indices=self.ce,class_counts_train=pd.Series(self.Cy[self.ct]).value_counts().to_dict(),positive_label=1,event=self.plan['classification_event'],rule='natural prevalence test sealed; no resampling before split')
    def cls_tuning(self):
        from imblearn.pipeline import Pipeline as ImbPipeline
        self.cls_search={};sampler=self.plan['sampler'];method=self.plan['classification_search'];rows=[]
        for name in ['none',sampler]:
            settings={'k_neighbors':3,'random_state':self.seed} if name=='smote' else {'random_state':self.seed}
            sampling='passthrough' if name=='none' else load_resampler(name,overrides=settings)
            pipeline=ImbPipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('vif',build_vif_selector(overrides={'threshold':5.})),('sample',sampling),('model',LogisticRegression(max_iter=1000,random_state=self.seed))])
            space={'model__C':[.1,1.,10.]}
            if method=='optuna':
                import optuna;optuna.logging.set_verbosity(optuna.logging.WARNING)
                search=run_optuna_search(self.CX.iloc[self.ct],self.Cy[self.ct],pipeline,task='classification',overrides={'search_space':{'model__C':{'type':'categorical','choices':[.1,1.,10.]}},'scoring':'roc_auc','n_iter':6,'cv':3,'random_state':self.seed})
            else:search=run_grid_search(self.CX.iloc[self.ct],self.Cy[self.ct],pipeline,task='classification',overrides={'search_space':space,'scoring':'roc_auc','cv':3,'random_state':self.seed})
            self.cls_search[name]=search;rows.append(dict(name=name,cv_auc=search.best_score,best_params=search.best_params))
        self.cls_name=max(self.cls_search,key=lambda n:self.cls_search[n].best_score)
        return dict(method=method,comparison=rows,selected=self.cls_name,folds=self.cls_search[self.cls_name].folds,sampling='only in imblearn Pipeline.fit of CV training folds; validation/test never resampled')
    def cls_logit(self):
        # Explanatory unpenalized GLM is distinct from tuned penalized predictive classifier.
        pipeline=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('vif',build_vif_selector(overrides={'threshold':5.}))]);a=pipeline.fit_transform(self.CX.iloc[self.ct],self.Cy[self.ct]);names=pipeline['vif'].get_feature_names_out().tolist();frame=pd.DataFrame(a,columns=names);self.logit=run_logistic(frame,self.Cy[self.ct]);extra=frame.copy();col=names[0];extra[col+'_squared']=extra[col]**2;expanded=run_logistic(extra,self.Cy[self.ct]);lr=2*(expanded.model.llf-self.logit.model.llf);p=float(stats.chi2.sf(max(lr,0),1))
        return dict(explanatory_model=self.logit,logit_specification=dict(H0='first feature quadratic log-odds term = 0',H1='quadratic term is not zero',lr_statistic=float(lr),df=1,p_value=p,reject_H0=p<.05),note='unpenalized natural prevalence explanation is separate from sampled/tuned L2 predictive model. No OLS residual normality requirement; a single quadratic test does not prove global logit linearity.')
    def cls_final(self):
        self.prob=self.cls_search[self.cls_name].model.predict_proba(self.CX.iloc[self.ce])[:,1];truth=self.Cy[self.ce];self.cls_auc=float(roc_auc_score(truth,self.prob));self.cls_ap=float(average_precision_score(truth,self.prob))
        fig,axes=plt.subplots(1,3,figsize=(14,4));plot_roc(truth,self.prob,ax=axes[0],overrides={'positive_label':1});plot_precision_recall(truth,self.prob,ax=axes[1],overrides={'positive_label':1});plot_calibration(truth,self.prob,ax=axes[2],overrides={'positive_label':1});plt.tight_layout();plt.show()
        return dict(selected=self.cls_name,test_indices=self.ce,truth=truth,probabilities=self.prob,roc_auc=self.cls_auc,average_precision=self.cls_ap,brier=float(brier_score_loss(truth,self.prob)),confusion_at_fixed_half=confusion_matrix(truth,self.prob>=.5),prevalence=float(truth.mean()),threshold=.5)
    def cls_decision(self):
        truth=self.Cy[self.ce];pred=self.prob>=.5;tn,fp,fn,tp=confusion_matrix(truth,pred).ravel();cost=int(self.plan['false_negative_cost']*fn+fp)
        return dict(false_negative_cost=self.plan['false_negative_cost'],false_positive_cost=1,holdout_cost_at_prespecified_half=cost,decision='ROC AUC measures ranking, AP reflects imbalance and Brier reflects probability quality. Resampling alters prevalence: calibrated probabilities or an operating threshold need separate training validation; do not optimize a threshold on final test.',logistic_not_continuous_regression='0/1 target uses logit mean; no ordinary R2/t/F/normal residual assumptions')
    def ts_design(self):
        horizon=self.plan['horizon'];self.ts_train=self.ts.iloc[:-horizon];self.ts_test=self.ts.iloc[-horizon:];return dict(n=len(self.ts),frequency='daily',period=self.plan['period'],train_end=str(self.ts_train.index[-1]),test_start=str(self.ts_test.index[0]),horizon=horizon,training_adf=run_adf(self.ts_train.to_numpy()),rule='chronological only; transforms and lag selection must use past observations')
    def ts_tuning(self):
        self.forecast=run_rolling_selection(self.ts,overrides={'horizon':self.plan['horizon'],'validation_size':7,'folds':2,'period':self.plan['period'],'candidate_orders':[[0,0,0],[1,0,0]],'maxiter':100});return dict(comparison=self.forecast.comparison,folds=self.forecast.folds,selected_order=self.forecast.selected_order,information=self.forecast.residual_diagnostics)
    def ts_diagnostics(self):
        return dict(diagnostics=self.forecast.residual_diagnostics,train_information=dict(aic=self.forecast.model.aic,bic=self.forecast.model.bic),meaning='Ljung-Box subtracts p+q+P+Q degrees of freedom and removes initialization. Nonrejection does not prove independence. Compare differencing/families via rolling error; AIC/BIC are not predictive R2.')
    def ts_final(self):
        f=self.forecast;fig,ax=plt.subplots(figsize=(9,4));plot_timeline(self.ts.index[-35:],self.ts.iloc[-35:],ax=ax,overrides={'rolling_window':0});ax.plot(self.ts_test.index,f.predictions,label='Selected forecast',color='C1');ax.plot(self.ts_test.index,f.baseline_predictions,label='Naive forecast',color='C2');ax.fill_between(self.ts_test.index,f.model.prediction_lower,f.model.prediction_upper,alpha=.2,label='Conditional interval');ax.legend();ax.set_title('Future holdout after rolling selection');plt.tight_layout();plt.show()
        return dict(test_metrics=f.test_metrics,test_indices=f.test_indices,truth=self.ts_test.to_numpy(),predictions=f.predictions,baseline_predictions=f.baseline_predictions,interval_lower=f.model.prediction_lower,interval_upper=f.model.prediction_upper,interval='conditional on fitted selected model; selection uncertainty not included')
