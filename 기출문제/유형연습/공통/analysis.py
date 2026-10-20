"""후기 유형 연습의 실행 계산. 데이터·Profile은 문제별 고정 연습 조건이다.

일반 분석 라이브러리의 새 공개 API가 아닌 풀이 전용 코드다.
각 문제는 독립 Context에 자료/중간결과/실제 수치 답안을 보존한다.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from itertools import combinations
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize, linalg
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from src.Preprocessing.resampling._common import load_resampler
from src.Preprocessing.resampling.random_over import run_random_over
from src.TimeSeries.arima import run_arima
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.multitest import multipletests


def clean(value):
 if value is pd.NA or value is pd.NaT:return None
 if isinstance(value,(pd.Timestamp,pd.Timedelta)):return str(value)
 if isinstance(value,pd.DataFrame):return clean(value.reset_index().to_dict(orient='records'))
 if isinstance(value,pd.Series):return clean(value.to_dict())
 if isinstance(value,np.ndarray):return clean(value.tolist())
 if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [clean(x) for x in value]
 if isinstance(value,(np.integer,)):return int(value)
 if isinstance(value,(np.bool_,)):return bool(value)
 if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
 return value


class Context:
 def __init__(self,folder):
  self.folder=Path(folder);self.state={};self.answers=[];self.current=None
  self.output=self.folder/'실행결과';self.output.mkdir(exist_ok=True)
 def csv(self,name):return pd.read_csv(self.folder/'data'/(name+'.csv'))
 def js(self,name):return json.loads((self.folder/'data'/(name+'.json')).read_text())
 def emit(self,label,value):
  value=clean(value);self.current['outputs'][label]=value
  print(label);print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
 def test(self,label,H0,H1,stat,p,df=None,padj=None,context='차이'):
  used=float(p if padj is None else padj);reject=used<.05
  r=dict(method=label,H0=H0,H1=H1,statistic=float(stat),p_value=float(p),df=clean(df),alpha=.05,reject_H0=reject,
         conclusion='유의수준 5%에서 '+context+('에 대한 근거가 있다.' if reject else '에 대한 근거가 충분하지 않다.'))
  if padj is not None:r['adjusted_p_value']=float(padj)
  self.current['tests'].append(r);print(json.dumps(r,ensure_ascii=False,indent=2))
 def fig(self,name):
  plt.tight_layout();plt.savefig(self.output/(name+'.png'),dpi=90,bbox_inches='tight');plt.show();plt.close('all')
 def run(self,qid,task):
  self.current=dict(question=qid,task=task,outputs={},tests=[]);print('\n=== '+qid+' ===')
  TASKS[task](self);self.answers.append(self.current)
  (self.output/'답안수치.json').write_text(json.dumps(clean(self.answers),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def regpipe(model=None):
 return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LinearRegression() if model is None else model)])


def classpipe(model=None,cat=None,num=None,sampling=None):
 model=LogisticRegression(max_iter=1500,random_state=7) if model is None else model
 if cat:
  pre=ColumnTransformer([('num',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
                         ('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))]),cat)])
 else:pre=None
 steps=[('pre',pre)] if pre is not None else [('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]
 if sampling:steps.append(('sample',load_resampler('random_over' if sampling=='over' else 'random_under',overrides={'random_state':7})))
 return ImbPipeline(steps+[('model',model)])


def model_compare(c,X,y,train,test,kind='regression',cv=None,sampling=None,cat=None,num=None,three=False):
 if kind=='regression':
  models={'Linear':regpipe(),'RF':regpipe(RandomForestRegressor(n_estimators=50,max_depth=5,random_state=7,n_jobs=1)),
          'GB':regpipe(GradientBoostingRegressor(n_estimators=50,max_depth=2,random_state=7))}
  cv=KFold(3,shuffle=True,random_state=7) if cv is None else cv;scoring='neg_mean_squared_error'
 else:
  models={'Logistic':classpipe(cat=cat,num=num,sampling=sampling),
          'RF':classpipe(RandomForestClassifier(n_estimators=60,max_depth=5,random_state=7,n_jobs=1),cat=cat,num=num,sampling=sampling)}
  if three:models['GB']=classpipe(GradientBoostingClassifier(n_estimators=40,max_depth=2,random_state=7),sampling=sampling)
  cv=StratifiedKFold(3,shuffle=True,random_state=7) if cv is None else cv;scoring='f1_macro' if y.nunique()>2 else 'f1'
 scores={}
 for name,model in models.items():
  raw=cross_val_score(model,X.loc[train],y.loc[train],cv=cv,scoring=scoring)
  scores[name]=float(np.mean(np.sqrt(-raw))) if kind=='regression' else float(raw.mean())
 chosen=min(scores,key=scores.get) if kind=='regression' else max(scores,key=scores.get)
 model=models[chosen].fit(X.loc[train],y.loc[train]);pred=model.predict(X.loc[test]);truth=y.loc[test]
 c.emit('CV selection',dict(scores=scores,chosen=chosen,metric='RMSE' if kind=='regression' else scoring,selection='training CV only'))
 if kind=='regression':
  c.emit('holdout',dict(RMSE=np.sqrt(mean_squared_error(truth,pred)),MAE=mean_absolute_error(truth,pred),R2=r2_score(truth,pred),n=len(truth)))
 else:
  out=dict(F1=f1_score(truth,pred,average='macro' if y.nunique()>2 else 'binary'),confusion_matrix=confusion_matrix(truth,pred),
           classification_report=classification_report(truth,pred,output_dict=True,zero_division=0),n=len(truth))
  if y.nunique()==2:
   prob=model.predict_proba(X.loc[test])[:,1];out['AUC']=roc_auc_score(truth,prob)
   fpr,tpr,_=roc_curve(truth,prob);plt.figure(figsize=(4,3));plt.plot(fpr,tpr,label=chosen);plt.plot([0,1],[0,1],'--',color='grey');plt.legend();plt.xlabel('False positive rate');plt.ylabel('True positive rate');c.fig('roc_'+c.current['question'])
  c.emit('holdout',out)
  fitted=model.named_steps['model']
  if hasattr(fitted,'feature_importances_'):c.emit('RF importance (transformed columns)',fitted.feature_importances_)
  if hasattr(fitted,'coef_'):c.emit('Logistic odds ratios (scaled numeric units)',np.exp(fitted.coef_))
 c.emit('split IDs',dict(train_index=list(X.loc[train].index),test_index=list(X.loc[test].index)))
 return model


def regression(c,d,y,features,diagnostics=True):
 X=sm.add_constant(d[features].astype(float));m=sm.OLS(d[y],X).fit()
 c.emit('coefficients',pd.DataFrame({'coef':m.params,'SE':m.bse,'t':m.tvalues,'p':m.pvalues}))
 c.emit('fit',dict(R2=m.rsquared,n=m.nobs))
 c.test('OLS overall F','all slopes=0','at least one slope != 0',m.fvalue,m.f_pvalue,[m.df_model,m.df_resid],context='전체 설명변수와 평균 반응의 선형 연관성')
 c.emit('VIF',dict(zip(X.columns,[variance_inflation_factor(X.values,i) for i in range(X.shape[1])])) )
 if diagnostics:
  jb=jarque_bera(m.resid);bp=het_breuschpagan(m.resid,X);reset=linear_reset(m,power=2,use_f=True)
  c.test('Jarque-Bera','normal residual distribution','not normal',jb[0],jb[1],2,context='오차 정규성 위반')
  c.test('Breusch-Pagan LM','constant error variance','variance associated with X',bp[0],bp[1],X.shape[1]-1,context='오차 등분산성 위반')
  c.test('RESET F','no added squared fitted term','added fitted-square coefficient !=0',float(reset.fvalue),float(reset.pvalue),context='선택 함수형태의 부족')
  fig,axes=plt.subplots(1,2,figsize=(7,3));axes[0].scatter(m.fittedvalues,m.resid,s=9);axes[0].axhline(0,color='grey');axes[0].set(xlabel='Fitted',ylabel='Residual');sm.qqplot(m.resid,line='s',ax=axes[1]);c.fig('residual_'+c.current['question'])
 return m


def categorical(c,d,a,b,adjust=None):
 table=pd.crosstab(d[a],d[b]);stat,p,df,expected=stats.chi2_contingency(table,correction=False)
 c.emit(a+'_table',table);c.emit(a+'_expected',expected)
 c.test('Pearson independence '+a+'-'+b,a+' independent of '+b,a+' associated with '+b,stat,p,df,padj=adjust,context=a+'와 '+b+'의 관련성')
 return p


def health_eda(c):
 d=c.csv('health');tr,te=train_test_split(d.index,test_size=.3,random_state=30);c.state.update(health=d,train=tr,test=te)
 c.emit('missing',d.isna().sum());c.emit('distributions',dict(train=d.loc[tr,'DBP'].describe(),test=d.loc[te,'DBP'].describe()))
 d[['age','weight','waist','pulse','DBP']].hist(figsize=(7,4));c.fig('health_distribution')

def health_pca(c):
 d=c.state['health'];cols=['age','weight','waist','pulse'];tr=c.state['train'];p=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('pca',PCA(n_components=.9))]).fit(d.loc[tr,cols])
 c.emit('training correlation',d.loc[tr,cols].corr());c.emit('PCA',dict(n=p.named_steps['pca'].n_components_,variance_ratio=p.named_steps['pca'].explained_variance_ratio_))

def health_ols(c):
 d=c.state['health'];tr=c.state['train'];cols=['age','weight','waist','pulse'];x=d.loc[tr].copy();x[cols]=SimpleImputer(strategy='median').fit_transform(x[cols]);regression(c,x,'DBP',cols)

def health_models(c):
 d=c.state['health'];model_compare(c,d[['age','weight','waist','pulse']],d.DBP,c.state['train'],c.state['test'])

def accident_tests(c):
 d=c.csv('accident');d['weekend']=(pd.to_datetime(d.date).dt.dayofweek>=5).astype(int);c.state['accident']=d
 tr=d.id%5!=0;p=[stats.chi2_contingency(pd.crosstab(d.loc[tr,x],d.loc[tr,'severe']),correction=False)[1] for x in ['weather','road']]
 for x,adj in zip(['weather','road'],multipletests(p,method='holm')[1]):categorical(c,d.loc[tr],x,'severe',adj)

def accident_sampling(c):
 d=c.state['accident'];d=d[d.id%5!=0];sampled=run_random_over(d[['weather','road','speed','weekend']],d.severe,overrides={'random_state':7});X,y=sampled.X_resampled,sampled.y_resampled
 c.emit('class counts',dict(before=d.severe.value_counts(),after=y.value_counts()));assert set(X.weather)<=set(d.weather)

def accident_models(c):
 d=c.state['accident'];model_compare(c,d[['weather','road','speed','weekend']],d.severe,d.id%5!=0,d.id%5==0,'classification',sampling='over',cat=['weather','road'],num=['speed','weekend'])

def transport(c):
 d=c.js('transport');cost=np.asarray(d['cost']);s=np.array(d['supply']);demand=np.array(d['demand']);m,n=cost.shape
 A=np.r_[np.kron(np.eye(m),np.ones((1,n))),np.tile(np.eye(n),(1,m))];b=np.r_[s,demand]
 res=optimize.linprog(cost.ravel(),A_eq=A,b_eq=b,bounds=(0,None),method='highs');assert res.success
 x=res.x.reshape(m,n);dual=float(b@res.eqlin.marginals)
 assert np.allclose(x.sum(1),s) and np.allclose(x.sum(0),demand)
 c.emit('plan',dict(amount=x,cost=res.fun,dual_objective=dual,primal_dual_gap=res.fun-dual,dual_variables=res.eqlin.marginals,reduced_cost=res.lower.marginals,success=res.success,supply=s,demand=demand))

def headset(c):categorical(c,c.csv('headset'),'age_group','preference')

def binomial(c):
 q=stats.binom.sf(3,6,.5);c.emit('probability',dict(q=q,P_at_least_3_families=stats.binom.sf(2,5,q),expected_families=5*q))

def obesity_prepare(c):
 d=c.csv('obesity');d['BMI']=d.weight/(d.height_cm/100)**2;tr,te=train_test_split(d.index,test_size=.3,random_state=31,stratify=d.body_class)
 c.state.update(obesity=d,ob_tr=tr,ob_te=te);c.emit('BMI and missing',dict(summary=d.describe(),missing=d.isna().sum()))

def obesity_model(c):
 d=c.state['obesity'];cols=['BMI','activity','age'];X=d[cols];y=d.body_class;tr=c.state['ob_tr'];te=c.state['ob_te']
 search=GridSearchCV(classpipe(),{'model__C':[.1,1.,10.]},scoring='f1_macro',cv=StratifiedKFold(3,shuffle=True,random_state=7)).fit(X.loc[tr],y.loc[tr])
 pred=search.predict(X.loc[te]);c.emit('tuning',dict(best=search.best_params_,CV=search.best_score_,holdout_macro_F1=f1_score(y.loc[te],pred,average='macro'),confusion=confusion_matrix(y.loc[te],pred)))

def checkup_features(c):
 d=c.csv('checkup');d['age']=((pd.to_datetime(d.exam_date)-pd.to_datetime(d.birth_date)).dt.days//365).astype(int);assert set(d.age)<={16,17}
 d['BMI']=d.weight/(d.height_cm/100)**2;rules=c.js('기준표');bounds=np.array([rules[str(a)+'_'+g] for a,g in zip(d.age,d.gender)])
 d['suitable']=((d.BMI>bounds[:,0])&(d.BMI<=bounds[:,1])).astype(int);c.state['checkup']=d
 c.emit('engineered',d[['id','age','gender','BMI','suitable']].head(10));c.emit('label counts',d.suitable.value_counts())
 d.to_csv(c.output/'checkup_engineered.csv',index=False)

def checkup_select(c):
 d=c.state['checkup'];tr=d.id%5!=0;raw=[]
 for col in ['exercise','sleep']:raw.append(stats.chi2_contingency(pd.crosstab(d.loc[tr,col],d.loc[tr,'suitable']),correction=False)[1])
 adj=multipletests(raw,method='holm')[1];selected=[]
 for col,p in zip(['exercise','sleep'],adj):
  categorical(c,d.loc[tr],col,'suitable',p)
  if p<.05:selected.append(col)
 c.state['selected_checkup']=selected;c.emit('selected train-only',selected)

def checkup_models(c):
 d=c.state['checkup'];selected=c.state['selected_checkup'];tr=d.id%5!=0;te=~tr
 if not selected:
  prior=d.loc[tr,'suitable'].mean();c.emit('fallback',dict(prior=prior,AUC=.5,reason='no selected variables'));return
 model_compare(c,d[selected],d.suitable,tr,te,'classification')
 # 원래 0/1 단위의 오즈비를 위해 별도 무규제 설명모형을 적합한다(선택 후 탐색).
 logit=sm.Logit(d.loc[tr,'suitable'],sm.add_constant(d.loc[tr,selected])).fit(disp=False)
 c.emit('unscaled exploratory odds ratios',dict(OR=np.exp(logit.params),CI=np.exp(logit.conf_int()),p=logit.pvalues))

def bayes(c):
 variance=1/(1/225+1/144);mean=variance*(90/225+108/144);c.emit('posterior',dict(mean=mean,variance=variance,credible95=stats.norm.ppf([.025,.975],loc=mean,scale=np.sqrt(variance))))

def ads_full(c):regression(c,c.csv('ads'),'sales',['TV','radio','paper'])

def ads_backward(c):
 d=c.csv('ads');cols=['TV','radio','paper'];history=[]
 while cols:
  m=sm.OLS(d.sales,sm.add_constant(d[cols])).fit();p=m.pvalues.drop('const');worst=p.idxmax()
  if p[worst]<=.05:break
  history.append(dict(remove=worst,p=float(p[worst])));cols.remove(worst)
 c.emit('backward history',history);regression(c,d,'sales',cols,diagnostics=False)

def poisson_probability(c):c.emit('Poisson(2.4)',dict(p0=stats.poisson.pmf(0,2.4),p_ge2=stats.poisson.sf(1,2.4)))

def housing_diagnostics(c):c.state['housing_model']=regression(c,c.csv('housing'),'target',['x1','x2','x3','x4','x5'])

def housing_pca(c):
 d=c.csv('housing');X=d[['x1','x2','x3','x4','x5']];z=StandardScaler().fit_transform(X);pc=PCA().fit_transform(z)
 orig=sm.OLS(d.target,sm.add_constant(X)).fit();new=sm.OLS(d.target,sm.add_constant(pc)).fit();err=float(np.max(np.abs(orig.fittedvalues-new.fittedvalues)))
 assert err<1e-8;c.emit('all-component invariance',dict(max_abs_fitted_difference=err,variance_ratio=PCA().fit(z).explained_variance_ratio_))

def housing_robust(c):
 m=c.state['housing_model'];rob=m.get_robustcov_results(cov_type='HC3');c.emit('SE comparison',pd.DataFrame({'OLS_SE':m.bse,'HC3_SE':rob.bse,'HC3_p':rob.pvalues}))

def category_expected(c):
 table=pd.crosstab(c.csv('category').cyl,c.csv('category').gear);expected=stats.contingency.expected_freq(table);c.emit('table',table);c.emit('expected',expected);c.emit('small expected cells',int((expected<5).sum()))

def category_mc(c):
 d=c.csv('category');obs=pd.crosstab(d.cyl,d.gear).to_numpy();exp=stats.contingency.expected_freq(obs);stat=float(((obs-exp)**2/exp).sum())
 sims=stats.random_table.rvs(obs.sum(1),obs.sum(0),size=4999,random_state=32);simstat=((sims-exp)**2/exp).sum((1,2));exceed=int((simstat>=stat-1e-12).sum());p=(exceed+1)/5000
 c.emit('Monte Carlo detail',dict(observed=obs,expected=exp,simulations=4999,exceed=exceed,MC_SE=np.sqrt(p*(1-p)/5000),row_sums=obs.sum(1),col_sums=obs.sum(0)))
 c.test('Fixed-margin RxC Pearson Monte Carlo','cyl independent of gear','not independent',stat,p,context='자동차 범주 간 전체 관련성')

def salary(c):
 d=c.csv('salary');a=d[d.group=='A'].salary;b=d[d.group=='B'].salary;r=stats.mannwhitneyu(a,b,alternative='two-sided',method='asymptotic')
 c.emit('group summaries',d.groupby('group').salary.describe());c.test('Mann-Whitney U','same distributions','different distributions',r.statistic,r.pvalue,context='두 급여 분포의 차이')

def hepatitis_prepare(c):
 d=c.csv('hepatitis');cols=['x'+str(i+1) for i in range(6)];tr=d.id%5!=0
 c.emit('missing/deletion loss',dict(missing=d[cols].isna().sum(),deleted_rows=int(d[cols].isna().any(axis=1).sum())))
 p=regpipe(PCA(n_components=.9)).fit(d.loc[tr,cols],d.loc[tr,'Category']);c.emit('PCA training variance',p.named_steps['model'].explained_variance_ratio_);c.state['hep']=d

def hepatitis_binary(c):
 d=c.state['hep'];cols=['x'+str(i+1) for i in range(6)];y=(d.Category>0).astype(int);tr=d.id%5!=0;te=~tr;cv=StratifiedKFold(3,shuffle=True,random_state=7);scores={}
 for sample in [None,'over','under']:
  scores[str(sample)]=float(cross_val_score(classpipe(sampling=sample),d.loc[tr,cols],y.loc[tr],cv=cv,scoring='f1').mean())
 chosen=max(scores,key=scores.get);sample=None if chosen=='None' else chosen;c.emit('sampling CV F1',dict(scores=scores,chosen=chosen))
 model=classpipe(sampling=sample).fit(d.loc[tr,cols],y.loc[tr]);pred=model.predict(d.loc[te,cols]);prob=model.predict_proba(d.loc[te,cols])[:,1]
 c.emit('holdout',dict(precision=precision_score(y.loc[te],pred),recall=recall_score(y.loc[te],pred),F1=f1_score(y.loc[te],pred),AUC=roc_auc_score(y.loc[te],prob)))

def hepatitis_multi(c):
 d=c.state['hep'];d=d[d.Category>0];model_compare(c,d[['x'+str(i+1) for i in range(6)]],d.Category,d.id%5!=0,d.id%5==0,'classification',three=True)

def time_merge(c,power=False):
 d=c.csv('power' if power else 'subway');valuecols=['p00','p12'] if power else ['h05','h06','h07'];target='generation' if power else 'users'
 long=d.melt(id_vars=['date','kind' if power else 'board'],value_vars=valuecols,value_name='value');daily=long.groupby('date',as_index=False).value.sum().rename(columns={'value':target})
 assert np.isclose(daily[target].sum(),d[valuecols].sum().sum())
 weather=c.csv('weather_power' if power else 'weather');assert not weather.date.duplicated().any()
 if not power:weather['rain']=weather.rain.fillna(0)
 joined=daily.merge(weather,on='date',how='left',validate='one_to_one',indicator=True);joined['weather_missing']=(joined['_merge']!='both').astype(int)
 joined['date']=pd.to_datetime(joined.date);joined=joined.sort_values('date').reset_index(drop=True);joined['weekend']=(joined.date.dt.dayofweek>=5).astype(int)
 month=joined.date.dt.month;joined['season']=np.select([month.isin([3,4,5]),month.isin([6,7,8]),month.isin([9,10,11])],[1,2,3],default=4)
 c.state['power' if power else 'subway']=joined;c.emit('merge validation',dict(rows=len(joined),input_days=len(daily),unmatched=int(joined.weather_missing.sum()),description=joined.describe()))
 if power:c.emit('descriptive correlation',joined[['generation','temp','wind']].corr())
 joined.to_csv(c.output/('power_joined.csv' if power else 'subway_joined.csv'),index=False)

def subway_merge(c):time_merge(c)
def power_merge(c):time_merge(c,True)
def time_models(c,key):
 d=c.state[key];cols=['temp','wind','weekend','season'] if key=='power' else ['temp','rain','weekend','weather_missing'];tr=d.date.dt.year<2023;te=~tr
 model_compare(c,d[cols],d['generation' if key=='power' else 'users'],tr,te,cv=TimeSeriesSplit(3))
 c.emit('temporal boundary',dict(train_max=str(d.loc[tr,'date'].max()),test_min=str(d.loc[te,'date'].min())))
def subway_models(c):time_models(c,'subway')
def power_models(c):time_models(c,'power')

def defect(c):
 d=c.csv('defect');a=d[d.group=='A'].rate;b=d[d.group=='B'].rate;r=stats.ttest_ind(a,b,equal_var=False)
 c.emit('summary',dict(A=a.describe(),B=b.describe(),difference=a.mean()-b.mean()));c.test('Welch t','mu_A=mu_B','mu_A != mu_B',r.statistic,r.pvalue,context='평균 불량률 차이')

def poisson_gof(c):
 x=c.csv('counts')['count'].to_numpy();n=len(x);lam=x.mean();maxk=max(int(x.max()),int(stats.poisson.ppf(1-1e-9,lam)))
 raw_exp=n*stats.poisson.pmf(np.arange(maxk+1),lam);raw_exp[-1]+=n*stats.poisson.sf(maxk,lam)
 raw_obs=np.bincount(np.minimum(x,maxk),minlength=maxk+1);groups=[];eo=0.;oo=0;start=0
 for k in range(maxk+1):
  eo+=raw_exp[k];oo+=raw_obs[k]
  if eo>=5:groups.append([start,k,int(oo),float(eo)]);eo=0.;oo=0;start=k+1
 if eo:
  groups[-1][1]=maxk;groups[-1][2]+=int(oo);groups[-1][3]+=float(eo)
 obs=np.array([g[2] for g in groups]);exp=np.array([g[3] for g in groups]);df=len(groups)-2;stat=float(((obs-exp)**2/exp).sum());p=stats.chi2.sf(stat,df)
 assert (exp>=5).all() and np.isclose(exp.sum(),n) and obs.sum()==n
 c.emit('pooled bins',dict(lambda_est=lam,bins=groups,n=n,df=df));c.test('Estimated-Poisson Pearson GOF','counts follow Poisson(lambda)','not Poisson',stat,p,df,context='포아송 분포 부적합')

def delay(c):
 d=c.csv('delay');d['interval']=pd.cut(d.minutes,[-np.inf,10,20,np.inf],labels=['le10','10to20','gt20']);categorical(c,d,'line','interval')

def arma(c):
 y=c.csv('series').value;cut=int(len(y)*.8);train=y.iloc[:cut];test=y.iloc[cut:];adf=adfuller(train,autolag='AIC');c.test('ADF','unit root','stationary',adf[0],adf[1],context='단위근 가설의 기각')
 result=run_arima(train,overrides={'steps':len(test),'trend':'c','candidate_orders':[[1,0,0],[2,0,0],[1,0,1],[2,0,1]]})
 scores=[dict(p=row['order'][0],q=row['order'][2],AIC=row['aic'],converged=row['converged']) for row in result.candidates]
 chosen=min([row for row in scores if row['converged']],key=lambda row:row['AIC']);m=result.model
 lb=acorr_ljungbox(m.resid,lags=[10],model_df=chosen['p']+chosen['q'],return_df=True).iloc[0]
 c.emit('ARMA candidates/forecast',dict(candidates=scores,selected=chosen,holdout_RMSE=np.sqrt(mean_squared_error(test,result.predictions))))
 c.test('Ljung-Box lag10','no residual autocorrelation through lag10','some residual autocorrelation',lb.lb_stat,lb.lb_pvalue,10-chosen['p']-chosen['q'],context='잔차 자기상관')
 fig,axes=plt.subplots(1,2,figsize=(7,3));plot_acf(train,lags=20,ax=axes[0]);plot_pacf(train,lags=20,ax=axes[1],method='ywm');c.fig('acf_pacf')

def clinical_prepare(c):
 d=c.csv('clinical');cols=['s1','s2','s3'];observed=d[cols].notna().all(axis=1);d['bleed']=np.where(observed,(d[cols].sum(axis=1)>=4).astype(int),np.nan)
 assert d.loc[d.ind==1,'bleed'].isna().all() and d.loc[d.ind.isin([0,2]),'bleed'].notna().all()
 c.emit('bleed table with missing',d.groupby('ind').bleed.value_counts(dropna=False));c.state['clinical']=d

def clinical_oof(c):
 d=c.state['clinical'];cols=['age','bp','signal'];known=d.ind==0;unknown=d.ind==1;test=d.ind==2;X=d[cols];y=d.bleed;cv=StratifiedKFold(3,shuffle=True,random_state=7)
 candidates={'Logistic':classpipe(),'RF':classpipe(RandomForestClassifier(n_estimators=60,max_depth=4,random_state=7))};scores={k:cross_val_score(v,X[known],y[known],cv=cv,scoring='f1').mean() for k,v in candidates.items()};chosen=max(scores,key=scores.get);model=candidates[chosen]
 prob=np.full(len(d),np.nan);prob[known]=cross_val_predict(model,X[known],y[known],cv=cv,method='predict_proba')[:,1]
 model.fit(X[known],y[known]);prob[~known]=model.predict_proba(X[~known])[:,1];d['bleed_pred']=prob;assert np.isfinite(prob).all()
 c.emit('bleed prediction',dict(CV=scores,chosen=chosen,OOF_rows=int(known.sum()),unknown_rows=int(unknown.sum()),test_F1=f1_score(y[test],prob[test]>=.5),test_AUC=roc_auc_score(y[test],prob[test])))
 c.emit('stage1 split',dict(training_ids=d.loc[known,'id'].tolist(),test_ids=d.loc[test,'id'].tolist()))

def clinical_dead(c):
 d=c.state['clinical'];model=classpipe(RandomForestClassifier(n_estimators=60,max_depth=4,random_state=7));cols=['age','bp','signal','bleed_pred'];tr=d.ind!=2;te=~tr;model.fit(d.loc[tr,cols],d.loc[tr,'dead']);pred=model.predict(d.loc[te,cols]);prob=model.predict_proba(d.loc[te,cols])[:,1]
 c.emit('dead holdout',dict(F1=f1_score(d.loc[te,'dead'],pred),AUC=roc_auc_score(d.loc[te,'dead'],prob),features=cols,train_ids=d.loc[tr,'id'].tolist(),test_ids=d.loc[te,'id'].tolist()))


def mixed_table(d):
 wide=d.pivot(index='id',columns='method',values='score')[['online','hybrid','traditional']];g=d.drop_duplicates('id').set_index('id').loc[wide.index,'gender'];y=wide.to_numpy();N,m=y.shape;G=g.nunique();grand=y.mean();sub=y.mean(1);method=y.mean(0)
 SS_between=m*((sub-grand)**2).sum();SSg=0.;SSi=0.;pool=np.zeros((m,m))
 for label in sorted(g.unique()):
  z=y[(g==label).to_numpy()];mu=z.mean();mj=z.mean(0);SSg+=m*len(z)*(mu-grand)**2;SSi+=len(z)*((mj-mu-method+grand)**2).sum();res=z-mj;pool+=res.T@res
 SSm=N*((method-grand)**2).sum();SSt=((y-grand)**2).sum();SSsub=SS_between-SSg;SSerr=SSt-SS_between-SSm-SSi
 dfsub=N-G;dferr=(N-G)*(m-1);cov=pool/(N-G);H=linalg.helmert(m);contrast=H@cov@H.T;eps=float(np.trace(contrast)**2/((m-1)*np.trace(contrast@contrast)))
 out=[]
 for name,ss,df,ms_error,gg in [('gender',SSg,G-1,SSsub/dfsub,False),('method',SSm,m-1,SSerr/dferr,True),('interaction',SSi,(G-1)*(m-1),SSerr/dferr,True)]:
  F=float((ss/df)/ms_error);den=dferr if gg else dfsub;adj=eps if gg else 1.;out.append(dict(effect=name,SS=ss,F=F,df1=df*adj,df2=den*adj,p_value=float(stats.f.sf(F,df*adj,den*adj)),raw_df1=df,raw_df2=den,p_uncorrected=float(stats.f.sf(F,df,den))))
 return out,eps,wide

def mixed_anova(c):
 d=c.csv('learning');rows,eps,wide=mixed_table(d);c.state['mixed_rows']=rows;c.emit('mixed SS/GG',dict(epsilon=eps,table=rows,subjects=len(wide)))
 for row in rows:
  h0={'gender':'gender marginal means equal','method':'all method marginal means equal','interaction':'method contrasts equal across gender'}[row['effect']]
  c.test('Mixed ANOVA '+row['effect'],h0,'not '+h0,row['F'],row['p_value'],[row['df1'],row['df2']],context=row['effect']+' 효과')

def mixed_posthoc(c):
 d=c.csv('learning');wide=d.pivot(index='id',columns='method',values='score');pairs=[]
 for a,b in combinations(['online','hybrid','traditional'],2):
  r=stats.ttest_rel(wide[a],wide[b]);pairs.append((a,b,r))
 adj=multipletests([x[2].pvalue for x in pairs],method='holm')[1]
 for (a,b,r),p in zip(pairs,adj):c.test('Paired t '+a+'-'+b,'mean within-person difference=0','mean difference !=0',r.statistic,r.pvalue,len(wide)-1,p,context=a+'와 '+b+'의 평균 차이')
 inter=next(x for x in c.state['mixed_rows'] if x['effect']=='interaction');c.emit('interaction caution',dict(interaction_significant=inter['p_value']<.05,scope='pooled marginal contrasts; if interaction significant examine simple effects with separate correction'))

def score_ols(c):regression(c,c.csv('score'),'score',['study','failures','absences','support','internet'])

def haversine(lat1,lon1,lat2,lon2):
 a,b,c,d=map(np.radians,[lat1,lon1,lat2,lon2]);v=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
 return 2*6371*np.arcsin(np.sqrt(np.clip(v,0,1)))

def geo_validate(c):
 rides=c.csv('rides');st=c.csv('stations');weather=c.csv('weather_geo');assert not rides.duplicated(['date','station_id']).any();assert st.station_id.is_unique;assert not weather.duplicated(['date','sensor_id']).any()
 for d in [st,weather]:assert d.lat.between(-90,90).all() and d.lon.between(-180,180).all()
 c.state.update(rides=rides,stations=st,weather_geo=weather);c.emit('input shapes',dict(rides=rides.shape,stations=st.shape,weather=weather.shape))

def geo_map(c):
 st=c.state['stations'];sensors=c.state['weather_geo'][['sensor_id','lat','lon']].drop_duplicates().sort_values('sensor_id');assert sensors.sensor_id.is_unique
 mapping=[]
 for _,row in st.iterrows():
  dist=haversine(row.lat,row.lon,sensors.lat.to_numpy(),sensors.lon.to_numpy());j=int(np.argmin(dist));mapping.append(dict(station_id=row.station_id,sensor_id=sensors.iloc[j].sensor_id,distance_km=float(dist[j])))
 assert np.isclose(haversine(35,126,35,126),0);c.state['geo_mapping']=pd.DataFrame(mapping);c.emit('nearest sensor',mapping)

def geo_join(c):
 d=c.state['rides'];mapping=c.state['geo_mapping'];w=c.state['weather_geo'];joined=d.merge(mapping,on='station_id',validate='many_to_one').merge(w,on=['date','sensor_id'],how='left',validate='many_to_one',indicator=True)
 assert len(joined)==len(d) and (joined['_merge']=='both').all();joined.to_csv(c.output/'rides_weather.csv',index=False);c.emit('joined',dict(rows=len(joined),unmatched=int((joined['_merge']!='both').sum()),preview=joined.head()))

def cascade_select(c):
 d=c.csv('cascade');tr=d.id%5!=0;features=['x1','x2','x3','x4'];selected=[];history=[];current=np.inf;cv=KFold(3,shuffle=True,random_state=7)
 while True:
  scores={}
  for x in features:
   if x not in selected:scores[x]=float(np.mean(np.sqrt(-cross_val_score(regpipe(),d.loc[tr,selected+[x]],d.loc[tr,'target'],cv=cv,scoring='neg_mean_squared_error'))))
  if not scores:break
  chosen=min(scores,key=scores.get);value=scores[chosen]
  history.append(dict(candidates=scores,chosen=chosen,RMSE=value,accepted=current-value>=.01))
  if current-value<.01:break
  selected.append(chosen);current=value
 c.state.update(cascade=d,cascade_features=selected);fit=regpipe().fit(d.loc[tr,selected],d.loc[tr,'target']);c.emit('forward CV history',history);c.emit('chosen regression',dict(features=selected,holdout_RMSE=np.sqrt(mean_squared_error(d.loc[~tr,'target'],fit.predict(d.loc[~tr,selected])))))

def cascade_oof(c):
 d=c.state['cascade'];cols=c.state['cascade_features'];tr=d.id%5!=0;cv=KFold(3,shuffle=True,random_state=7);pred=np.empty(len(d));pred[tr]=cross_val_predict(regpipe(),d.loc[tr,cols],d.loc[tr,'target'],cv=cv);m=regpipe().fit(d.loc[tr,cols],d.loc[tr,'target']);pred[~tr]=m.predict(d.loc[~tr,cols]);d['predicted_target']=pred
 c.emit('OOF construction',dict(OOF_train_rows=int(tr.sum()),unseen_test_rows=int((~tr).sum()),input_columns=cols,selection_nested=False))

def cascade_classify(c):
 d=c.state['cascade'];cols=['x1','x2','x3','x4','predicted_target'];tr=d.id%5!=0;m=classpipe(RandomForestClassifier(n_estimators=60,max_depth=4,random_state=7)).fit(d.loc[tr,cols],d.loc[tr,'label']);prob=m.predict_proba(d.loc[~tr,cols])[:,1];pred=prob>=.5
 c.emit('cascade holdout',dict(F1=f1_score(d.loc[~tr,'label'],pred),AUC=roc_auc_score(d.loc[~tr,'label'],prob),features=cols,importance=m.named_steps['model'].feature_importances_,training_ids=d.loc[tr,'id'].tolist(),test_ids=d.loc[~tr,'id'].tolist()))

def json_flatten(c):
 data=c.js('records');d=pd.json_normalize(data,record_path='records',meta=['date',['meta','source']]).rename(columns={'sensors.temp':'temperature','sensors.load':'load','meta.source':'source'});d=d[['date','source','entity','temperature','load']]
 assert len(d)==sum(len(x['records']) for x in data) and not d.duplicated(['date','entity']).any();d[['temperature','load']]=d[['temperature','load']].apply(pd.to_numeric);c.state['json_flat']=d;d.to_csv(c.output/'flattened.csv',index=False);c.emit('flat shape/preview',dict(rows=len(d),preview=d.head()))

def json_aggregate(c):
 d=c.state['json_flat'];c.emit('entity mean temperature',d.groupby('entity').temperature.mean());c.emit('daily total load',d.groupby('date').load.sum())

def factor_suitability(c):
 d=c.csv('survey');R=d.corr().to_numpy();p=R.shape[0];n=len(d);stat=-(n-1-(2*p+5)/6)*np.linalg.slogdet(R)[1];df=p*(p-1)//2;inv=np.linalg.inv(R);partial=-inv/np.sqrt(np.outer(np.diag(inv),np.diag(inv)));np.fill_diagonal(partial,0);r=R.copy();np.fill_diagonal(r,0);kmo=(r*r).sum()/((r*r).sum()+(partial*partial).sum());c.emit('suitability',dict(correlation=R,KMO=kmo,n=n))
 c.test('Bartlett sphericity','population correlation is identity','not identity',stat,stats.chi2.sf(stat,df),df,context='문항 간 공통 상관 구조')

def factor_count(c):
 X=StandardScaler().fit_transform(c.csv('survey'));r=np.random.RandomState(36);eigen=np.linalg.eigvalsh(np.corrcoef(X,rowvar=False))[::-1];null=[]
 for _ in range(200):
  perm=np.column_stack([r.permutation(X[:,j]) for j in range(X.shape[1])]);null.append(np.linalg.eigvalsh(np.corrcoef(perm,rowvar=False))[::-1])
 threshold=np.percentile(null,95,axis=0);above=eigen>threshold;k=0
 for flag in above:
  if not flag:break
  k+=1
 k=max(k,1);c.state['factor_count']=k;c.emit('parallel analysis',dict(eigen=eigen,permuted95=threshold,retained=k,permutations=200,rule='leading consecutive eigenvalues above matching 95th percentile'))
 plt.figure(figsize=(5,3));plt.plot(range(1,7),eigen,'o-',label='Observed');plt.plot(range(1,7),threshold,'o--',label='Permutation 95%');plt.legend();plt.xlabel('Component');plt.ylabel('Eigenvalue');c.fig('factor_scree')

def factor_fit(c):
 X=StandardScaler().fit_transform(c.csv('survey'));k=c.state['factor_count'];fa=FactorAnalysis(n_components=k,rotation='varimax',random_state=36,max_iter=1000).fit(X);L=fa.components_.T;cov=fa.get_covariance();R=np.cov(X,rowvar=False,bias=True)
 c.emit('factor model',dict(factors=k,loadings=L,communality=(L*L).sum(1),uniqueness=fa.noise_variance_,covariance_RMSE=np.sqrt(np.mean((cov-R)**2)),iterations=fa.n_iter_))

TASKS={name:value for name,value in list(globals().items()) if callable(value) and name in '''health_eda health_pca health_ols health_models accident_tests accident_sampling accident_models transport headset binomial obesity_prepare obesity_model checkup_features checkup_select checkup_models bayes ads_full ads_backward poisson_probability housing_diagnostics housing_pca housing_robust category_expected category_mc salary hepatitis_prepare hepatitis_binary hepatitis_multi subway_merge subway_models defect poisson_gof delay arma clinical_prepare clinical_oof clinical_dead power_merge power_models mixed_anova mixed_posthoc score_ols geo_validate geo_map geo_join cascade_select cascade_oof cascade_classify json_flatten json_aggregate factor_suitability factor_count factor_fit'''.split()}
