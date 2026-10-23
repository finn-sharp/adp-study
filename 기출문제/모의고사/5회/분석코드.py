"""모의고사 5회: 이 파일 폴더에서 실행하는 순차 예시 풀이."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt
from IPython.display import display
HERE=Path.cwd()
REPO=next(p for p in [HERE]+list(HERE.parents) if (p/'src/Recipes/loader.py').exists())
sys.path.insert(0,str(REPO))
from src.Optimization.production_allocation import run_production_allocation
from src.Visualization.optimization.convergence import plot_convergence
from src.Visualization.regression.observed_predicted import plot_observed_predicted
from src.Visualization.qq.normal import plot_normal
quality=pd.read_csv(HERE/'data/quality.csv')
processes=pd.read_csv(HERE/'data/processes.csv')
cycles=pd.read_csv(HERE/'data/cycle_times.csv')
answers=[]
def record(question,**values):
    answers.append(dict(question=question,**values))
    dest=HERE/'실행결과';dest.mkdir(exist_ok=True)
    (dest/'답안수치.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


# 1-1
display(quality.describe()); display(quality.isna().sum()); record('1-1',n=len(quality),missing_temperature=int(quality.temperature.isna().sum()))

# 1-2
train,test=train_test_split(np.arange(len(quality)),test_size=.3,random_state=2026)
X=quality[['temperature','pressure','speed']];y=quality['yield']
models={'linear':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LinearRegression())]),'forest':Pipeline([('impute',SimpleImputer(strategy='median')),('model',RandomForestRegressor(n_estimators=60,max_depth=4,random_state=2026,n_jobs=1))])}
cv=KFold(3,shuffle=True,random_state=2026)
record('1-2',train_indices=train.tolist(),test_indices=test.tolist());print('train/test:',len(train),len(test))

# 1-3
scores={name:np.sqrt(-cross_val_score(model,X.iloc[train],y.iloc[train],cv=cv,scoring='neg_mean_squared_error')) for name,model in models.items()}
comparison=pd.DataFrame([dict(model=name,cv_rmse=float(s.mean()),cv_std=float(s.std())) for name,s in scores.items()]);display(comparison)
selected=min(scores,key=lambda name:scores[name].mean());record('1-3',comparison=comparison.to_dict('records'),selected=selected)

# 1-4
fitted=models[selected].fit(X.iloc[train],y.iloc[train]);pred=fitted.predict(X.iloc[test])
metrics=dict(rmse=float(np.sqrt(mean_squared_error(y.iloc[test],pred))),mae=float(mean_absolute_error(y.iloc[test],pred)),r2=float(r2_score(y.iloc[test],pred)))
fig,ax=plt.subplots(figsize=(6,4));plot_observed_predicted(y.iloc[test],pred,ax=ax);plt.tight_layout();plt.show();print(metrics);record('1-4',**metrics)

# 2-1
H=8*5*50*60;unit=processes.purchase_cost+processes.operating_cost;one=H/processes.minutes_per_unit;print(H,one.tolist());record('2-1',annual_minutes=H,one_machine_capacity=one.tolist(),unit_cost=unit.tolist())

# 2-2
required=np.ceil(50000*processes.minutes_per_unit/H).astype(int);capacity=required*one;print(required.tolist(),capacity.tolist());record('2-2',initial_cost=float(unit.sum()),initial_production=float(one.min()),shortfall=50000-float(one.min()),minimum_required=required.tolist(),capacities=capacity.tolist())

# 2-3
base_args=(processes.minutes_per_unit,processes.purchase_cost,processes.operating_cost)
allocation=run_production_allocation(*base_args);print(allocation.process_table);print(allocation.ga_result.stop_reason)
fig,ax=plt.subplots(figsize=(6,4));plot_convergence(allocation.ga_result,ax=ax);plt.tight_layout();plt.show()
record('2-3',solution=allocation.ga_result.best_solution.tolist() if allocation.ga_result.feasible else None,cost=allocation.total_cost,production=allocation.annual_production,feasible=allocation.ga_result.feasible)

# 2-4
from itertools import product
exhaustive=[(float(np.array(n)@unit),n) for n in product(range(1,11),repeat=3) if min(np.array(n)*H/processes.minutes_per_unit)>=50000]
exact_cost,exact_solution=min(exhaustive)
runs=[]
for seed in range(5):
    r=run_production_allocation(*base_args,overrides={'random_state':seed,'verify_exact':False})
    runs.append(dict(seed=seed,feasible=r.ga_result.feasible,cost=r.total_cost,optimal=r.total_cost==exact_cost))
print(exact_cost,exact_solution);display(pd.DataFrame(runs));record('2-4',exact_cost=exact_cost,exact_solution=list(exact_solution),ga_gap=allocation.objective_gap,runs=runs,feasible_rate=float(np.mean([r['feasible'] for r in runs])),optimal_rate=float(np.mean([r['optimal'] for r in runs])),minimum_cost=min(r['cost'] for r in runs if r['feasible']),maximum_cost=max(r['cost'] for r in runs if r['feasible']))

# 2-5
scenarios=[]
for demand in [70000.,90000.]:
    r=run_production_allocation(*base_args,overrides={'target_production':demand});scenarios.append(dict(target=demand,solution=r.exact_solution.tolist(),cost=r.exact_cost,capacities=(r.exact_solution*H/processes.minutes_per_unit).tolist()))
impossible=run_production_allocation(*base_args,overrides={'budget':29000.});print(scenarios,impossible.exact_status);record('2-5',scenarios=scenarios,budget_status=impossible.exact_status)

# 2-6
purchase=float(np.array(exact_solution)@processes.purchase_cost);operating=float(np.array(exact_solution)@processes.operating_cost)
print('구매/연간 운영:',purchase,operating);record('2-6',purchase_cost=purchase,annual_operating_cost=operating,first_year_cost=purchase+operating,capacity_slack=allocation.annual_production-50000)

# 3-1
summary=cycles.groupby(['process','phase']).minutes.agg(['size','mean','std']);display(summary);record('3-1',summary=summary.reset_index().to_dict('records'))

# 3-2
normality=[];fig,axes=plt.subplots(2,3,figsize=(12,7))
for ax,((process,phase),part) in zip(axes.flat,cycles.groupby(['process','phase'])):
    residual=part.minutes-part.minutes.mean();stat,p=stats.shapiro(residual)
    normality.append(dict(process=int(process),phase=phase,statistic=float(stat),p_value=float(p)))
    plot_normal(residual,ax=ax,overrides={'fit_parameters':True,'title':str(process)+' '+phase})
adjusted=multipletests([r['p_value'] for r in normality],method='holm')[1]
for r,p in zip(normality,adjusted):r.update(p_holm=float(p),reject_H0=bool(p<.05),H0='population is normal',H1='population is not normal')
plt.tight_layout();plt.show();display(pd.DataFrame(normality));record('3-2',tests=normality)

# 3-3
old=cycles[(cycles.process==3)&(cycles.phase=='old')].minutes.to_numpy();new=cycles[(cycles.process==3)&(cycles.phase=='new')].minutes.to_numpy()
a=new.var(ddof=1)/len(new);b=old.var(ddof=1)/len(old);difference=float(new.mean()-old.mean());stat=difference/np.sqrt(a+b);df=(a+b)**2/(a*a/(len(new)-1)+b*b/(len(old)-1));p=float(stats.t.cdf(stat,df))
report=dict(method='Welch one-sided',H0='mu_new-mu_old >= 0',H1='mu_new-mu_old < 0',difference=difference,statistic=float(stat),df=float(df),p_value=p,reject_H0=bool(p<.05),conclusion='평균 감소를 지지' if p<.05 else '평균 감소 근거 부족')
print(report);record('3-3',**report)

# 3-4
upper=[]
for process,part in cycles[cycles.phase=='new'].groupby('process'):
    bound=float(part.minutes.mean()+stats.t.ppf(1-.05/3,len(part)-1)*part.minutes.std(ddof=1)/np.sqrt(len(part)))
    upper.append(bound)
conservative=run_production_allocation(upper,processes.purchase_cost,processes.operating_cost)
print(upper,conservative.exact_solution,conservative.exact_cost);record('3-4',mean_upper=upper,solution=conservative.exact_solution.tolist(),cost=conservative.exact_cost,family_confidence=.95)
