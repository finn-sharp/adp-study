"""회차별 후기 PDF의 상세 조건을 별도 보충 문제로 복원합니다."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def register(book, section, q, exams):
    root = Path(__file__).resolve().parents[2] / '기출문제'
    rng = np.random.RandomState(20261015)
    manifest = json.loads((root/'복원연습_데이터명세.json').read_text())

    def save(folder, name, data):
        path = root/folder/'data'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        relative = str(path.relative_to(root))
        manifest[:] = [m for m in manifest if m['path'] != relative]
        manifest.append({'path':relative,'kind':'synthetic','seed':20261015,'rows':len(data),
                         'note':'회차별 후기의 컬럼과 조건을 보완한 합성 자료. 원 시험 수치가 아님.'})

    folder='18회/보충';n=500
    amount=rng.lognormal(7, .7, n);count=rng.randint(1,50,n)
    grade=pd.qcut(amount/count,5,labels=False)+1
    customers=pd.DataFrame({'id':np.arange(n),'days':rng.randint(1,365,n),
        'duration':rng.randint(1,730,n),'count':count,'amount':amount,'grade':grade})
    customers.loc[rng.choice(n,20,False),'amount']=np.nan
    save(folder,'customers.csv',customers)
    b=book(folder,'ADP 18회 - 상세 후기 보충','adp_18_summary.pdf pp.1~6: 5등급 분류, 파생변수, SOM, RF/신경망 포함 4개 모델')
    s=section(b,'1. 고객등급 다중분류','customers.csv','id는 제외. 5개 등급 합성 자료 500행. 훈련/검증/시험을 먼저 고정하며 지도 라벨은 SOM 격자의 학습 표본 분포에만 사용.')
    q(s,'1-1','결측·이상치를 탐색하고 고객등급을 설명할 파생변수 3개를 생성하시오.',"""
customers=load('customers.csv');print(customers.describe());print(customers.isna().sum())
customers.drop(columns=['id','grade']).hist(figsize=(9,6));plt.tight_layout();plt.show()
X=customers.drop(columns=['id','grade']).copy();y=customers.grade
X['mean_purchase']=X.amount/X['count'].clip(lower=1)
X['purchase_frequency']=X['count']/X.duration.clip(lower=1)
X['recency_ratio']=X.days/X.duration.clip(lower=1)
train,test=train_test_split(np.arange(len(y)),test_size=.3,random_state=1234,stratify=y)
fit,valid=train_test_split(train,test_size=.25,random_state=1234,stratify=y.iloc[train])
print(X.head());print('fit/valid/test:',len(fit),len(valid),len(test))
""",'구매당 금액·거래 빈도·상대 최근성으로 정의합니다. 0분모를 명시적으로 방어하며 결측 대치·표준화는 fit 내부에서 적합합니다.')
    q(s,'1-2','SOM의 학습 격자에 학습 라벨 분포를 매핑하여 5등급 예측 확률을 만들고 validation macro F1 및 OvR ROC로 평가하시오.',"""
from src.MachineLearning.Decomposition.som import run_som
from sklearn.metrics import roc_auc_score
imputer=SimpleImputer(strategy='median').fit(X.iloc[fit]);fit_X=imputer.transform(X.iloc[fit]);valid_X=imputer.transform(X.iloc[valid])
som=run_som(fit_X,overrides={'rows':4,'columns':4,'n_iter':800})
train_bmu=som.model.transform(fit_X).astype(int);valid_bmu=som.model.transform(valid_X).astype(int)
classes=np.array([1,2,3,4,5]);global_counts=np.array([(y.iloc[fit]==c).sum() for c in classes],float)
prior=global_counts/global_counts.sum();cell_prob={}
for row in range(4):
    for col in range(4):
        mask=(train_bmu[:,0]==row)&(train_bmu[:,1]==col)
        counts=np.array([(y.iloc[fit].to_numpy()[mask]==c).sum() for c in classes],float)
        cell_prob[(row,col)]=(counts+prior)/(counts.sum()+1.)
probs=np.array([cell_prob[tuple(cell)] for cell in valid_bmu]);pred=classes[probs.argmax(axis=1)]
print({'validation_macro_F1':f1_score(y.iloc[valid],pred,average='macro'),'OvR_macro_AUC':roc_auc_score(y.iloc[valid],probs,multi_class='ovr',labels=classes)})
for i,c in enumerate(classes):
    fpr,tpr,_=roc_curve((y.iloc[valid]==c).astype(int),probs[:,i]);plt.plot(fpr,tpr,label=str(c))
plt.legend(title='grade');plt.xlabel('FPR');plt.ylabel('TPR');plt.show()
""",'SOM 본체는 비지도입니다. 각 BMU의 학습 라벨 빈도에 전체 학습 prior 1개를 더해 빈 격자도 처리합니다. validation 라벨은 격자 매핑에 사용하지 않습니다.')
    q(s,'1-3','RF·신경망을 포함한 지도모델 4개를 validation F1·ROC로 비교하고, 선정 모델을 train CV로 튜닝한 뒤 test를 한 번 평가하시오.',"""
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from src.MachineLearning.Optimization.grid_search import run_grid_search
algorithms={'rf':RandomForestClassifier(n_estimators=40,random_state=1234),'mlp':MLPClassifier(hidden_layer_sizes=(20,),max_iter=400,random_state=1234),'tree':DecisionTreeClassifier(random_state=1234),'logistic':LogisticRegression(max_iter=1000)}
spaces={'rf':{'model__max_depth':[3,6,None]},'mlp':{'model__alpha':[.001,.01]},'tree':{'model__max_depth':[3,6,None]},'logistic':{'model__C':[.1,1.,10.]}}
rows=[]
for name,est in algorithms.items():
    pipe=tabular_pipeline(X,est).fit(X.iloc[fit],y.iloc[fit]);prob=pipe.predict_proba(X.iloc[valid]);pred=pipe.predict(X.iloc[valid])
    rows.append({'model':name,'F1':f1_score(y.iloc[valid],pred,average='macro'),'AUC':roc_auc_score(y.iloc[valid],prob,multi_class='ovr',labels=pipe.classes_)})
    for i,c in enumerate(pipe.classes_):
        fpr,tpr,_=roc_curve((y.iloc[valid]==c).astype(int),prob[:,i]);plt.plot(fpr,tpr,label=str(c))
    plt.title(name+' validation ROC');plt.legend();plt.xlabel('FPR');plt.ylabel('TPR');plt.show()
ranking=pd.DataFrame(rows);print(ranking);winner=ranking.loc[ranking.F1.idxmax(),'model']
tuned=run_grid_search(X.iloc[train],y.iloc[train],tabular_pipeline(X,algorithms[winner]),overrides={'scoring':'f1_macro','search_space':spaces[winner]})
pred=tuned.model.predict(X.iloc[test]);prob=tuned.model.predict_proba(X.iloc[test])
print(winner,tuned.best_params,{'test_macro_F1':f1_score(y.iloc[test],pred,average='macro'),'test_OvR_AUC':roc_auc_score(y.iloc[test],prob,multi_class='ovr',labels=tuned.model.classes_)})
pd.DataFrame({'row_id':test,'actual':y.iloc[test].to_numpy(),'prediction':pred}).to_csv(OUT/'selected_predictions.csv',index=False)
""",'최종 튜닝의 train은 fit+validation입니다. 모델 선정의 validation 사용과 CV 재적합을 구분하고 test는 최종 단일 평가합니다. 튜닝 개선은 합성 자료에서 보장하지 않습니다.')

    folder='19회/보충';t=np.arange(80)
    save(folder,'traffic_quarterly.csv',pd.DataFrame({'date':pd.date_range('2000-01-01',periods=80,freq='QS'),'value':100+.7*t+12*np.sin(2*np.pi*t/4)+rng.normal(0,2,80)}))
    b=book(folder,'ADP 19회 - 분기 시계열 보충','adp_19_summary.pdf pp.4~9: 20년 분기별 Traffic EPS')
    s=section(b,'1. 분기 시계열','traffic_quarterly.csv','80개 관측, 1년=4분기. 기존 월별 연습과 별개이며 정상성·정규성·이분산·자기상관을 구분.')
    q(s,'2-1','분해와 ADF로 정상성을 점검하고 차분 전후를 비교하시오.',"""
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
quarter=load('traffic_quarterly.csv');train_q=quarter.value.iloc[:64]
seasonal_decompose(pd.Series(train_q.to_numpy(),index=pd.to_datetime(quarter.date.iloc[:64])),period=4).plot();plt.show()
print({'ADF_raw':adfuller(train_q)[1],'ADF_seasonal_difference':adfuller(train_q.diff(4).dropna())[1]})
""",'ADF 귀무가설은 단위근입니다. 정규성의 검정이 아니며 이분산은 시간별 분산 변화, 자기상관은 잔차 간 상관입니다.')
    q(s,'2-2','SARIMA 후보를 학습 AIC로 비교해 선택하고, 잔차·잡음과 보류 RMSE/MAE를 보고하시오.',"""
quarter_model,quarter_metrics=seasonal_forecast(quarter,OUT,period=4)
pd.Series(quarter_model.resid).plot();plt.title('Quarterly residuals');plt.show()
sm.graphics.tsa.plot_acf(quarter_model.resid,lags=12);plt.show()
print(sm.stats.acorr_ljungbox(quarter_model.resid,lags=[4,8],return_df=True));print(stats.shapiro(quarter_model.resid))
""",'학습 AIC 선정과 test 오차를 구분합니다. 초기 상태 추정의 영향을 고려하고 잔차 정규성만으로 독립성·모형 적합성을 단정하지 않습니다.')

    folder='20회/보충';n=348;t=np.arange(n);actual=15+10*np.sin(2*np.pi*t/366)+rng.normal(0,2,n)
    dates=pd.date_range('2016-01-03',periods=n)
    save(folder,'temps.csv',pd.DataFrame({'year':dates.year,'month':dates.month,'day':dates.day,'hour':0,'week':dates.day_name(),'temp_1':np.r_[actual[0],actual[:-1]],'temp_2':np.r_[actual[:2],actual[:-2]],'actual':actual,'average':15+10*np.sin(2*np.pi*t/366),'friend':actual+rng.normal(0,3,n)}))
    b=book(folder,'ADP 20회 - 날씨 회귀 상세 후기 보충','adp_20_summary.pdf pp.1~4: temps.csv RF·SVM·예측 기준선')
    s=section(b,'1. 날씨 예측','temps.csv','348개 합성 일별 관측. average=미리 알려진 전년도 기준선으로 가정. friend는 예측 시점 가용성 불명확하여 제외. 미래 예측이므로 시간순 7:3.')
    q(s,'1-1','EDA·요일 원핫 전처리와 시간순 분할을 설명하고 전년도 평균 기준선 MAE를 구하시오.',"""
weather=load('temps.csv');print(weather.describe(include='all'));print(weather.isna().sum())
features=['month','day','week','temp_1','temp_2','average'];X=weather[features];y=weather.actual
cut=int(len(y)*.7);train=np.arange(cut);test=np.arange(cut,len(y))
print('baseline test MAE:',np.mean(np.abs(y.iloc[test]-weather.average.iloc[test])))
weather.plot(y=['actual','average'],figsize=(9,3));plt.show()
""",'후기는 랜덤 분할을 제안하지만 미래 날씨 예측에서는 시간순 평가로 보완합니다. 평균 기준선은 학습 y 평균과 전년도 average를 구분합니다.')
    q(s,'1-2','RF와 SVR를 train TimeSeriesSplit RMSE로 비교하고 변수 중요도를 분석하시오.',"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
from sklearn.inspection import permutation_importance
models={};rows=[]
for name,est in [('rf',RandomForestRegressor(n_estimators=60,random_state=1234)),('svr',SVR())]:
    pipe=tabular_pipeline(X,est);scores=cross_val_score(pipe,X.iloc[train],y.iloc[train],cv=TimeSeriesSplit(3),scoring='neg_mean_squared_error')
    rows.append({'model':name,'CV_RMSE':np.sqrt(-scores).mean()});models[name]=pipe
ranking=pd.DataFrame(rows);print(ranking);winner=ranking.loc[ranking.CV_RMSE.idxmin(),'model'];final=models[winner].fit(X.iloc[train],y.iloc[train])
importance=permutation_importance(final,X.iloc[train],y.iloc[train],n_repeats=5,random_state=1234,scoring='neg_mean_squared_error')
pd.Series(importance.importances_mean,index=features).sort_values().plot.barh();plt.title('Training permutation importance (descriptive)');plt.show()
""",'RBF SVR는 coef_가 없어 공통 permutation importance를 사용합니다. 여기 중요도는 학습 자료의 설명이며 일반화 중요도가 아닙니다.')
    q(s,'1-3','선정 모델의 보류 RMSE·MAE·R²를 기준선과 비교하고 운영상 한계를 서술하시오.',"""
pred=final.predict(X.iloc[test]);print(winner,{'RMSE':np.sqrt(mean_squared_error(y.iloc[test],pred)),'MAE':np.mean(np.abs(y.iloc[test]-pred)),'R2':r2_score(y.iloc[test],pred)})
pd.DataFrame({'row_id':test,'actual':y.iloc[test].to_numpy(),'prediction':pred,'baseline':weather.average.iloc[test].to_numpy()}).to_csv(OUT/'weather_predictions.csv',index=False)
print('합성 자료의 튜닝 우위는 실제 데이터 성능 근거가 아님. lag는 예측 시점 관측 가능성을 점검. MAPE는 실제값 분모, 섭씨 0 부근에서는 불안정.')
""",'test로 최종 모델을 고르지 않습니다. lag 입력은 매일 과거 실제값이 새로 관측되는 one-step 운영 가정입니다.')

    b=book('22회/후기보충','ADP 22회 - 수익 최적화 후기 보충','adp_22_summary.pdf p.2: 제품1 12만원·제품2 18만원, 재료 제약 최대 수익')
    s=section(b,'1. 제품 생산 수익','데이터 없음','후기에서 미제공한 재료 제약은 제품1=(1,1,1), 제품2=(2,2,2), 재료량=(12,12,18)로 보완. 기존 기대 개수 문제와 별개.')
    q(s,'3-1','정수 생산량 조건에서 재료 제약 하 최대 수익과 생산 조합을 구하시오.',"""
rows=[{'product1':x,'product2':y,'profit_10000KRW':12*x+18*y} for x in range(13) for y in range(7) if x+2*y<=12 and x+2*y<=18]
production=pd.DataFrame(rows);best=production.loc[production.profit_10000KRW.idxmax()];print(best);production.to_csv(OUT/'feasible_production.csv',index=False)
assert best.profit_10000KRW==144
""",'재료 계수는 원 후기에서 제공되지 않아 명시적으로 보완했습니다. 생산순서 확률과 결정론적 수익 최적화는 서로 다른 문제입니다.')
    # 18회 상세 후기의 분류 요구를 주 문제지에도 반영. 기존 보충은 집중 연습으로 보존.
    customer_exam = exams['18회/보충']
    save('18회','customers.csv',customers)
    main18=book('18회','ADP 18회 - 상세 후기 대조 복원','adp_18_summary.pdf pp.1~11; 기존 주제 모음 대조')
    main18['sections'] = customer_exam['sections'].copy()
    save('18회','english_documents.csv',pd.DataFrame({'text':[
        'Customers purchase steel products and services.',
        'Traffic accidents increase during rainy seasons.',
        'Customers compare product prices and quality.',
        'Students analyze traffic and customer data.']}))
    save('18회','english_lexicon.csv',pd.DataFrame({'word':['customers','steel','products','services','traffic','accidents','seasons','product','prices','quality','students','data'], 'pos':['Noun']*12}))
    s=section(main18,'2. 영문 텍스트','english_documents.csv + english_lexicon.csv','후기 원문 TXT는 미첨부. 영문 합성 문서와 명사 POS 사전을 사용. 실제 영어 POS 모델을 실행한 결과가 아님.')
    q(s,'2-1','영문 문서를 소문자화·특수기호 정제·불용어 제거하고 명사 빈도를 시각화하시오.',"""
english=load('english_documents.csv');lexicon=load('english_lexicon.csv');stop={'and','the','a','an','during'}
tokens=[[t for t in clean_text(text.lower()).split() if t not in stop] for text in english.text]
vocab=set(lexicon.word);nouns=[[t for t in document if t in vocab] for document in tokens]
print(tokens);word_plot(nouns,12)
""",'원문에 없는 합성 빈도이며 동봉 사전에 명시된 명사만 추출합니다. 실제 원문에는 설치된 영어 POS 모델과 모델 버전을 명시해야 합니다.')
    s=section(main18,'3. 교통사고 계절 시계열','series.csv','합성 월별 144개. 학습 앞 80% 안에서 보간·진단·후보 선택, 마지막 20% 고정 예측.')
    q(s,'3-1','정상성을 진단하고 차분 전후와 ACF/PACF를 비교하시오.',"""
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
traffic18=load('series.csv');train18=traffic18.value.iloc[:int(len(traffic18)*.8)].interpolate().ffill().bfill()
print({'ADF_raw':adfuller(train18)[1],'ADF_diff':adfuller(train18.diff().dropna())[1],'ADF_seasonal_diff':adfuller(train18.diff(12).dropna())[1]})
fig,axes=plt.subplots(2,2,figsize=(10,6));plot_acf(train18,lags=24,ax=axes[0,0]);plot_pacf(train18,lags=24,ax=axes[0,1]);plot_acf(train18.diff().dropna(),lags=24,ax=axes[1,0]);plot_pacf(train18.diff().dropna(),lags=24,ax=axes[1,1]);plt.tight_layout();plt.show()
""",'차분한 자료에 ARMA를 적합하거나 원자료에 ARIMA를 적합하는 관점을 구분합니다. 단위근 검정 하나로 모든 정상성 조건을 입증하지 않습니다.')
    q(s,'3-2','ARIMA/SARIMA 후보를 최소 3개 제시하고 학습 AIC로 비교하시오.',"""
traffic18_model,traffic18_metrics=seasonal_forecast(traffic18,OUT,period=12,orders=[(1,1,0),(0,1,1),(1,1,1)])
""",'3개 비계절 차수에 동일한 계절항 (1,0,0,12)을 결합합니다. 수렴한 후보만 비교하고 test로 차수를 고르지 않습니다.')
    q(s,'3-3','선정 모형의 잔차 ACF·정규성·자기상관을 진단하시오.',"""
res18=pd.Series(traffic18_model.resid).iloc[12:]
fig,axes=plt.subplots(1,2,figsize=(9,3));plot_acf(res18,lags=24,ax=axes[0]);sm.qqplot(res18,line='s',ax=axes[1]);plt.tight_layout();plt.show()
print(stats.shapiro(res18));print(sm.stats.acorr_ljungbox(res18,lags=[12,24],model_df=3,return_df=True))
""",'초기 상태 추정의 영향을 줄여 첫 계절을 제외합니다. fitted 잔차 진단이며 관측 잡음의 실제값과 동일하다고 주장하지 않습니다. Ljung-Box model_df는 선택 모형의 AR/MA 계수 수입니다.')
    q(s,'3-4','고정 horizon 보류 예측의 RMSE·MAE를 보고하고 MAPE의 한계를 설명하시오.',"""print(traffic18_metrics);print('합성 교통사고 자료. count 비음수 제약·0값에 대한 MAPE 불안정, 계절 나이브 기준선과 추가 비교 필요.')""",'예측 결과 CSV는 앞 셀에서 저장했습니다. 누락된 보류 실제값은 오차 집계에서만 제외하며 학습에 보류 값을 사용하지 않습니다.')

    # 19회 원 후기에 맞춰 주 문제의 월별 변형을 분기별로 교체.
    save('19회','traffic_quarterly.csv',pd.read_csv(root/'19회/보충/data/traffic_quarterly.csv'))
    main19=exams['19회'];main19['source']='adp_19_summary.pdf pp.1~9 및 기출문제 모음'
    main19['sections']=main19['sections'][:1]
    main19['sections'][0]['questions'][1]['code']=main19['sections'][0]['questions'][1]['code'].replace('OUT,algorithms=credit_alg','OUT,seed=2022,algorithms=credit_alg')
    s=section(main19,'2. Traffic EPS 분기 시계열','traffic_quarterly.csv','20년×4분기=80개 합성 자료. period=4. 원 시험값은 아님.')
    for original in exams['19회/보충']['sections'][0]['questions']:
        q(s,original['id'],original['text'],original['code'],original['reason'],'adp_19_summary.pdf pp.4~9')
    q(s,'2-2','학습 시계열에 STL 분해·차분을 적용하고 정규성·정상성·이분산의 차이를 설명하시오.',"""
from statsmodels.tsa.seasonal import STL
STL(pd.Series(train_q.to_numpy(),index=pd.to_datetime(quarter.date.iloc[:64])),period=4).fit().plot();plt.show()
print(train_q.rolling(4).agg(['mean','std']).tail());print('정규성은 분포, 정상성은 시간 불변성, 이분산은 분산 변화, 자기상관은 시차 의존성입니다.')
""",'후기의 개념 혼용을 교정합니다. 합성 예제와 실제 데이터의 분기 시간 격자를 확인해야 합니다.')
    initial, combined, decomposition = s['questions']
    fit_code, residual_code = combined['code'].split('\n', 1)
    s['questions'] = [initial, decomposition]
    q(s,'2-3','SARIMA 후보를 최소 3개 비교하고 학습 AIC로 모형을 선정하시오.',fit_code,
      '비계절 차수 3개와 분기 계절항 period=4를 비교합니다. 수렴·유한 AIC 조건을 만족하는 후보만 선택합니다.')
    q(s,'2-4','선정 모형의 잔차·잡음을 시각화하고 자기상관·정규성을 진단하시오.',residual_code,
      '잔차 자기상관과 이분산을 구분합니다. 모형 적합 잔차를 실제 잡음과 동일시하지 않습니다. 보류 오차는 후보 선택에 사용하지 않습니다.')

    # 요약의 여행 예제는 실제 원 시험 지문으로 오인하지 않도록 별도 보충.
    sample=pd.DataFrame({'sex':[1]*15+[2]*15,'travel':([1]*5+[2]*5+[3]*5)*2,
        'points':[2,3,4,4,2,4,5,5,3,3,8,9,8,7,7,6,7,6,5,7,3,4,5,4,5,6,6,6,7,8]})
    save('21회/보충','travel_example.csv',sample)
    manifest[-1]['kind']='source_example';manifest[-1]['seed']=None
    manifest[-1]['note']='adp_21_summary.pdf pp.7~8의 30명 여행태도 학습 예제 값 그대로. 원 시험 데이터로 주장하지 않음.'
    b=book('21회/보충','ADP 21회 - 요약 자료 ANOVA 학습 예제','adp_21_summary.pdf pp.7~8 (인쇄물 14~16/16)',note='요약 자료에 실제 제시된 예제 30명 수치입니다. 실제 21회 시험 데이터라고 단정하지 않습니다.')
    s=section(b,'1. 성별×여행빈도','travel_example.csv','sex 두 범주, travel 세 범주, points 1~9 점수. 균형 2×3 설계.')
    q(s,'4-1','상호작용 포함 이원 ANOVA와 상호작용 그래프를 작성하고 셀 평균을 해석하시오.',"""
travel=load('travel_example.csv');anova(travel,'points','sex','travel')
from statsmodels.graphics.factorplots import interaction_plot
interaction_plot(travel.travel,travel.sex,travel.points);plt.show()
print(travel.groupby(['sex','travel']).points.agg(['count','mean','std']))
""",'상호작용이 유의하면 주변 평균만으로 일정한 효과를 주장하지 않고 단순 효과·셀 평균을 우선합니다. 주효과가 수학적으로 무의미해지는 것은 아닙니다. ordinal 점수 OLS 가정과 작은 셀 표본 한계를 명시합니다.')
    (root/'복원연습_데이터명세.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
