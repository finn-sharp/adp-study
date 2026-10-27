"""
================================================================================
applied_mock_data.py - 회차별 명시 생성 원리의 고정 합성 데이터. 유리한 p값으로 재선택하지 않습니다.
================================================================================
설명:
    - 회차별 명시 생성 원리의 고정 합성 데이터. 유리한 p값으로 재선택하지 않습니다.
분석 가정:
    - 저장소 경로와 입력 명세를 기준으로 실행합니다. 생성·검사 대상과 외부 의존성은 각 실행 함수의 조건을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""
import numpy as np
import pandas as pd
from scipy.special import expit

def generate(folder,plan):
    data=folder/'data';data.mkdir(parents=True,exist_ok=True);rng=np.random.RandomState(plan['seed']);n=240;X=rng.normal(size=(n,4));X[:,2]=.96*X[:,0]+.12*rng.normal(size=n);e=rng.normal(size=n);kind=plan['regression_kind']
    if kind=='lognormal':y=np.exp(1+.5*X[:,0]-.3*X[:,1]+.3*e)
    elif kind=='quadratic':y=4+2*X[:,0]**2+.7*X[:,1]+.4*e
    elif kind=='heteroscedastic':y=5+2*X[:,0]-.5*X[:,1]+np.exp(.65*X[:,0])*.6*e
    elif kind=='influence':
        y=5+2*X[:,0]-.5*X[:,1]+.5*e;out=rng.choice(n,5,replace=False);X[out,0]+=5;X[out,2]+=4;y[out]+=20
    elif kind=='ordered':
        for i in range(1,n):e[i]+=.75*e[i-1]
        y=5+2*X[:,0]-.5*X[:,1]+e
    elif kind=='exponential':X[:,0]=rng.uniform(.1,3,n);X[:,2]=rng.normal(size=n);y=2*np.exp(.6*X[:,0])+1+.3*e
    elif kind=='growth':X[:,0]=rng.uniform(0,3,n);X[:,2]=rng.normal(size=n);y=10*expit(2*(X[:,0]-1.5))+.25*e
    elif kind=='gamma':
        mu=np.exp(1+.45*X[:,0]-.2*X[:,1]);y=rng.gamma(6,mu/6)
    elif kind=='poisson':y=rng.poisson(np.exp(1+.4*X[:,0]-.2*X[:,1]))
    else:y=20+2*X[:,0]-.5*X[:,1]+.5*e
    # Missingness is generated after y, independent of the outcome.
    X[rng.choice(n,8,replace=False),1]=np.nan
    reg=pd.DataFrame(X,columns=plan['features']);reg.insert(0,'id',np.arange(1,n+1));reg['target']=y;reg.to_csv(data/'regression.csv',index=False,float_format='%.10f')
    m=360;C=rng.normal(size=(m,4));C[:,3]=.95*C[:,0]+.12*rng.normal(size=m);logit=-1.5+1.1*C[:,0]-.65*C[:,1]+(.45*C[:,2]**2 if plan['number']%2==0 else .3*C[:,2]);target=rng.binomial(1,expit(logit));C[rng.choice(m,9,replace=False),2]=np.nan
    classification=pd.DataFrame(C,columns=['c1','c2','c3','c4']);classification.insert(0,'id',np.arange(1,m+1));classification['target']=target;classification.to_csv(data/'classification.csv',index=False,float_format='%.10f')
    t=np.arange(140);noise=rng.normal(scale=.6,size=140)
    for i in range(1,len(noise)):noise[i]+=(.35+.03*plan['number'])*noise[i-1]
    series=20+.02*plan['number']*t+noise
    if plan['period']>1:series+=2.5*np.sin(2*np.pi*t/plan['period'])
    pd.DataFrame(dict(date=pd.date_range('2024-01-01',periods=140),target=series)).to_csv(data/'timeseries.csv',index=False,float_format='%.10f')
