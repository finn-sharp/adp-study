"""
================================================================================
recent_exam_data.py - 후기 유형 연습용 합성 자료. 시드·행·필드·정답 규약을 고정한다.
================================================================================
설명:
    - 후기 유형 연습용 합성 자료. 시드·행·필드·정답 규약을 고정한다.
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
from pathlib import Path
import json
import numpy as np
import pandas as pd

def generate(root):
 for round_ in range(30,37):
  dest=Path(root)/(str(round_)+'회')/'data';dest.mkdir(parents=True,exist_ok=True)
  r=np.random.RandomState(20261000+round_)
  def save(name,d):
   if isinstance(d,dict):d=pd.DataFrame(d)
   d.to_csv(dest/(name+'.csv'),index=False,float_format='%.9g')
  def js(name,d): (dest/(name+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  n=240;z=r.normal(size=(n,6));ids=np.arange(1,n+1)
  if round_==30:
   age=r.randint(25,70,n);weight=68+10*z[:,0];waist=weight*.7+35+3*z[:,1]
   h=pd.DataFrame(dict(id=ids,age=age,weight=weight,waist=waist,pulse=70+8*z[:,2],DBP=40+.35*age+.2*weight+.12*waist+4*z[:,3]))
   h.loc[::23,'weight']=np.nan;save('health',h)
   weather=r.choice(['clear','rain'],n);road=r.choice(['urban','rural'],n);speed=r.uniform(20,100,n)
   pr=1/(1+np.exp(-(-4+.035*speed+.7*(weather=='rain'))))
   save('accident',dict(id=ids,date=pd.date_range('2023-01-01',periods=n).astype(str),weather=weather,road=road,speed=speed,severe=r.binomial(1,pr)))
   js('transport',dict(cost=[[4,8,6],[5,3,7],[6,5,2]],supply=[30,40,50],demand=[40,35,45]))
   a=r.choice(['young','middle','senior'],n);p=r.uniform(size=n)
   save('headset',dict(id=ids,age_group=a,preference=np.where(p<np.where(a=='young',.75,np.where(a=='middle',.5,.3)),'wireless','wired')))
  elif round_==31:
   height=170+8*z[:,0];weight=68+14*z[:,1];bmi=weight/(height/100)**2
   d=pd.DataFrame(dict(id=ids,height_cm=height,weight=weight,activity=r.randint(1,5,n),age=r.randint(18,65,n),body_class=np.where(bmi<21,0,np.where(bmi<26,1,2))))
   d.loc[::25,'activity']=np.nan;save('obesity',d)
   # 범주 특성이 체형과 연관되지만 실제 의료/생활 습관 인과 모형은 아니다.
   age=r.choice([16,17],n);gender=r.choice(['M','F'],n);ht=168+8*z[:,0]
   exercise=r.binomial(1,.5,n);sleep=r.binomial(1,.5,n);wt=(23+4*z[:,1]-2.4*exercise-1.4*sleep)*(ht/100)**2
   exam=pd.Timestamp('2024-07-01');dob=[exam-pd.Timedelta(days=int(a*365+day)) for a,day in zip(age,r.randint(0,365,n))]
   save('checkup',dict(id=ids,exam_date=str(exam.date()),birth_date=[str(x.date()) for x in dob],gender=gender,height_cm=ht,weight=wt,exercise=exercise,sleep=sleep))
   js('기준표',{'16_M':[18,25],'16_F':[17.5,24],'17_M':[18.5,25.5],'17_F':[18,24.5]})
   tv=r.uniform(0,250,n);radio=r.uniform(0,50,n);paper=r.uniform(0,100,n)
   save('ads',dict(TV=tv,radio=radio,paper=paper,sales=3+.05*tv+.18*radio+1.3*z[:,3]))
  elif round_==32:
   x1=z[:,0];x2=.95*x1+.15*z[:,1];x3=z[:,2];x4=z[:,3];x5=z[:,4]
   save('housing',dict(id=ids,x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,target=20+3*x1-2*x3+1.2*x4*x4+(1+.5*np.abs(x1))*z[:,5]))
   # 희소 셀이 있는 표를 개별 관측으로 펼친다.
   table=np.array([[9,1,0],[2,7,1],[0,1,4]]);rows=[]
   for i in range(3):
    for j in range(3):rows += [dict(cyl=[4,6,8][i],gear=[3,4,5][j])]*int(table[i,j])
   save('category',pd.DataFrame(rows));save('salary',dict(group=['A']*120+['B']*120,salary=np.r_[r.lognormal(5,.35,120),r.lognormal(5.2,.6,120)]))
  elif round_==33:
   c=r.choice([0,1,2,3],n,p=[.52,.22,.16,.1]);x=z.copy();x[:,0]+=c*.7;x[:,1]+=c*.4
   d=pd.DataFrame(x,columns=['x'+str(i+1) for i in range(6)]);d.insert(0,'id',ids);d['Category']=c;d.loc[::19,'x2']=np.nan;save('hepatitis',d)
   dates=pd.date_range('2021-01-01','2023-12-31',freq='14D');k=len(dates);temp=15+12*np.sin(np.arange(k)/26*2*np.pi);rain=r.exponential(3,k)
   users=300+5*temp-8*rain+20*(dates.dayofweek>=5)+r.normal(0,15,k)
   rows=[]
   for date,u in zip(dates,users):
    for b in ['on','off']:rows.append(dict(date=str(date.date()),board=b,h05=u*.15,h06=u*.15,h07=u*.2))
   save('subway',pd.DataFrame(rows));w=pd.DataFrame(dict(date=dates.astype(str),temp=temp,rain=rain));w.loc[::13,'rain']=np.nan;w=w.drop(index=[7,18]);save('weather',w)
   save('defect',dict(group=['A']*100+['B']*100,rate=np.r_[r.normal(4,.8,100),r.normal(4.35,1.3,100)]))
   save('counts',dict(count=r.poisson(3.2,400)))
   lines=r.choice(['L1','L2','L3','L4'],n);mins=r.exponential(8+np.array([int(x[1:]) for x in lines]),n)
   save('delay',dict(line=lines,minutes=mins))
   eps=r.normal(size=260);s=np.zeros(260)
   for t in range(2,260):s[t]=.65*s[t-1]-.2*s[t-2]+eps[t]+.25*eps[t-1]
   save('series',dict(value=s[40:]))
  elif round_==34:
   n=300;ids=np.arange(1,n+1);x=r.normal(size=(n,3));s=np.clip(np.floor(1+x+0.4*r.normal(size=(n,3))),0,2)
   ind=np.where(ids%5==0,2,np.where(ids%4==0,1,0));s[ind==1,0]=np.nan
   risk=-.7+.7*x[:,0]+.4*x[:,1]+.8*x[:,2];dead=r.binomial(1,1/(1+np.exp(-risk)))
   save('clinical',dict(id=ids,age=50+10*x[:,0],bp=120+15*x[:,1],signal=x[:,2],s1=s[:,0],s2=s[:,1],s3=s[:,2],ind=ind,dead=dead))
   dates=pd.date_range('2021-01-01','2023-12-31',freq='14D');k=len(dates);temp=15+10*np.sin(np.arange(k)*2*np.pi/26);wind=r.uniform(1,10,k)
   generation=50+4*wind+1.5*temp+r.normal(0,5,k);rows=[]
   for date,v in zip(dates,generation):
    for kind in ['wind','solar']:rows.append(dict(date=str(date.date()),kind=kind,p00=v*.2,p12=v*.3))
   save('power',pd.DataFrame(rows));save('weather_power',dict(date=dates.astype(str),temp=temp,wind=wind))
   nsub=48;gender=np.array(['F']*24+['M']*24);base=r.normal(0,5,nsub);rows=[]
   for i in range(nsub):
    for j,method in enumerate(['online','hybrid','traditional']):rows.append(dict(id=i,gender=gender[i],method=method,score=65+3*(gender[i]=='M')+base[i]+j*4+r.normal(0,2+2*j)))
   save('learning',pd.DataFrame(rows));study=r.uniform(0,8,240);fails=r.poisson(1,240);absent=r.poisson(3,240);support=r.binomial(1,.5,240);internet=r.binomial(1,.7,240)
   save('score',dict(study=study,failures=fails,absences=absent,support=support,internet=internet,score=50+2*study-4*fails-1.2*absent+3*support+2*internet+r.normal(0,12,240)))
  elif round_==35:
   stations=pd.DataFrame(dict(station_id=['S1','S2','S3'],lat=[35.16,35.19,35.12],lon=[126.85,126.9,126.92]));save('stations',stations)
   sensors=pd.DataFrame(dict(sensor_id=['W1','W2','W3'],lat=[35.15,35.2,35.13],lon=[126.84,126.91,126.93]));weather=[];rides=[]
   for date in pd.date_range('2023-01-01',periods=12):
    for _,st in stations.iterrows():rides.append(dict(date=str(date.date()),station_id=st.station_id,rides=int(r.randint(100,500))))
    for _,s in sensors.iterrows():weather.append(dict(date=str(date.date()),sensor_id=s.sensor_id,lat=s.lat,lon=s.lon,temp=float(r.normal(15,3))))
   save('rides',pd.DataFrame(rides));save('weather_geo',pd.DataFrame(weather))
  else:
   x=z[:,:4];target=2*x[:,0]-x[:,1]+r.normal(0,.6,n);label=r.binomial(1,1/(1+np.exp(-(-.2+.7*target+.5*x[:,2]))))
   d=pd.DataFrame(x,columns=['x1','x2','x3','x4']);d.insert(0,'id',ids);d['target']=target;d['label']=label;save('cascade',d)
   records=[]
   for day in range(1,7):records.append(dict(date='2024-01-%02d'%day,meta={'source':'simulated'},records=[dict(entity=e,sensors={'temp':float(r.normal(20,2)),'load':int(r.randint(1,10))}) for e in ['E1','E2','E3']]))
   js('records',records);f=r.normal(size=(240,2));load=np.array([[.8,.1],[.75,.1],[.9,.05],[.1,.85],[.05,.8],[.1,.9]])
   save('survey',pd.DataFrame(f@load.T+r.normal(0,.35,(240,6)),columns=['item'+str(i+1) for i in range(6)]))
