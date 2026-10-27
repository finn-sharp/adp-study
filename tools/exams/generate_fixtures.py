"""
================================================================================
generate_fixtures.py - 복원 연습용 합성 CSV/TXT 생성. 모든 자료는 원 시험 자료가 아닙니다.
================================================================================
설명:
    - 복원 연습용 합성 CSV/TXT 생성. 모든 자료는 원 시험 자료가 아닙니다.
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

ROOT = Path(__file__).resolve().parents[2] / '기출문제'
rng = np.random.RandomState(20261014)
MANIFEST = []

def save(folder, name, data):
    path = ROOT / folder / 'data' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.DataFrame): data.to_csv(path, index=False)
    else: path.write_text(data, encoding='utf-8')
    MANIFEST.append({'path': str(path.relative_to(ROOT)), 'kind': 'synthetic', 'seed': 20261014,
                     'rows': len(data) if isinstance(data, pd.DataFrame) else None,
                     'note': '사용자 자료의 컬럼/주제에 맞춰 만든 연습 자료. 원 시험 수치/정답이 아님.'})

def text_data(folder, name, custom='태교여행'):
    words=['영화','가족','여행','호텔','공구','배송','맛','학교','행복',custom]
    adjective=['좋다','편하다','빠르다','즐겁다']
    texts=[' '.join(rng.choice(words,6))+' '+rng.choice(adjective)+' 123 !@' for _ in range(60)]
    save(folder,name,pd.DataFrame({'date':pd.date_range('2026-01-15',periods=60).strftime('%Y-%m-%d'),'text':texts}))
    save(folder,'lexicon.csv',pd.DataFrame({'word':words+adjective,'pos':['Noun']*len(words)+['Adjective']*len(adjective)}))

def retail(n=750):
    group=rng.choice(3,n);customers=rng.choice(np.arange(1200,1260),n)
    return pd.DataFrame({'InvoiceNo':np.arange(n),'CustomerID':customers,'StockCode':['P'+str(g) for g in group],
     'StockCategory':group,'Date':pd.date_range('2025-01-01',periods=n,freq='h').strftime('%Y-%m-%d %H:%M:%S'),
     'Quantity':rng.poisson(2+group*3,n)+1,'UnitPrice':np.round(rng.uniform(2,20,n)*(group+1),2),
     'Country':rng.choice(['KR','JP','US'],n)})

def series(folder,name='series.csv',n=144,period=12):
    x=np.arange(n);v=80+.15*x+12*np.sin(2*np.pi*x/period)+rng.normal(0,2,n);v[min(35,n//3)]=np.nan
    save(folder,name,pd.DataFrame({'date':pd.date_range('2014-01-01',periods=n,freq='MS'),'value':v}))

# 모의 1: 번호 개수가 제한된 합성 회차로 Apriori 부담을 제어, 6개 중복 없는 숫자.
f='모의고사/1회';n=400
save(f,'lotto.csv',pd.DataFrame([rng.choice(np.arange(1,13),6,replace=False) for _ in range(n)],columns=['n'+str(i) for i in range(1,7)]))
n=320;pos=rng.choice(['ST','CM','CB','GK','LW','CDM','LB','RB'],n);age=rng.randint(18,36,n);overall=rng.randint(60,90,n);wage=rng.uniform(1,30,n);height=[str(rng.randint(5,7))+"'"+str(rng.randint(0,12)) for _ in range(n)]
value=500+overall*30+wage*150-age*10+np.where(pos=='GK',-1000,0)+rng.normal(0,500,n)
save(f,'FIFA.csv',pd.DataFrame({'Position':pos,'Preferred_Foot':rng.choice(['Left','Right'],n),'Age':age,'Overall':overall,'Wage':wage,'Height':height,'Weight_lb':rng.normal(160,12,n),'Value':value}))
text_data(f,'reviews.csv','기생충')
# 모의 2
f='모의고사/2회';n=240;cgpa=rng.uniform(6,10,n);gre=250+cgpa*8+rng.normal(0,10,n)
save(f,'Admission.csv',pd.DataFrame({'GRE':gre,'TOEFL':80+cgpa*3+rng.normal(0,5,n),'Univ_Rating':rng.randint(1,6,n),'SOP':rng.randint(1,6,n),'LOR':rng.randint(1,6,n),'CGPA':cgpa,'Research':rng.randint(0,2,n),'Chance_of_Admit':np.clip(.08*cgpa+rng.normal(0,.05,n),0,1)}))
n=420;sex=rng.choice(['male','female'],n);pclass=rng.randint(1,4,n);prob=np.where(sex=='female',.7,.25)-.05*(pclass-1);age=rng.uniform(1,85,n);age[rng.choice(n,25,False)]=np.nan
save(f,'titanic.csv',pd.DataFrame({'pclass':pclass,'sex':sex,'sibsp':rng.poisson(.5,n),'parch':rng.poisson(.4,n),'fare':rng.exponential(30,n),'embarked':rng.choice(['S','C','Q',''],n),'age':age,'cabin':rng.choice(['A1','B2',''],n),'survived':(rng.rand(n)<prob).astype(int)}));text_data(f,'speech.csv','민주주의')
# 모의 3
f='모의고사/3회';n=300;income=rng.uniform(20,100,n);price=rng.uniform(60,150,n);ads=rng.uniform(0,20,n)
save(f,'Carseats.csv',pd.DataFrame({'Urban':rng.choice(['Yes','No'],n),'Income':income,'Price':price,'Advertising':ads,'CompPrice':rng.uniform(60,150,n),'Population':rng.uniform(50,500,n),'Age':rng.uniform(20,70,n),'Education':rng.randint(10,19,n),'Sales':10+.04*income-.04*price+.12*ads+rng.normal(0,.6,n)}))
n=600;pc1=rng.choice([1,5,9],n);pc2=rng.choice([1,2,np.nan],n);pc3=rng.choice([1,2,np.nan],n)
save(f,'BlackFriday.csv',pd.DataFrame({'User_ID':rng.randint(1,100,n),'Product_ID':['P'+str(i) for i in rng.randint(1,30,n)],'Occupation':rng.randint(0,10,n),'Marital_Status':rng.randint(0,2,n),'Product_Category_1':pc1,'Product_Category_2':pc2,'Product_Category_3':pc3,'Gender':rng.choice(['M','F'],n),'Age':rng.choice(['18-25','26-35','36-45'],n),'City_Category':rng.choice(['A','B','C'],n),'Stay_In_Current_City_Years':rng.choice(['0','1','2','4+'],n),'Purchase':pc1*500+rng.normal(0,200,n)}));text_data(f,'comments.csv','공구')
# 모의 4: 절대 NA>=10000 조건을 실제로 실행할 수 있게 10520행.
f='모의고사/4회';n=10520;rain=rng.exponential(3,n);hum=rng.uniform(30,100,n);cloud=rng.uniform(0,9,n);cloud[:10010]=np.nan
save(f,'weatherAUS.csv',pd.DataFrame({'Date':pd.date_range('1995-01-01',periods=n),'Location':rng.choice(['A','B','C'],n),'Rainfall':rain,'Humidity':hum,'Cloud9am':cloud,'RainTomorrow':np.where(rng.rand(n)<(hum/150+rain/30).clip(0,1),'Yes','No')}))
n=172;g=rng.uniform(0,100,n);fb=rng.uniform(0,60,n);tw=rng.uniform(0,30,n);emp=rng.randint(2,50,n)
save(f,'bike_marketing.csv',pd.DataFrame({'pop_density':rng.choice(['low','medium','high'],n),'google_adwords':g,'facebook':fb,'twitter':tw,'marketing_total':g+fb+tw,'employees':emp,'revenues':300+g*4+fb*2+emp*8+rng.normal(0,70,n)}));text_data(f,'instagram.csv','태교여행')
# 추가 회차
f='18회';n=180;z=rng.choice(3,n)
save(f,'segments.csv',pd.DataFrame({'age':rng.normal(30+z*10,2,n),'income':rng.normal(100+z*80,10,n),'spend':rng.normal(20+z*30,4,n)}));text_data(f,'documents.csv','고객');series(f)
f='19회';n=450;age=rng.randint(20,70,n);income=rng.normal(50,10,n)
save(f,'credit.csv',pd.DataFrame({'gender':rng.choice(['M','F'],n),'age':age,'credit_grade':rng.randint(1,8,n),'income':income,'churn':(rng.rand(n)<1/(1+np.exp(-(.04*(age-40)-.04*(income-50))))).astype(int)}));series(f,'traffic.csv',n=240)
f='21회';n=395;absence=rng.poisson(5,n);health=rng.randint(1,6,n)
save(f,'student.csv',pd.DataFrame({'school':rng.choice(['GP','MS'],n),'sex':rng.choice(['F','M'],n),'paid':rng.choice(['yes','no'],n),'activities':rng.choice(['yes','no'],n),'famrel':rng.randint(1,6,n),'freetime':rng.randint(1,6,n),'goout':rng.randint(1,6,n),'dalc':rng.randint(1,6,n),'walc':rng.randint(1,6,n),'health':health,'absences':absence,'grade':np.clip(16-.6*absence+.4*health+rng.normal(0,2,n),0,20)}))
X=rng.normal(size=(120,10));y=X[:,0]*4+X[:,1]*2+rng.normal(size=120)
save(f,'regression.csv',pd.DataFrame(np.column_stack([X,y]),columns=['x'+str(i) for i in range(1,11)]+['y']))
x=np.linspace(-3,3,100);save(f,'polynomial.csv',pd.DataFrame({'x':x,'y':2+x+x*x+rng.normal(0,.8,100)}));save(f,'anova.csv',pd.DataFrame({'group':np.repeat(list('abcd'),25),'x':rng.normal(size=100),'y':rng.normal(size=100)+np.repeat([0,1,2,3],25)}))
f='24회';n=395;age=rng.randint(15,22,n);study=rng.randint(1,5,n)
save(f,'student_absence.csv',pd.DataFrame({'sex':rng.choice(['F','M'],n),'age':age,'parents_separated':rng.randint(0,2,n),'mother_education':rng.randint(0,5,n),'father_education':rng.randint(0,5,n),'guardian':rng.choice(['mother','father','other'],n),'reason':rng.choice(['near','course'],n),'travel_time':rng.randint(1,5,n),'study_time':study,'failure':rng.randint(0,4,n),'family_relation':rng.randint(1,6,n),'leisure':rng.randint(1,6,n),'absences':np.maximum(0,12-study*2+rng.normal(0,3,n))}))
n=10;ad=rng.uniform(5,20,n);rd=rng.uniform(2,10,n);save(f,'advertising.csv',pd.DataFrame({'advertising':ad,'rd':rd,'sales':10+ad*3+rd*2+rng.normal(0,1,n)}));save(f,'sample.csv',pd.DataFrame({'value':[9.8,10.2,9.9,10.5,10.1,10.3,9.7,10.0,10.4]}))
f='25회';save(f,'retail.csv',retail());series(f);n=20;before=rng.normal(135,8,n);save(f,'paired_bp.csv',pd.DataFrame({'before':before,'after':before-rng.normal(4,4,n)}));save(f,'production.csv',pd.DataFrame({'factory':np.repeat(list('XYZ'),10),'output':rng.normal(size=30)+np.repeat([0,1,2],10)}));save(f,'npv.csv',pd.DataFrame({'year':[1,2,3],'plan1':[10,20,15],'plan2':[15,14,19],'plan3':[12,11,30],'plan4':[13,25,20],'plan5':[16,30,24]}))
f='26회';df=retail();df.loc[0,'Quantity']=np.nan;df.loc[1,'Quantity']=900;df.loc[2,'Country']=np.nan;save(f,'retail.csv',df);save(f,'regional_support.csv',pd.DataFrame({'region':list('ABC'),'yes':[100,200,300],'no':[150,130,180]}));save(f,'blood_pressure.csv',pd.DataFrame({'sex':['M']*16+['F']*9,'value':np.r_[rng.normal(110,8,16),rng.normal(101,7,9)]}));series(f,'silver.csv',n=9);n=140;weight=rng.normal(70,12,n);height=rng.normal(170,10,n);save(f,'body.csv',pd.DataFrame({'Weight':weight,'Height':height,'Waist':.7*weight-.1*height+50+rng.normal(0,4,n)}))
f='28회';n=420;age=rng.randint(10,19,n);grade=rng.choice([1,2,3,4,5],n,p=[.52,.22,.13,.09,.04]);save(f,'student.csv',pd.DataFrame({'age':age,'ordinal':rng.randint(1,5,n),'guardian':rng.choice(['mother','father','other'],n),'constant':'A','score':grade+rng.normal(0,1,n),'target':grade}));save(f,'survival.csv',pd.DataFrame({'time':rng.randint(1,25,36),'event':rng.binomial(1,.65,36)}));save(f,'school_scores.csv',pd.DataFrame({'school':['A']*12+['B']*24,'score':np.r_[rng.normal(70,5,12),rng.gamma(8,9,24)]}));save(f,'tasting.csv',pd.DataFrame({'before':rng.binomial(1,.4,60),'after':rng.binomial(1,.6,60)}));n=100;z=rng.normal(size=n);save(f,'partial.csv',pd.DataFrame({'x':z+rng.normal(0,.5,n),'y':z+rng.normal(0,.5,n),'z':z}))
(ROOT/'복원연습_데이터명세.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2)+'\n')
print('synthetic fixtures:',len(MANIFEST))
