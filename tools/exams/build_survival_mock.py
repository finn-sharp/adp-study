"""생존분석 통계 확장 모의: 창작 지문·고정 합성 CSV·문제지·풀이 생성.

기존 22묶음의 복원 명세와 독립적으로 관리한다. Python 3.7 구문을
유지하지만 실행 환경은 별도 requirements 파일을 따른다.
"""
from pathlib import Path
import json
import sys
import textwrap
import numpy as np
import pandas as pd
import nbformat as nbf
from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / '기출문제/모의고사/생존분석_통계'
sys.path.insert(0, str(Path(__file__).parent))
import render_question_papers as paper

SECTIONS = [
    dict(title='1. 공급 장비의 보증기간 검토', data='supplier.csv', level='출제 후기 확인 유형',
         context='품질팀은 공급사 A·B 장비를 설치한 날부터 최초 고장 또는 관찰 종료까지 추적했다. 장비별 관측은 독립이며, 관찰 종료 시점은 고장시간과 독립이라고 가정한다. time은 개월, event=1은 최초 고장, event=0은 그때까지 고장이 관찰되지 않은 우측 중도절단이다. 모두 설치일부터 관찰했으며 경쟁사건·재발 고장은 없다. 같은 시점의 고장과 관찰 종료는 해당 시점 직전 위험집합에 함께 포함한다. 모든 소문항은 같은 자료를 사용한다.', questions=[
        ('1-1', '관측 단위·시작점·관심 사건을 정의하고 event=0의 의미를 설명하시오. 이 관측을 삭제하거나 정상 수명으로 취급한 평균 비교가 왜 부적절한지 설명하시오.', 3),
        ('1-2', '공급사 A의 각 관찰 시점 직전 위험집합 크기, 고장 수, 관찰 종료 수를 표로 만들고, 중도절단을 반영하는 비모수적 생존함수 추정 방법을 선택하시오. 12개월의 계산을 손계산과 코드로 대조하시오.', 4),
        ('1-3', '공급사별 25·30·35개월 생존확률과 해당 시점까지 최초 고장이 발생할 누적확률을 구하시오. 25개월을 넘겨 작동한 A 장비가 35개월도 넘겨 작동할 조건부 확률을 계산하고, 누적확률과 순간 위험률을 구분하시오.', 4),
        ('1-4', '공급사별 생존곡선·95% 점별 신뢰구간·중도절단 표시·위험집합 수를 제시하고 중앙 생존시간을 구하시오. 관찰 중 생존확률이 0.5 이하로 내려가지 않은 집단과 관찰 범위 밖의 60개월 수명을 어떻게 보고할지 설명하시오.', 4),
        ('1-5', '공급사에 따라 전체 생존분포가 다른지 유의수준 5%에서 판단하시오. 적절한 검정과 근거·가정·귀무가설·대립가설·통계량·자유도·p-value·기각 여부 및 보증정책 관점의 결론을 제시하시오. 작은 표본과 곡선 교차가 해석에 주는 한계를 설명하시오.', 5),
    ]),
    dict(title='2. 서비스 정책별 유지기간 평가', data='subscription.csv', level='응용 창작 유형',
         context='서비스팀은 가입 시점 정책 A·B·C에 따른 최초 해지까지의 기간을 비교한다. 독립 가입자이며 time은 개월, event=1은 최초 해지, event=0은 행정적 관찰 종료다. 관찰 종료는 해지시간과 독립이고 시작 시점의 지연 진입은 없다. 정책은 무작위 배정되지 않았다. 사전에 전체 차이 검정 후 필요 시 세 쌍을 모두 비교하고, 비교군 전체 오류율을 5%로 관리하기로 했다. 비교할 시점 24개월과 검정 절차는 자료 확인 전에 정했다.', questions=[
        ('2-1', '집단별 표본 수·사건 수·중도절단 비율·최대 관찰기간을 제시하고 생존곡선을 그리시오. 각 집단의 24개월 미해지 확률을 추정하되, 단순 잔존 고객 비율과 구분하시오.', 4),
        ('2-2', '세 정책의 전체 유지기간 분포가 같은지 평가할 방법을 선택하시오. 귀무·대립가설, 가정, 통계량·자유도·p-value·기각 여부와 업무 결론을 제시하시오. 전체 검정만으로 어느 두 정책이 다른지 알 수 있는지도 설명하시오.', 5),
        ('2-3', '앞서 정한 절차에 따라 후속 비교 필요 여부를 판단하시오. 필요하면 세 쌍의 원 p-value와 가족단위 오류율을 통제한 보정 p-value, 쌍별 가설·기각 여부·결론을 보고하시오. 가장 작은 p-value만 골라 보고하는 문제를 설명하시오.', 5),
        ('2-4', '사전에 정한 24개월까지 생존곡선 아래 면적을 계산하여 집단별 평균 미해지 유지기간을 요약하시오. 이 값과 중앙 생존시간의 차이를 설명하시오. 곡선 교차 시 전체 검정 하나와 단일 위험비만으로 정책을 추천하기 어려운 이유, 추가 검증과 인과 해석의 한계를 서술하시오. 이 소문항은 구간 추정이나 가설검정을 요구하지 않는다.', 6),
    ]),
    dict(title='3. 설비 상태를 반영한 고장 위험 평가', data='equipment.csv', level='확장 연습: 기출 확인 아님',
         context='운영팀은 장비별 최초 고장까지의 time(개월)과 event(1=고장, 0=행정적 종료)를 기록했다. temperature10은 (온도-60)/10, load10은 (부하율-70)/10, new_part는 신규 부품 사용=1·기존=0이다. 시작 시점에 측정한 세 변수를 모두 사용하고 식별자 id는 제외한다. 장비 관측은 독립, 지연 진입·경쟁사건은 없고, 공변량 조건부 중도절단은 비정보적이라고 가정한다. 연속 변수의 로그 위험에 대한 선형성은 작업 가정이다. 진단용 시간변환은 rank로 사전 지정하고 세 변수 진단의 가족단위 오류율을 5%로 관리한다.', questions=[
        ('3-1', '온도·부하를 보정하며 신규 부품과 고장 발생 위험의 관계를 설명할 회귀 방법을 선택하시오. 모형식·기준집단·계수 단위·핵심 가정과 사건 수를 제시하고 세 변수를 모두 넣어 적합하시오.', 4),
        ('3-2', '각 계수의 귀무가설·대립가설을 설정하고 추정계수·표준오차·위험비·위험비의 95% 신뢰구간·검정통계량·양측 p-value·기각 여부를 보고하시오. 계수별 검정은 개별 5% 탐색 결과로 표시하고 다중검정 한계를 밝히시오. 온도 10도 증가 및 신규 부품의 의미를 위험비와 확률비를 구분해 설명하시오.', 5),
        ('3-3', '온도 70도·부하율 80%인 장비에 대해 기존/신규 부품 사용 시 24개월 생존확률과 누적 고장확률을 예측하시오. 같은 온도·부하 조건의 상대적 위험과 24개월 고장확률비가 같은지 설명하시오. 표본 내 조건부 추정의 한계도 밝히시오.', 3),
        ('3-4', '각 변수의 상대적 위험이 시간에 따라 일정하다는 가정을 잔차 기반으로 점검하시오. 진단 가설·통계량·원 p-value·세 변수 보정 p-value·판정을 제시하고 잔차 그림을 함께 해석하시오. 기각하지 못한 경우의 표현과 위반 시 대안을 설명하시오.', 5),
        ('3-5', '신규 부품 도입에 대한 업무 결론을 효과 크기·불확실성·가정 진단과 연결해 작성하시오. 위험비가 1 미만인 것을 생존확률 증가율로 해석하는 오류, 관찰 자료에서 인과 효과를 단정하는 오류, 검정 결과만으로 부품 비용·안전을 결정하는 한계를 설명하시오.', 3),
    ]),
]

ANSWERS = {
'1-1': ('장비 한 대가 단위이고 설치일부터 최초 고장까지의 시간이 관심 변수다. event=0은 종료 시점까지의 무고장만 알려준다. 그 시점이 실제 수명이 아니므로 삭제하면 선택 편향, 고장 처리하면 하향 왜곡을 만들 수 있다. 독립적/비정보적 중도절단이 핵심 작업 가정이다.', '''
display(supplier.groupby('group').agg(n=('event','size'), failures=('event','sum')))
print('우측 중도절단: 실제 수명은 기록된 종료 시점보다 길다. event=0은 고장 아님.')
'''),
'1-2': ('Kaplan-Meier 곱한계 추정을 선택한다. 각 고장 시점에서 S(t)=이전 S×(1-d/n)이며 중도절단은 그 이후 위험집합만 줄인다. 12개월 직전 A의 n=15, d=1, c=1이고 S(12)=(15/16)×(14/15)=0.875다. 동시 고장·절단 순서를 지문의 규약과 일치시킨다.', '''
a = supplier[supplier.group=='A']
km_a = KaplanMeierFitter(label='A').fit(a.time, a.event)
display(km_a.event_table)
hand12 = (15/16)*(14/15)
assert np.isclose(km_a.predict(12), hand12)
print('A: n(12)=15, d(12)=1, c(12)=1, S(12)=',hand12)
'''),
'1-3': ('F(t)=1-S(t)는 t개월까지 고장이 발생할 누적확률이며 순간 위험률이 아니다. 조건부 확률은 S(35)/S(25)다. 사건 시점에서는 우연속 계단값을 사용하고 S(25)>0임을 확인한다.', '''
kms = {}
rows = []
for g, d in supplier.groupby('group'):
    km = KaplanMeierFitter(label=g).fit(d.time,d.event)
    kms[g] = km
    for t in [25,30,35]:
        s = float(km.predict(t)); rows.append(dict(group=g,month=t,S=s,F=1-s))
probabilities = pd.DataFrame(rows)
display(probabilities)
conditional = float(kms['A'].predict(35)/kms['A'].predict(25))
print('P(T>35 | T>25), A =',conditional)
record('1-3', probabilities=rows, conditional_A=conditional)
'''),
'1-4': ('그림은 우측 중도절단 표시와 Greenwood log-log 점별 신뢰구간, 시점별 위험집합을 포함한다. 곡선이 0.5 이하에 도달하지 않으면 중앙수명 미도달로 보고한다. 60개월은 두 집단 최대 관찰기간 밖이므로 마지막 계단값을 외삽 근거로 제시하지 않는다. 신뢰구간은 전체 곡선의 동시 신뢰대역이 아니다.', '''
fig, ax = plt.subplots(figsize=(7,4))
for km in kms.values(): km.plot_survival_function(ax=ax,show_censors=True,ci_show=True)
add_at_risk_counts(*kms.values(), ax=ax)
ax.set(xlabel='Months',ylabel='Survival probability',title='Supplier survival (pointwise 95% CI)')
plt.tight_layout(); plt.show()
medians = {g: (float(k.median_survival_time_) if np.isfinite(k.median_survival_time_) else None) for g,k in kms.items()}
print('Median (None = not reached):', medians)
print('60개월: 관찰 범위 밖으로 추정 보고하지 않는다.')
record('1-4', median_months=medians, outside_followup_60_months='not estimated')
'''),
'1-5': ('두 독립 집단의 우측 중도절단 생존분포 비교에 로그순위 검정을 선택한다. H0: 모든 t에서 S_A(t)=S_B(t), H1: 적어도 어떤 t에서 다르다. 독립성·비정보적 중도절단과 충분한 사건 수가 필요하다. 비례위험에서 검정력이 좋고 교차 시 차이를 놓칠 수 있으나 교차 자체가 영가설 하 크기를 자동으로 무효화하는 것은 아니다. 소표본의 점근 p-value를 신중히 해석한다.', '''
b = supplier[supplier.group=='B']
lr = logrank_test(a.time,b.time,event_observed_A=a.event,event_observed_B=b.event)
report('1-5','Log-rank','S_A(t)=S_B(t) for all t','different for some t',lr.test_statistic,lr.p_value,1,
       '공급사별 전체 생존분포가 다르다는 근거')
print('생존분포 차이는 원인 효과나 특정 보증기간의 경제적 최적성을 증명하지 않는다.')
'''),
'2-1': ('각 가입자의 추적 길이가 다르므로 단순 잔존 비율은 동일한 24개월 생존확률이 아니다. 집단별 Kaplan-Meier 추정과 위험집합을 사용한다.', '''
display(subscription.groupby('group').agg(n=('event','size'),events=('event','sum'),max_time=('time','max')))
display(subscription.groupby('group').event.apply(lambda x:1-x.mean()).rename('censor_fraction'))
sub_kms = {}
fig,ax = plt.subplots(figsize=(7,4))
for g,d in subscription.groupby('group'):
    k = KaplanMeierFitter(label=g).fit(d.time,d.event);sub_kms[g]=k
    k.plot_survival_function(ax=ax,show_censors=True)
ax.set(xlabel='Months',ylabel='Retention probability');plt.tight_layout();plt.show()
retention24={g:float(k.predict(24)) for g,k in sub_kms.items()}
print('S(24):',retention24);record('2-1',retention24=retention24)
'''),
'2-2': ('세 독립 집단의 전체 곡선 비교에 다집단 로그순위 검정을 사용한다. H0: 세 생존함수가 모두 같다. H1: 적어도 하나 다르다. 자유도는 2다. 전체 기각은 모든 쌍이 다름을 뜻하지 않는다. 정책은 비무작위이므로 연관성만 판단한다.', '''
overall = multivariate_logrank_test(subscription.time,subscription.group,subscription.event)
report('2-2','Multivariate log-rank','S_A=S_B=S_C for all t','at least one differs',
       overall.test_statistic,overall.p_value,2,'정책별 전체 유지기간 분포가 다르다는 근거')
'''),
'2-3': ('전체 검정이 기각되면 미리 정한 세 쌍 모두를 비교하고 Holm 방식으로 가족단위 오류율을 통제한다. 쌍별 H0는 두 생존함수 동일, H1은 일부 시점에서 차이. 보정 p-value로 판단하며, 기각하지 못함은 동일성의 증명이 아니다.', '''
if overall.p_value < ALPHA:
    pairs=[]
    for g,h in combinations(sorted(subscription.group.unique()),2):
        d=subscription[subscription.group==g];e=subscription[subscription.group==h]
        r=logrank_test(d.time,e.time,event_observed_A=d.event,event_observed_B=e.event)
        pairs.append(dict(pair=g+'-'+h,statistic=float(r.test_statistic),p_value=float(r.p_value)))
    adj=holm([x['p_value'] for x in pairs])
    for row,padj in zip(pairs,adj):
        report('2-3','Pairwise log-rank '+row['pair'],'S_left=S_right for all t','different for some t',
               row['statistic'],row['p_value'],1,'해당 두 정책의 생존분포가 다르다는 근거',padj)
    display(pd.DataFrame(pairs).assign(p_holm=adj,reject=np.array(adj)<ALPHA))
else:
    print('전체 검정 비기각: 사전 규칙에 따라 후속 비교하지 않는다.')
'''),
'2-4': ('24개월 제한평균생존시간(RMST)은 0~24개월 생존곡선 아래 면적으로 모든 집단에 같은 제한 시점을 적용한다. 중앙수명은 S(t)가 0.5 이하가 되는 최초 시점이다. 여기서 RMST 차이는 기술통계이며 유의성은 주장하지 않는다. 교차 곡선은 시간별 차이가 상쇄될 수 있어 사전 지정 RMST·시간구간별 효과 등 추가 분석을 고려하되 결과를 보고 유리한 검정을 선택하지 않는다.', '''
rmst={g:float(restricted_mean_survival_time(k,t=24)) for g,k in sub_kms.items()}
print('RMST(0,24), months:',rmst)
record('2-4',rmst_24=rmst,meaning='descriptive only; no test/CI requested')
print('비무작위 정책: 교란·비용·고객구성·후속 표본을 검토해야 한다.')
'''),
'3-1': ('Cox 비례위험 모형 h(t|x)=h0(t)exp(β1 temperature10+β2 load10+β3 new_part)을 선택한다. 기준은 온도60도·부하70%·기존 부품이다. temperature10·load10 한 단위는 각각 10도·10%p다. 모형은 비례위험·로그 위험 선형성·독립성·조건부 비정보적 중도절단을 가정한다. 단순 고장여부 분류는 관찰기간 차이를 반영하지 못한다.', '''
xcols=['temperature10','load10','new_part']
model_data=equipment[['time','event']+xcols].copy()
cph=CoxPHFitter().fit(model_data,duration_col='time',event_col='event')
print('장비 수:',len(model_data),'사건 수:',int(model_data.event.sum()))
display(cph.summary[['coef','se(coef)','exp(coef)','p']])
'''),
'3-2': ('변수별 H0: βj=0(HR=1), H1: βj≠0. Wald z=β/SE이며 양측 검정이다. HR=exp(β), 95% CI=exp(β±1.96SE)다. 다른 조건이 같은 순간의 위험비이지 누적 고장확률비 또는 생존확률 증가율이 아니다. 세 계수 p값은 개별 탐색 검정이며 가족단위 오류율 통제를 주장하지 않는다.', '''
coef=cph.summary[['coef','se(coef)','exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','z','p']]
display(coef)
for name,row in coef.iterrows():
    report('3-2','Cox Wald z: '+name,'beta_'+name+'=0','beta_'+name+' != 0',row['z'],row['p'],None,
           name+'과 조건부 고장 위험의 연관성이 있다는 근거')
record('3-2',coefficients=coef.reset_index().to_dict(orient='records'))
'''),
'3-3': ('temperature10=1, load10=1로 변환한다. Cox 누적 기준위험을 이용한 생존확률과 1-S(24)를 보고한다. 동일 조건의 위험비 exp(β_new)와 고정 시점의 누적 고장확률비는 일반적으로 다르다. 학습 자료에 적합한 조건부 추정이므로 외부 예측 정확도나 인과 개선율로 해석하지 않는다.', '''
profiles=pd.DataFrame({'temperature10':[1.,1.],'load10':[1.,1.],'new_part':[0,1]},index=['old','new'])
pred=cph.predict_survival_function(profiles,times=[24]).iloc[0]
display(pd.DataFrame({'S24':pred,'F24':1-pred}))
hr=float(np.exp(cph.params_['new_part']))
print('HR new/old:',hr,'F24 new/old:',float((1-pred['new'])/(1-pred['old'])))
record('3-3',S24=pred.to_dict(),HR=hr)
'''),
'3-4': ('Schoenfeld 잔차 기반 PH 진단의 H0는 해당 계수가 시간에 따라 일정함, H1은 시간에 따라 변함이다. 계수=0 검정과 다르다. rank 시간변환과 세 변수 Holm 보정을 사전에 정했다. 보정 후 비기각은 PH 가정이 증명됐다는 뜻이 아니다. 작은 표본의 검정력, 함수형태 오류도 검토한다. 위반 시 층화·시간상호작용·시간가변 계수 등 대안을 고려한다.', '''
ph=proportional_hazard_test(cph,model_data,time_transform='rank').summary
ph['p_holm']=holm(ph.p.to_list())
display(ph)
for name,row in ph.iterrows():
    report('3-4','Schoenfeld rank diagnostic: '+name,'beta_'+name+' constant over time',
           'beta_'+name+' varies with time',row.test_statistic,row.p,1,
           name+'의 비례위험 가정 위반을 시사하는 근거',row.p_holm)
res=cph.compute_residuals(model_data,kind='scaled_schoenfeld')
rank_time=model_data.loc[res.index,'time'].rank().to_numpy()
fig,axes=plt.subplots(1,3,figsize=(10,3))
for ax,name in zip(axes,xcols):
    ax.scatter(rank_time,res[name],s=8,alpha=.5)
    line=np.polyfit(rank_time,res[name].to_numpy(),1)
    xx=np.array([rank_time.min(),rank_time.max()]);ax.plot(xx,np.polyval(line,xx),color='black')
    ax.axhline(0,color='grey',lw=.6);ax.set(title=name,xlabel='Event-time rank')
plt.tight_layout();plt.show()
print('잔차 직선은 시각적 참고선이며 독립적인 확정 검정이 아니다. 비기각은 가정의 증명이 아니다.')
'''),
'3-5': ('실제 추정 HR·CI·계수 p와 PH 진단을 함께 확인해 조건부 연관성을 보고한다. 무작위 배정이 없는 합성 관찰 설계로 비용 절감이나 안전 개선의 인과 효과를 주장할 수 없다. 결측·비선형성·미측정 교란·외부 검증·예방정비에 따른 정보성 절단 가능성을 추가 점검한다.', '''
r=cph.summary.loc['new_part']
print('신규 부품 HR={:.4f}, 95% CI=({:.4f},{:.4f}), individual p={:.6g}'.format(
    r['exp(coef)'],r['exp(coef) lower 95%'],r['exp(coef) upper 95%'],r['p']))
print('위 수치는 합성 자료의 조건부 연관성이다. 부품 도입은 비용·안전·실험 설계·외부 검증을 추가해 판단한다.')
out=Path('실행결과');out.mkdir(exist_ok=True)
(out/'답안수치.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\\n',encoding='utf-8')
'''),
}

SETUP = '''
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import restricted_mean_survival_time
ALPHA=.05
results=[]
def record(question,**kwargs):
    results.append(dict(question=question,**kwargs))
def holm(p):
    p=np.asarray(p,dtype=float);order=np.argsort(p);s=p[order]
    out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*s))
    return out.tolist()
def report(question,method,H0,H1,stat,p,df,evidence,adjusted=None):
    used=float(p) if adjusted is None else float(adjusted)
    reject=used<ALPHA
    conclusion=evidence+('가 있다.' if reject else '가 충분하지 않다.')
    row=dict(method=method,H0=H0,H1=H1,statistic=float(stat),p_value=float(p),df=df,
             alpha=ALPHA,reject_H0=bool(reject),conclusion=conclusion)
    if adjusted is not None:row['p_holm']=float(adjusted)
    record(question,**row);display(pd.DataFrame([row]))
supplier=pd.read_csv('data/supplier.csv')
subscription=pd.read_csv('data/subscription.csv')
equipment=pd.read_csv('data/equipment.csv')
'''


def make_data():
    data=DEST/'data';data.mkdir(parents=True,exist_ok=True)
    ta=[8,12,12,16,20,20,24,25,28,30,32,35,36,40,42,45]
    ea=[1,1,0,1,1,0,1,0,1,1,0,1,0,1,0,0]
    tb=[10,15,15,18,22,25,25,28,30,32,35,36,38,40,42,45]
    eb=[0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0]
    pd.DataFrame(dict(id=range(1,33),group=['A']*16+['B']*16,time=ta+tb,event=ea+eb)).to_csv(data/'supplier.csv',index=False)
    rng=np.random.RandomState(20261019);rows=[]
    for g,scale in [('A',10),('B',18),('C',30)]:
        latent=rng.exponential(scale,size=80);end=rng.uniform(12,36,size=80)
        for t,c in zip(latent,end):rows.append(dict(group=g,time=min(t,c),event=int(t<=c)))
    d=pd.DataFrame(rows);d.insert(0,'id',np.arange(1,len(d)+1));d.to_csv(data/'subscription.csv',index=False,float_format='%.8f')
    n=360;x=rng.normal(size=n);load=rng.normal(size=n);part=rng.binomial(1,.5,size=n)
    eta=.55*x+.35*load-.7*part
    latent=-np.log(rng.uniform(size=n))/(.035*np.exp(eta));end=rng.uniform(24,48,size=n)
    pd.DataFrame(dict(id=np.arange(1,n+1),time=np.minimum(latent,end),event=(latent<=end).astype(int),
                      temperature10=x,load10=load,new_part=part)).to_csv(data/'equipment.csv',index=False,float_format='%.8f')


def render_pdf():
    paper.ST=paper.styles()
    story=[paper.p('분석전문가(ADP) 실기','title'),paper.p('생존분석 통계 모의 · 14문항 / 60점 / 권장 90분','subtitle'),
           paper.ruled_table([['성명','작성일','득점'],['','____년 __월 __일','       / 60']],[paper.WIDTH/3]*3),
           Spacer(1,12),paper.p('응시 안내','section'),
           paper.p('웹 출제 후기를 참고해 새로 작성한 학습용 모의문제입니다. 공식 기출 복원이나 실제 시험 배점이 아닙니다. 모든 CSV는 고정 합성 자료이며 폴더의 data 파일을 사용하십시오. 유의수준은 5%, 양측 검정을 기본으로 하며 문항에 정한 보정 절차를 따릅니다. 방법 선택·가정·가설·계산·판단·상황의 결론을 구분해 작성하십시오. 추정만 요구하는 문항에 p-value를 억지로 제시하지 마십시오.','small'),
           paper.p('자료 구성: 1번은 출제 후기에서 확인된 유형, 2번은 응용 창작, 3번은 확장 연습입니다. 세 대문항은 서로 다른 데이터셋입니다. 출처·유형 근거와 풀이 API는 동봉 README에서 확인하십시오.','small')]
    for i,sec in enumerate(SECTIONS):
        story += [PageBreak(),paper.section_heading(sec['title']),Spacer(1,12),
                  paper.p('제공 자료: data/'+sec['data']+' | '+sec['level'],'small'),paper.p(sec['context'],'body')]
        d=pd.read_csv(DEST/'data'/sec['data'])
        story += [paper.ruled_table([list(d.columns)]+d.head(3).round(3).astype(str).values.tolist(),[paper.WIDTH/len(d.columns)]*len(d.columns)),Spacer(1,15)]
        for qid,text,points in sec['questions']:
            block=paper.p(qid+'. ['+str(points)+'점] '+text,'question')
            story += [block]
    def furniture(c,doc):
        c.setFont('ExamGothic',8);c.setFillColor(colors.HexColor('#555555'))
        c.drawString(paper.MARGIN,810,'ADP 학습용 모의 · 생존분석 통계')
        c.drawRightString(595-paper.MARGIN,810,'창작 문제 / 공식 시험지 아님')
        c.line(paper.MARGIN,802,595-paper.MARGIN,802)
        c.line(paper.MARGIN,48,595-paper.MARGIN,48)
        c.drawCentredString(297.5,34,'- %d -'%doc.page)
    doc=SimpleDocTemplate(str(DEST/'생존분석_통계_문제지.pdf'),pagesize=A4,
                          leftMargin=paper.MARGIN,rightMargin=paper.MARGIN,topMargin=64,bottomMargin=66,
                          title='ADP 생존분석 통계 모의',author='ADP Study')
    doc.build(story,onFirstPage=furniture,onLaterPages=furniture)


def build():
    make_data()
    (DEST/'문항명세.json').write_text(json.dumps(dict(type='original_survival_mock',points=60,questions=14,
        sections=SECTIONS),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cells=[nbf.v4.new_markdown_cell('# 생존분석 통계 모의 풀이\n\n창작 합성 자료입니다. 출제 확인 유형과 확장 연습의 구분은 README를 참고하세요. 노트북의 현재 폴더에서 실행합니다. Python 3.12 실행 환경은 requirements-py312.txt에 고정합니다. 각 문항의 선택 근거·가설은 설명 셀, 실제 계산값·기각 여부는 바로 아래 실행 출력과 실행결과/답안수치.json에 있습니다.'),nbf.v4.new_code_cell(textwrap.dedent(SETUP))]
    for sec in SECTIONS:
        cells.append(nbf.v4.new_markdown_cell('## '+sec['title']+'\n\n'+sec['context']))
        for qid,text,points in sec['questions']:
            explanation,code=ANSWERS[qid]
            cells += [nbf.v4.new_markdown_cell('### '+qid+'\n\n**문제 ('+str(points)+'점)** '+text+'\n\n**선택 근거·가정·해석**\n\n'+explanation),nbf.v4.new_code_cell(textwrap.dedent(code))]
    book=nbf.v4.new_notebook(cells=cells,metadata=dict(adp=dict(type='survival_supplement',subquestions=[q[0] for s in SECTIONS for q in s['questions']]),kernelspec=dict(name='python3',display_name='Python 3',language='python')))
    # 독립 보충 묶음: 기존 22묶음 전수 검사의 *풀이.ipynb 대상과 구분한다.
    nbf.write(book,DEST/'생존분석_통계_해설.ipynb')
    render_pdf()
    print('Created:',DEST)


if __name__=='__main__':
    build()
