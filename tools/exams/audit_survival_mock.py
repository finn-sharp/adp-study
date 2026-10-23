"""생존분석 확장 모의의 수치·PDF·노트북 대응을 독립 계산으로 검증한다."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import nbformat
import fitz

REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'기출문제/모의고사/생존분석_통계'


def km_value(d,t):
    """사건 시점 직전 위험집합에서 직접 곱한계 계산."""
    value=1.
    for time in sorted(d.loc[(d.event==1)&(d.time<=t),'time'].unique()):
        n=int((d.time>=time).sum());events=int(((d.time==time)&(d.event==1)).sum())
        value*=1-events/n
    return value


def manual_logrank(d):
    """각 시점 다변량 초기하 공분산으로 U와 V를 누적한다."""
    groups=sorted(d.group.unique());k=len(groups);u=np.zeros(k);v=np.zeros((k,k))
    for time in sorted(d.loc[d.event==1,'time'].unique()):
        risk=np.array([((d.group==g)&(d.time>=time)).sum() for g in groups],float)
        events=np.array([((d.group==g)&(d.time==time)&(d.event==1)).sum() for g in groups],float)
        n=risk.sum();fail=events.sum();p=risk/n;u+=events-fail*p
        if n>1:v+=fail*(n-fail)/(n-1)*(np.diag(p)-np.outer(p,p))
    stat=float(u[:-1]@np.linalg.pinv(v[:-1,:-1])@u[:-1])
    return stat,float(chi2.sf(stat,k-1))


def audit():
    manifest=json.loads((DEST/'문항명세.json').read_text())
    qids=[q[0] for s in manifest['sections'] for q in s['questions']]
    assert len(qids)==len(set(qids))==14
    assert sum(q[2] for s in manifest['sections'] for q in s['questions'])==60
    results=json.loads((DEST/'실행결과/답안수치.json').read_text())
    book=nbformat.read(DEST/'생존분석_통계_해설.ipynb',4);nbformat.validate(book)
    assert book.metadata.adp.subquestions==qids
    assert len([c for c in book.cells if c.cell_type=='code'])==15
    for c in book.cells:
        if c.cell_type=='code':
            ast.parse(c.source,feature_version=(3,7))
            assert c.execution_count is not None
            assert not any(o.output_type=='error' for o in c.outputs)
    ast.parse((REPO/'tools/exams/build_survival_mock.py').read_text(),feature_version=(3,7))
    compact=lambda x:re.sub(r'\s+','',x)
    pdf=fitz.open(DEST/'생존분석_통계_문제지.pdf')
    alltext=compact(''.join(page.get_text() for page in pdf))
    markdown='\n'.join(c.source for c in book.cells if c.cell_type=='markdown')
    for s in manifest['sections']:
        assert compact(s['context']) in alltext
        for qid,text,_ in s['questions']:
            assert compact(text) in alltext and text in markdown,qid
            assert not re.search('Kaplan|Cox|log.?rank|로그순위|Schoenfeld',text,re.I),qid
    for page in pdf:
        assert abs(page.rect.width-595.276)<1 and abs(page.rect.height-841.89)<1
        for x0,y0,x1,y1,*_ in page.get_text('words'):
            assert x0>=40 and x1<=558 and y0>=20 and y1<=822,(x0,y0,x1,y1)
    datasets={}
    for name,n in [('supplier',32),('subscription',240),('equipment',360)]:
        d=pd.read_csv(DEST/'data'/(name+'.csv'));datasets[name]=d
        assert len(d)==n and d.id.is_unique and not d.isna().any().any()
        assert (d.time>0).all() and set(d.event.unique())=={0,1}
    byq=lambda q:[r for r in results if r['question']==q]
    for r in results:
        if 'p_value' in r:
            used=r.get('p_holm',r['p_value'])
            assert np.isfinite(r['statistic']) and 0<=r['p_value']<=1
            assert r['reject_H0']==(used<.05) and r['H0'] and r['H1'] and r['conclusion']
    supplier=datasets['supplier']
    assert km_value(supplier[supplier.group=='A'],12)==.875
    for r in byq('1-3')[0]['probabilities']:
        ref=km_value(supplier[supplier.group==r['group']],r['month'])
        assert np.isclose(r['S'],ref) and np.isclose(r['S']+r['F'],1)
    assert byq('1-4')[0]['median_months']=={'A':30.,'B':None}
    for d,r in [(supplier,byq('1-5')[0]),(datasets['subscription'],byq('2-2')[0])]:
        stat,p=manual_logrank(d)
        assert np.isclose(r['statistic'],stat,rtol=1e-8) and np.isclose(r['p_value'],p,atol=1e-15,rtol=1e-8)
    for r in byq('2-3'):
        left,right=r['method'].split()[-1].split('-')
        d=datasets['subscription'];d=d[d.group.isin([left,right])]
        stat,p=manual_logrank(d)
        assert np.isclose(r['statistic'],stat,rtol=1e-8) and np.isclose(r['p_value'],p,rtol=1e-8)
    for g,value in byq('2-4')[0]['rmst_24'].items():
        d=datasets['subscription'];d=d[d.group==g]
        times=[0.]+sorted(d.loc[(d.event==1)&(d.time<24),'time'].unique())+[24.]
        ref=sum((right-left)*km_value(d,left) for left,right in zip(times[:-1],times[1:]))
        assert np.isclose(value,ref)
    coef=byq('3-2')[-1]['coefficients']
    for row in coef:
        assert row['p']>0 # serialization must not round tiny p-values to zero
        assert np.isclose(row['exp(coef)'],np.exp(row['coef']))
        assert np.isclose(row['p'],2*norm.sf(abs(row['z'])),rtol=1e-8)
    # 연속 사건시간(동률 없음)의 부분우도 점수·정보행렬을 독립 계산한다.
    d=datasets['equipment'];cols=[r['covariate'] for r in coef]
    beta=np.array([r['coef'] for r in coef]);x=d[cols].to_numpy();w=np.exp(x@beta)
    assert d.loc[d.event==1,'time'].is_unique
    score=np.zeros(3);info=np.zeros((3,3))
    for i in d.index[d.event==1]:
        mask=(d.time>=d.loc[i,'time']).to_numpy();z=x[mask];weights=w[mask];weights=weights/weights.sum()
        mean=(weights[:,None]*z).sum(0)
        score+=x[i]-mean;info+=(z.T*weights)@z-np.outer(mean,mean)
    assert np.max(np.abs(score))<.001
    assert np.allclose(np.sqrt(np.diag(np.linalg.inv(info))),[r['se(coef)'] for r in coef],rtol=1e-5)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DEST/'생존분석_통계_해설.ipynb',DEST/'생존분석_통계_문제지.pdf']}
    for p in (DEST/'data').glob('*.csv'):files['data/'+p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(status='passed',questions=14,points=60,code_cells=15,pdf_pages=len(pdf),
                hypothesis_reports=sum('p_value' in r for r in results),
                independent_checks=['KM risk sets/ties','two-/three-group log-rank covariance',
                                    'RMST step integral','Cox partial-likelihood score/information','Wald tail probability'],
                python=sys.version.split()[0],sha256=files)
    (DEST/'검증결과.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    audit()
