"""
================================================================================
audit_recent_exams.py - 회차별 지문·실제 실행·가설·수치와 핵심 풀이를 독립 대조한다.
================================================================================
설명:
    - 회차별 지문·실제 실행·가설·수치와 핵심 풀이를 독립 대조한다.
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
import sys
import json
import ast
import hashlib
import re
import math
import numpy as np
import pandas as pd
import nbformat
import fitz
from scipy import stats
from scipy.special import gammaln
import pingouin as pg
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'기출문제/유형연습'
sys.path.insert(0,str(ROOT/'공통'))
from analysis import mixed_table,haversine


def exact_table_tail(obs):
 row=obs.sum(1);col=obs.sum(0);exp=np.outer(row,col)/obs.sum();stat=((obs-exp)**2/exp).sum();tail=0.;total=0.
 const=gammaln(row+1).sum()+gammaln(col+1).sum()-gammaln(obs.sum()+1)
 for a in range(min(row[0],col[0])+1):
  for b in range(min(row[0]-a,col[1])+1):
   c=row[0]-a-b
   if c>col[2]:continue
   for d in range(min(row[1],col[0]-a)+1):
    for e in range(min(row[1]-d,col[1]-b)+1):
     f=row[1]-d-e;third=col-np.array([a+d,b+e,c+f])
     if f<0 or (third<0).any() or third.sum()!=row[2]:continue
     table=np.array([[a,b,c],[d,e,f],third]);prob=np.exp(const-gammaln(table+1).sum());total+=prob
     if ((table-exp)**2/exp).sum()>=stat-1e-10:tail+=prob
 assert np.isclose(total,1,atol=1e-10)
 return tail


def audit():
 rows=[];all_answers={};hashes={}
 compact=lambda t:re.sub(r'\s+','',t)
 for number in range(30,37):
  folder=ROOT/(str(number)+'회');manifest=json.loads((folder/'문항명세.json').read_text());qs=[q for s in manifest['sections'] for q in s['questions']];ids=[q['id'] for q in qs]
  assert len(ids)==len(set(ids))==manifest['questions']
  book=nbformat.read(next(folder.glob('*해설.ipynb')),4);nbformat.validate(book)
  assert book.metadata.adp.subquestions==ids
  codes=[x for x in book.cells if x.cell_type=='code'];assert len(codes)==len(ids)+1
  for cell in codes:
   ast.parse(cell.source,feature_version=(3,7));assert cell.execution_count is not None
   assert not any(o.output_type=='error' for o in cell.outputs)
   # 검정 실패/수렴 실패를 오류 없는 노트북으로 오인하지 않는다.
   for o in cell.outputs:
    if o.output_type=='stream':assert not re.search('FitFailedWarning|ConvergenceWarning|All the .* fits failed',o.text)
  pdfpath=next(folder.glob('*문제지.pdf'));pdf=fitz.open(pdfpath);pdftext=compact(''.join(p.get_text() for p in pdf));md='\n'.join(c.source for c in book.cells if c.cell_type=='markdown')
  for s in manifest['sections']:
   assert compact(s['context']) in pdftext
   for q in s['questions']:assert compact(q['text']) in pdftext and q['text'] in md,q['id']
  for p in pdf:
   assert abs(p.rect.width-595.276)<1 and abs(p.rect.height-841.89)<1
   for x0,y0,x1,y1,*_ in p.get_text('words'):assert x0>=40 and x1<=558 and y0>=20 and y1<=822,(number,x0,y0,x1,y1)
  answers=json.loads((folder/'실행결과/답안수치.json').read_text());assert [a['question'] for a in answers]==ids
  all_answers[number]={a['task']:a for a in answers}
  tests=0
  for answer in answers:
   assert answer['outputs'] or answer['tests'],answer
   for t in answer['tests']:
    tests+=1;assert t['H0'] and t['H1'] and t['method'] and t['conclusion']
    p=t.get('adjusted_p_value',t['p_value']);assert 0<=p<=1 and np.isfinite(t['statistic']);assert t['reject_H0']==(p<t['alpha'])
   for label,data in answer['outputs'].items():
    if label in ['split IDs','stage1 split','dead holdout','cascade holdout']:
     train=data.get('train_index',data.get('training_ids',data.get('train_ids')));test=data.get('test_index',data.get('test_ids'))
     if train is not None:assert set(train).isdisjoint(test)
  rows.append(dict(round=number,questions=len(ids),code_cells=len(codes),hypothesis_reports=tests,pdf_pages=len(pdf),figures=sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs),status='passed'))
  for p in [pdfpath,next(folder.glob('*해설.ipynb')),folder/'분석코드.py',folder/'실행결과/답안수치.json']+sorted((folder/'data').glob('*')):
   hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 for p in [ROOT/'공통/analysis.py']+list((REPO/'tools/exams').glob('*recent_exam*.py')):
  ast.parse(p.read_text(),feature_version=(3,7))
 # LP의 원/쌍대 실행값으로 최적성 인증을 검증한다.
 ans=all_answers[30]['transport']['outputs']['plan'];spec=json.loads((ROOT/'30회/data/transport.json').read_text());C=np.array(spec['cost']);x=np.array(ans['amount']);dual=np.array(ans['dual_variables']);m,n=C.shape
 A=np.r_[np.kron(np.eye(m),np.ones((1,n))),np.tile(np.eye(n),(1,m))];b=np.r_[spec['supply'],spec['demand']]
 assert np.all(x>=-1e-10) and np.allclose(A@x.ravel(),b)
 assert np.all(A.T@dual<=C.ravel()+1e-8) and np.isclose((C*x).sum(),b@dual) and abs(ans['primal_dual_gap'])<1e-7
 # 중첩 확률의 독립 조합 계산.
 q=sum(math.comb(6,k) for k in range(4,7))/2**6;p=sum(math.comb(5,k)*q**k*(1-q)**(5-k) for k in range(3,6));r=all_answers[30]['binomial']['outputs']['probability'];assert np.isclose(r['P_at_least_3_families'],p) and np.isclose(r['expected_families'],5*q)
 # 정규-정규 사후 모수 수식 대조.
 r=all_answers[31]['bayes']['outputs']['posterior'];assert np.isclose(r['mean'],(144*90+225*108)/(144+225)) and np.isclose(r['variance'],144*225/(144+225))
 # RxC 전체 표를 작은 표본에서 완전열거하여 Monte Carlo p를 대조.
 d=pd.read_csv(ROOT/'32회/data/category.csv');obs=pd.crosstab(d.cyl,d.gear).to_numpy();exact=exact_table_tail(obs);mc=all_answers[32]['category_mc'];p=mc['tests'][0]['p_value'];se=mc['outputs']['Monte Carlo detail']['MC_SE'];assert abs(p-exact)<=4*se+.0002,(p,exact,se)
 # 33회 적합도: 모수 추정 df와 전체 기대빈도.
 g=all_answers[33]['poisson_gof'];info=g['outputs']['pooled bins'];bins=info['bins'];exp=np.array([x[3] for x in bins]);ob=np.array([x[2] for x in bins]);ref=((ob-exp)**2/exp).sum();t=g['tests'][0]
 assert t['df']==len(bins)-2 and np.all(exp>=5) and np.isclose(t['p_value'],stats.chi2.sf(ref,len(bins)-2))
 # 혼합 ANOVA: 수작업 오차항의 F를 별도 패키지에 대조.
 learning=pd.read_csv(ROOT/'34회/data/learning.csv');manual,eps,_=mixed_table(learning);reference=pg.mixed_anova(data=learning,dv='score',within='method',between='gender',subject='id',correction=True)
 for item in manual:
  label='Interaction' if item['effect']=='interaction' else item['effect'];rf=reference.loc[reference.Source==label,'F'].iloc[0];assert np.isclose(item['F'],rf,rtol=1e-7),(item,rf)
 assert .5-1e-10<=eps<=1+1e-10
 shifted=learning.copy();shifted['score']+=1000;again,eps2,_=mixed_table(shifted);assert np.allclose([x['F'] for x in manual],[x['F'] for x in again],rtol=1e-7) and np.isclose(eps,eps2)
 # 구면 거리: 경도 1도 적도 거리를 닫힌식으로 대조.
 assert np.isclose(haversine(0,0,0,1),6371*np.pi/180) and np.isclose(haversine(0,1,0,0),haversine(0,0,0,1))
 assert sum(r['questions'] for r in rows)==53
 result=dict(status='passed',bundles=7,questions=53,notebooks=rows,pdf_pages=sum(r['pdf_pages'] for r in rows),hypothesis_reports=sum(r['hypothesis_reports'] for r in rows),figures=sum(r['figures'] for r in rows),
             independent_checks=dict(LP_primal_dual=True,binomial_enumeration=True,normal_posterior=True,RxC_exact_tail=exact,RxC_MC=p,mixed_ANOVA_reference=True,mixed_shift_invariance=True,haversine_closed_form=True),python=sys.version.split()[0],sha256=hashes)
 (ROOT/'검증결과.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False,indent=2))
if __name__=='__main__':audit()
