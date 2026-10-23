"""생산 모의의 문항/PDF/실행 출력·정수 전수·추론 수치를 독립 대조."""
from pathlib import Path
from itertools import product
import ast,json,sys,hashlib,re,subprocess,tempfile,os
import numpy as np
import pandas as pd
from scipy import stats
import nbformat,fitz
REPO=Path(__file__).resolve().parents[2];DEST=REPO/'기출문제/모의고사/5회'

def audit():
    spec=json.loads((DEST/'문항명세.json').read_text());ids=[q[0] for sec in spec['sections'] for q in sec['questions']]
    assert len(ids)==len(set(ids))==14 and sum(q[2] for s in spec['sections'] for q in s['questions'])==100
    book=nbformat.read(DEST/'모의고사_5회_풀이.ipynb',4);nbformat.validate(book);assert book.metadata.adp.subquestions==ids
    code=[c for c in book.cells if c.cell_type=='code'];assert len(code)==15
    for c in code:
        assert c.execution_count and not any(o.output_type=='error' for o in c.outputs)
        if sys.version_info>=(3,8):ast.parse(c.source,feature_version=(3,7))
    answers=json.loads((DEST/'실행결과/답안수치.json').read_text());by={r['question']:r for r in answers};assert set(by)==set(ids)
    compact=lambda s:re.sub(r'\s+','',s)
    pdf=fitz.open(DEST/'모의고사_5회_문제지.pdf');text=compact(''.join(p.get_text() for p in pdf));md='\n'.join(c.source for c in book.cells if c.cell_type=='markdown')
    for sec in spec['sections']:
        assert compact(sec['context']) in text
        for _,q,_ in sec['questions']:assert compact(q) in text and q in md,q
    for page in pdf:
        assert abs(page.rect.width-595.276)<1 and abs(page.rect.height-841.89)<1
        for x0,y0,x1,y1,*_ in page.get_text('words'):assert x0>=40 and x1<=558 and y0>=20 and y1<=822,(x0,y0,x1,y1)
    feasible=[(6000*a+8200*b+4800*c,[a,b,c]) for a,b,c in product(range(1,11),repeat=3) if min(a*40000,b*60000,c*30000)>=50000];cost,solution=min(feasible)
    assert cost==29800 and solution==[2,1,2];assert by['2-4']['exact_cost']==cost and by['2-4']['exact_solution']==solution
    assert by['2-3']['cost']-cost==by['2-4']['ga_gap'];assert by['2-6']['purchase_cost']==25000 and by['2-6']['annual_operating_cost']==4800
    assert by['2-5']['budget_status']=='infeasible'
    for r in by['2-5']['scenarios']:
        ref=min((6000*a+8200*b+4800*c,[a,b,c]) for a,b,c in product(range(1,11),repeat=3) if min(a*40000,b*60000,c*30000)>=r['target'])
        assert r['cost']==ref[0] and r['solution']==ref[1]
    cycles=pd.read_csv(DEST/'data/cycle_times.csv');old=cycles[(cycles.process==3)&(cycles.phase=='old')].minutes;new=cycles[(cycles.process==3)&(cycles.phase=='new')].minutes
    result=stats.ttest_ind(new,old,equal_var=False,alternative='less');assert np.isclose(by['3-3']['statistic'],result.statistic) and np.isclose(by['3-3']['p_value'],result.pvalue,rtol=1e-10,atol=1e-15)
    for r in by['3-2']['tests']:
        part=cycles[(cycles.process==r['process'])&(cycles.phase==r['phase'])].minutes
        ref=stats.shapiro(part);assert np.isclose(r['p_value'],ref.pvalue) and r['reject_H0']==(r['p_holm']<.05)
    for i,(_,part) in enumerate(cycles[cycles.phase=='new'].groupby('process')):
        ref=stats.t.interval(1-2*.05/3,len(part)-1,loc=part.minutes.mean(),scale=stats.sem(part.minutes))[1]
        assert np.isclose(by['3-4']['mean_upper'][i],ref)
    # Same exact code and execution sequence in the standalone artifact.
    script=(DEST/'분석코드.py').read_text()
    for cell in code:assert cell.source in script
    record=dict(status='passed',python=sys.version.split()[0],questions=14,points=100,code_cells=15,pages=len(pdf),figures=sum('image/png' in o.get('data',{}) for c in code for o in c.get('outputs',[])),checks=['PDF/노트북 동일 지문','A4/본문 경계','정수1000조합 독립 전수','목표/예산 민감도','Welch SciPy 대조','정규성/동시상한','노트북/분석코드 동일'],pdf_sha256=hashlib.sha256((DEST/'모의고사_5회_문제지.pdf').read_bytes()).hexdigest())
    (DEST/'검증결과.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');return record
if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
