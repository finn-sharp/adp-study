"""응용10회: 문항·배점·지문/PDF·직접 수계산·원행/미래 분리·생성 자료 대조."""
from pathlib import Path
import ast,hashlib,json,re,sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import nbformat,fitz
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'기출문제/모의고사/응용10회'

def frame(value):return pd.DataFrame(value['data'],columns=value['columns'],index=value['index'])
def metric(y,p):
    residual=y-p
    return dict(rmse=float(np.sqrt(np.mean(residual**2))),mae=float(np.mean(np.abs(residual))),r2=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)))
def close(a,b):assert np.isclose(a,b,rtol=1e-7,atol=1e-9),(a,b)

def audit():
    reports=[];hashes=[]
    for number in range(1,11):
        folder=DEST/('%02d회'%number);spec=json.loads((folder/'문항명세.json').read_text());plan=spec['plan'];q=[q for s in spec['sections'] for q in s['questions']];ids=[r['id'] for r in q];assert len(q)==len(set(ids))==15 and sum(r['points'] for r in q)==100
        book=nbformat.read(folder/('응용모의_%02d회_풀이.ipynb'%number),4);nbformat.validate(book);code=[c for c in book.cells if c.cell_type=='code'];assert len(code)==16 and book.metadata.adp.subquestions==ids
        script=(folder/'분석코드.py').read_text();md='\n'.join(c.source for c in book.cells if c.cell_type=='markdown')
        for c in code:
            assert c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs);assert c.source in script
            if sys.version_info>=(3,8):ast.parse(c.source,feature_version=(3,7))
        pdfpath=folder/('응용모의_%02d회_문제지.pdf'%number);pdf=fitz.open(pdfpath);compact=lambda s:re.sub(r'\s+','',s);pdftext=compact(''.join(p.get_text() for p in pdf))
        for s in spec['sections']:
            assert compact(s['context']) in pdftext
            for row in s['questions']:assert row['text'] in md and compact(row['text']) in pdftext,row['id']
        for page in pdf:
            assert abs(page.rect.width-595.276)<1 and abs(page.rect.height-841.89)<1
            for x0,y0,x1,y1,*_ in page.get_text('words'):assert x0>=40 and x1<=558 and y0>=20 and y1<=822,(number,x0,y0,x1,y1)
        answers=json.loads((folder/'실행결과/답안수치.json').read_text());assert [r['question'] for r in answers]==ids;by={r['question']:r['result'] for r in answers}
        reg=pd.read_csv(folder/'data/regression.csv');classification=pd.read_csv(folder/'data/classification.csv');ts=pd.read_csv(folder/'data/timeseries.csv');assert len(reg)==240 and len(classification)==360 and len(ts)==140
        hashes.append(hashlib.sha256((folder/'data/regression.csv').read_bytes()).hexdigest());train=by['1-1']['train_indices'];test=by['1-1']['test_indices'];assert not set(train)&set(test) and set(train)|set(test)==set(range(240))
        X=reg.drop(columns=['id','target']);median=X.iloc[train].median();a=X.iloc[train].fillna(median);mean=a.mean();sd=a.std(ddof=0);design=sm.add_constant((a-mean)/sd);native=sm.OLS(reg.target.iloc[train].to_numpy(),design.to_numpy()).fit();base=by['1-2'];close(base['fit_metrics']['r2'],native.rsquared);close(base['fit_metrics']['aic'],native.aic);close(base['fit_metrics']['bic'],native.bic)
        coef=frame(base['coefficients']);np.testing.assert_allclose(coef.coefficient.astype(float),native.params,rtol=1e-7,atol=1e-8)
        final=by['1-6'];y=reg.target.iloc[test].to_numpy();pred=np.asarray(final['predictions']);np.testing.assert_allclose(final['truth'],y)
        for k,v in metric(y,pred).items():close(v,final['selected_test'][k])
        basepred=native.predict(sm.add_constant((X.iloc[test].fillna(median)-mean)/sd,has_constant='add').to_numpy())
        for k,v in metric(y,basepred).items():close(v,final['baseline_test'][k])
        close(final['delta_r2'],final['selected_test']['r2']-final['baseline_test']['r2'])
        for fold in by['1-5']['folds']:
            assert not set(fold['train_indices'])&set(fold['validation_indices']);assert max(fold['train_indices']+fold['validation_indices'])<len(train)
            if plan['ordered_regression']:assert max(fold['train_indices'])<min(fold['validation_indices'])
        if 'removed_original_positions' in final['details']:assert set(final['details']['removed_original_positions'])<=set(train) and not set(final['details']['removed_original_positions'])&set(test)
        # Natural final class distribution, rank U statistic for AUC, direct Brier/cost.
        ct=by['2-1']['train_indices'];ce=by['2-1']['test_indices'];assert not set(ct)&set(ce) and set(ct)|set(ce)==set(range(360));cl=by['2-4'];truth=classification.target.iloc[ce].to_numpy();prob=np.asarray(cl['probabilities']);np.testing.assert_array_equal(cl['truth'],truth)
        ranks=stats.rankdata(prob);n1=truth.sum();n0=len(truth)-n1;auc=(ranks[truth==1].sum()-n1*(n1+1)/2)/(n1*n0);close(auc,cl['roc_auc']);close(np.mean((truth-prob)**2),cl['brier']);close(truth.mean(),cl['prevalence'])
        confusion=np.asarray(cl['confusion_at_fixed_half']);assert confusion.sum()==len(ce);close(by['2-5']['holdout_cost_at_prespecified_half'],confusion[1,0]*plan['false_negative_cost']+confusion[0,1]);assert cl['threshold']==.5
        for fold in by['2-2']['folds']:assert not set(fold['train_indices'])&set(fold['validation_indices']) and max(fold['train_indices']+fold['validation_indices'])<len(ct)
        specification=by['2-3']['logit_specification'];close(specification['p_value'],stats.chi2.sf(max(0,specification['lr_statistic']),1));assert specification['reject_H0']==(specification['p_value']<.05)
        # Every rolling validation ends before the final horizon.
        future=by['3-4'];h=plan['horizon'];np.testing.assert_allclose(future['truth'],ts.target.to_numpy()[-h:]);expected=np.resize(ts.target.to_numpy()[-h-plan['period']:-h],h);np.testing.assert_allclose(future['baseline_predictions'],expected)
        for fold in by['3-2']['folds']:assert max(fold['train_indices'])<min(fold['validation_indices']) and max(fold['validation_indices'])<len(ts)-h
        for name,values in [('selected',future['predictions']),('baseline',future['baseline_predictions'])]:
            for k,v in metric(np.asarray(future['truth']),np.asarray(values)).items():close(v,future['test_metrics'][name][k])
        denom=np.mean(np.abs(np.diff(ts.target.to_numpy()[:-h],n=1))) if plan['period']==1 else np.mean(np.abs(ts.target.to_numpy()[plan['period']:-h]-ts.target.to_numpy()[:-h-plan['period']]))
        close(denom,future['test_metrics']['mase_denominator']);close(future['test_metrics']['selected']['mae']/denom,future['test_metrics']['mase']);assert np.all(np.asarray(future['interval_lower'])<=np.asarray(future['interval_upper']))
        report=dict(number=number,status='passed',questions=15,points=100,code_cells=16,figures=sum('image/png' in o.get('data',{}) for c in code for o in c.get('outputs',[])),pdf_pages=len(pdf),selected_regression=final['selected_name'],baseline_r2=final['baseline_test']['r2'],selected_r2=final['selected_test']['r2'],classification_auc=cl['roc_auc'],selected_classifier=cl['selected'],python=sys.version.split()[0],pdf_sha256=hashlib.sha256(pdfpath.read_bytes()).hexdigest(),checks=['PDF/풀이 동일 지문·배점·A4 경계','기준OLS 직접 적합 대조','원단위 보류 R²/RMSE/MAE 직접 계산','CV/미래/삭제 원행 분리','AUC rank-U·Brier·비용 직접 계산','시계열 naive·MASE·rolling 경계'])
        (folder/'검증결과.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');reports.append(report)
    assert len(set(hashes))==10
    output=dict(status='passed',exams=10,questions=150,code_cells=160,figures=sum(r['figures'] for r in reports),pdf_pages=sum(r['pdf_pages'] for r in reports),examination_data='independent fixed synthetic datasets, not official exam or empirical business data',reports=reports)
    (DEST/'검증결과.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');return output
if __name__=='__main__':
    report=audit();print(json.dumps({k:v for k,v in report.items() if k!='reports'},ensure_ascii=False))
