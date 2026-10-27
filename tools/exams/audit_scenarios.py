"""
================================================================================
audit_scenarios.py - 상황형 문제-PDF-노트북-실행 가설 답안의 대응과 계산 기록을 검사한다.
================================================================================
설명:
    - 상황형 문제-PDF-노트북-실행 가설 답안의 대응과 계산 기록을 검사한다.
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
import ast
import json
import re
import unicodedata
import nbformat
import fitz
import numpy as np
from scenario_answers import EXTRA

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '기출문제'


def compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def audit():
    manifest = json.loads((ROOT/'상황형_문항명세.json').read_text())
    summary = []; reports_count = 0; count = 0
    forbidden = re.compile(r'ANOVA|McNemar|Kruskal|Mann.?Whitney|Welch|카이제곱|부호\s*검정|Ljung|Shapiro|Bartlett|Breusch|ADF', re.I)
    for folder, exam in manifest.items():
        dest = ROOT/folder
        book = nbformat.read(next(dest.glob('*풀이.ipynb')), 4)
        assert book.metadata.adp.type == 'scenario_practice', folder
        items = [q for s in exam['sections'] for q in s['questions']]
        markdown = {c.source.splitlines()[0][4:]: c.source for c in book.cells
                    if c.cell_type == 'markdown' and c.source.startswith('### ')}
        pdf = fitz.open(next(dest.glob('*문제지.pdf')))
        pdf_text = compact(''.join(page.get_text() for page in pdf))
        for section in exam['sections']:
            assert not forbidden.search(section['title']+' '.join(section['info'])), folder
            data = section.get('data', '')
            for filename in data.split(' + '):
                if filename.endswith('.csv'):
                    assert (dest/'data'/filename).exists() or (dest/'data'/(filename+'.gz')).exists(), (folder,filename)
        for q in items:
            count += 1
            assert q['text'] != q['source_text'], (folder,q['id'])
            assert q['text'] in markdown[q['id']], (folder,q['id'])
            assert compact(q['text']) in pdf_text, (folder,q['id'],'PDF text missing')
            if (folder,q['id']) in EXTRA:
                assert not forbidden.search(q['text']), (folder,q['id'],'test name exposed')
                assert q['answer_rationale'], (folder,q['id'])
        for c in book.cells:
            if c.cell_type == 'code':
                ast.parse(c.source, feature_version=(3,7))
                assert c.execution_count is not None, folder
                assert not any(o.output_type=='error' for o in c.outputs), folder
        reports = json.loads((dest/'실행결과/가설검정_답안.json').read_text())
        expected = {qid for f,qid in EXTRA if f==folder}
        assert {r['question'] for r in reports} == expected, (folder,expected,{r['question'] for r in reports})
        for r in reports:
            assert np.isfinite(r['statistic']), r
            assert 0<=r['p_value']<=1 and 0<r['alpha']<1, r
            used = r.get('adjusted_p_value',r['p_value'])
            assert r['reject_H0'] == (used<r['alpha']), r
            assert all(r[k] for k in ['method','H0','H1','assumptions','conclusion']), r
        reports_count += len(reports)
        summary.append(dict(folder=folder,questions=len(items),inference_questions=len(expected),
                            hypothesis_reports=len(reports),pages=len(pdf),status='passed'))
    assert len(manifest)==22 and count==162
    result=dict(status='passed',bundles=22,questions=count,hypothesis_reports=reports_count,exams=summary)
    (ROOT/'상황형_검증결과.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='exams'})
    return result


if __name__ == '__main__':
    audit()
