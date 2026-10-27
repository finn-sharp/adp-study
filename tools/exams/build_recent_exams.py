"""
================================================================================
build_recent_exams.py - 후기 근거를 정제한 30~36회 유형 연습 7묶음 재생성.
================================================================================
설명:
    - 후기 근거를 정제한 30~36회 유형 연습 7묶음 재생성.
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

from module_spec import write_python
from pathlib import Path
import sys
import json
import textwrap
import nbformat
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Spacer, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from recent_exam_specs import EXAMS,SOURCES
from recent_exam_data import generate
import render_question_papers as paper

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'기출문제/유형연습'
SETUP='''from pathlib import Path
import sys
FOLDER = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
repo = next(p for p in [FOLDER]+list(FOLDER.parents) if (p/'tools/exams').is_dir())
sys.path.insert(0,str(repo/'기출문제/유형연습/공통'))
from analysis import Context
ctx=Context(FOLDER)
'''

def render(folder,number,sections):
 paper.ST=paper.styles();count=sum(len(s['questions']) for s in sections)
 scope='통계 영역 부분 연습' if number=='32' else '공간 전처리 부분 연습' if number=='35' else '확인 유형 중심 연습'
 story=[paper.p('분석전문가(ADP) 실기','title'),paper.p(number+'회 후기 유형 연습 · '+str(count)+'문항','subtitle'),
        paper.ruled_table([['성명','작성일','자료 구분'],['','____년 __월 __일',scope]],[paper.WIDTH/3]*3),Spacer(1,12),
        paper.p('공식 원문이 아닌 창작 재구성입니다. 데이터·조건·모형 후보·분할을 보완했으며 실제 시험 점수·정답을 뜻하지 않습니다. 각 data 파일과 지문 조건을 따르십시오. 별도 해설 노트북·분석코드.py·출처 및 보완 명세를 제공합니다. 유의수준 5%·양측을 기본으로 하고 다중비교는 명시한 보정 절차를 적용합니다.','small'),Spacer(1,12)]
 for i,s in enumerate(sections):
  if i:story.append(PageBreak())
  lead=[paper.section_heading(s['title']),Spacer(1,8),paper.p('제공 자료: '+s['data'],'small'),paper.p(s['context'],'small')]
  for obj in lead[:-1]:obj.keepWithNext=True
  story+=lead;story.append(Spacer(1,8))
  for q in s['questions']:story.append(paper.p(q['id']+'. '+q['text'],'question'))
 def furniture(c,d):
  c.saveState();c.setFont('ExamGothic',8);c.setFillColor(colors.HexColor('#555555'))
  c.drawString(paper.MARGIN,810,'ADP · '+number+'회 후기 유형 연습');c.drawRightString(595-paper.MARGIN,810,'창작 학습 자료 / 공식 시험지 아님')
  c.line(paper.MARGIN,802,595-paper.MARGIN,802);c.line(paper.MARGIN,48,595-paper.MARGIN,48);c.drawCentredString(297.5,34,'- %d -'%d.page);c.restoreState()
 doc=SimpleDocTemplate(str(folder/('ADP_'+number+'회_유형_문제지.pdf')),pagesize=A4,leftMargin=paper.MARGIN,rightMargin=paper.MARGIN,topMargin=65,bottomMargin=65,title='ADP '+number+'회 후기 유형 연습',author='ADP Study')
 doc.build(story,onFirstPage=furniture,onLaterPages=furniture)

def build():
 generate(ROOT)
 for number,sections in EXAMS.items():
  folder=ROOT/(number+'회');ids=[q['id'] for s in sections for q in s['questions']]
  manifest=dict(round=int(number),kind='source_inspired_original_practice',scope='partial' if number in ['32','35'] else 'confirmed_topics_only',sources=SOURCES[number],questions=len(ids),sections=sections)
  (folder/'문항명세.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  cells=[nbformat.v4.new_markdown_cell('# '+number+'회 후기 유형 연습 해설\n\n공식 원문·정답이 아닌 창작 합성 자료입니다. README와 출처·보완 명세를 먼저 확인하세요. Python 3.12 별도 실행 환경을 사용합니다. 아래 문제 지문은 PDF와 같으며 설명 셀의 선택 근거·가정·가설과 실행 출력의 계산값·기각 여부·상황 결론을 함께 읽으세요. `분석코드.py`로 같은 계산을 일괄 실행할 수 있습니다.'),nbformat.v4.new_code_cell(SETUP)]
  codes=[SETUP];explanation=['# '+number+'회 해설 및 채점 관점\n\n실제 수치는 실행결과/답안수치.json과 실행 노트북에서 확인합니다. 공식 시험 배점이나 확정 정답이 아닙니다.']
  for s in sections:
   cells.append(nbformat.v4.new_markdown_cell('## '+s['title']+'\n\n'+s['context']))
   explanation.append('## '+s['title'])
   for q in s['questions']:
    code="ctx.run("+repr(q['id'])+", "+repr(q['task'])+")"
    cells += [nbformat.v4.new_markdown_cell('### '+q['id']+'\n\n**문제** '+q['text']+'\n\n**풀이·선택 근거·해석**\n\n'+q['answer']),nbformat.v4.new_code_cell(code)]
    codes.append('# '+q['id']+': '+q['text']+'\n'+code);explanation.append('### '+q['id']+'\n\n'+q['answer']+'\n\n가설·실제 수치·기각 여부·결론: 실행결과 JSON의 question='+q['id']+' 항목.')
  book=nbformat.v4.new_notebook(cells=cells,metadata=dict(adp=dict(type='recent_exam_practice',subquestions=ids),kernelspec=dict(name='python3',display_name='Python 3',language='python')))
  nbformat.write(book,folder/('ADP_'+number+'회_유형_해설.ipynb'))
  write_python(folder/'분석코드.py', '"""문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드."""\n'+'\n\n'.join(codes)+'\n', encoding='utf-8')
  (folder/'해설.md').write_text('\n\n'.join(explanation)+'\n',encoding='utf-8')
  note='32회는 확인된 통계 유형만 포함합니다.' if number=='32' else '35회는 하버사인 거리·공간 연결 유형만 포함하며 통계 문항을 확보하지 못해 임의 복원하지 않았습니다.' if number=='35' else '확인한 주제만 다루며 원 시험 전체 문항의 완전 복원이 아닙니다.'
  readme='# '+number+'회 후기 유형 연습\n\n'+note+' 총 '+str(len(ids))+'문항입니다. 모든 데이터는 고정 합성 자료이며 실제 시험 자료가 아닙니다.\n\n- 문제지: ADP_'+number+'회_유형_문제지.pdf\n- 실행 해설: ADP_'+number+'회_유형_해설.ipynb\n- 실행 코드: 분석코드.py (공통 계산은 ../공통/analysis.py)\n- 해설: 해설.md, 문항명세.json, 실행결과/답안수치.json\n- 자료: data/, 실행결과/, 실행검증.json\n\n실행 환경·시드·검증·출처의 정정 사항은 상위 [안내](../README.md)와 [정제 명세](../출처_보완명세.md)를 확인하세요.\n\n```bash\n# 이 폴더에서 실행; 저장소 전체를 함께 내려받아야 합니다.\npython 분석코드.py\n```\n\n## 확인 근거\n\n'+'\n'.join('- '+url for url in SOURCES[number])+'\n'
  (folder/'README.md').write_text(readme,encoding='utf-8')
  render(folder,number,sections)
  print(number,len(ids))
 if True:
  info=[]
  for f in sorted(ROOT.glob('*회/data/*')):
   if f.suffix=='.csv':
    d=pd.read_csv(f);info.append(dict(path=str(f.relative_to(ROOT)),rows=len(d),columns=list(d.columns),seed=20261000+int(f.parents[1].name[:-1]),origin='original_synthetic'))
   else:info.append(dict(path=str(f.relative_to(ROOT)),format='json',origin='original_synthetic'))
  (ROOT/'데이터명세.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':build()
