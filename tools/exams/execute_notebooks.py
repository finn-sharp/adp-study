"""
================================================================================
execute_notebooks.py - 노트북을 독립 프로세스에서 실제 순차 실행하고 text/HTML/PNG 출력을 보존.
================================================================================
설명:
    - 노트북을 독립 프로세스에서 실제 순차 실행하고 text/HTML/PNG 출력을 보존.
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
import sys,io,time,os,base64,contextlib,traceback,json
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import IPython.display as ipdisplay
from IPython.lib.pretty import pretty

path=Path(sys.argv[1]).resolve();book=nbformat.read(path,as_version=4)
os.chdir(path.parent);outputs=[];stdout=io.StringIO();stderr=io.StringIO()
def flush():
    for buffer,name in [(stdout,'stdout'),(stderr,'stderr')]:
        text=buffer.getvalue()
        if text:outputs.append(nbformat.v4.new_output('stream',name=name,text=text));buffer.seek(0);buffer.truncate(0)
def display(*objects,**kwargs):
    flush()
    for obj in objects:
        data={'text/plain':pretty(obj)}
        if hasattr(obj,'_repr_html_'):
            html=obj._repr_html_()
            if html is not None:data['text/html']=html
        outputs.append(nbformat.v4.new_output('display_data',data=data,metadata={}))
def show(*args,**kwargs):
    flush()
    for n in plt.get_fignums():
        fig=plt.figure(n);png=io.BytesIO();fig.savefig(png,format='png',dpi=100,bbox_inches='tight')
        outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(png.getvalue()).decode(),'text/plain':repr(fig)},metadata={}))
    plt.close('all')
ipdisplay.display=display;plt.show=show
namespace={'__name__':'__main__'};count=0;start=time.perf_counter()
try:
    for i,cell in enumerate(book.cells):
        if cell.cell_type!='code':continue
        count+=1;outputs=[];stdout=io.StringIO();stderr=io.StringIO()
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            exec(compile(cell.source,str(path)+':cell:'+str(i),'exec'),namespace)
            if plt.get_fignums():show()
        flush();cell.outputs=outputs;cell.execution_count=count
    nbformat.validate(book)
    book.metadata['execution_validation']={'method':'sequential isolated Python process with actual text/HTML/PNG output','python':sys.version.split()[0]}
    nbformat.write(book,path)
    result={'path':path.name,'status':'passed','python':sys.version.split()[0],'code_cells':count,
            'subquestions':len(book.metadata.adp.subquestions),'seconds':round(time.perf_counter()-start,2),
            'figures':sum('image/png' in o.get('data',{}) for c in book.cells for o in c.get('outputs',[]))}
    (path.parent/'실행검증.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
except Exception:
    print('FAILED',path,'cell',i,cell.source[:200]);traceback.print_exc();sys.exit(1)
