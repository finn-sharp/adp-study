"""이관 전 git ref와 현재 30개 수치 답안을 비교합니다. 기존 파일을 변경하지 않습니다."""
from pathlib import Path
import argparse
import json
import math
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def compare(base):
    differences=[];files=0;numbers=0;new_files=[]
    def visit(a,b,path):
        nonlocal numbers
        if isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):differences.append(dict(path=path,reason='keys'));return
            for key in a:visit(a[key],b[key],path+'/'+key)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):differences.append(dict(path=path,reason='length'));return
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        elif isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and not isinstance(b,bool):
            numbers+=1
            if not math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10):differences.append(dict(path=path,reason='numeric',before=a,after=b))
        elif a!=b:differences.append(dict(path=path,reason='value',before=a,after=b))
    paths=sorted(p for p in (ROOT/'기출문제').rglob('*.json') if p.name in ('답안수치.json','가설검정_답안.json'))
    for path in paths:
        relative=path.relative_to(ROOT).as_posix()
        present=subprocess.run(['git','cat-file','-e',base+':'+relative],cwd=str(ROOT),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if present.returncode:
            new_files.append(relative);continue
        previous=subprocess.check_output(['git','show',base+':'+relative],cwd=str(ROOT))
        visit(json.loads(previous),json.loads(path.read_text()),relative);files+=1
    return dict(status='passed' if not differences else 'failed',base=base,files=files,
                numeric_values=numbers,new_files=new_files,relative_tolerance=1e-8,absolute_tolerance=1e-10,differences=differences)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    parser.add_argument('--output');args=parser.parse_args();report=compare(args.base)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    print(text);raise SystemExit(report['status']!='passed')
