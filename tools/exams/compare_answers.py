"""
================================================================================
compare_answers.py - 이전 git ref와 전체 수치 답안을 비교합니다. 회차 경로 이관도 추적합니다.
================================================================================
설명:
    - 이전 git ref와 전체 수치 답안을 비교합니다. 회차 경로 이관도 추적합니다.
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
import argparse
import json
import math
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def previous_path(relative):
    """종합6~15회의 이전 응용01~10회 경로. 데이터 사례 번호는 유지합니다."""
    parts=Path(relative).parts
    if len(parts)>3 and parts[:2]==('기출문제','모의고사'):
        name=parts[2]
        if name.endswith('회') and name[:-1].isdigit() and 6<=int(name[:-1])<=15:
            return Path(*parts[:2],'응용10회','%02d회'%(int(name[:-1])-5),*parts[3:]).as_posix()
    return relative


def compare(base):
    differences=[];files=0;numbers=0;new_files=[];moved_files=[]
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
        source=relative
        if present.returncode and previous_path(relative)!=relative:
            source=previous_path(relative)
            present=subprocess.run(['git','cat-file','-e',base+':'+source],cwd=str(ROOT),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if not present.returncode:moved_files.append(dict(before=source,after=relative))
        if present.returncode:
            new_files.append(relative);continue
        previous=subprocess.check_output(['git','show',base+':'+source],cwd=str(ROOT))
        visit(json.loads(previous),json.loads(path.read_text()),relative);files+=1
    return dict(status='passed' if not differences else 'failed',base=base,files=files,
                numeric_values=numbers,new_files=new_files,moved_files=moved_files,relative_tolerance=1e-8,absolute_tolerance=1e-10,differences=differences)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    parser.add_argument('--output');args=parser.parse_args();report=compare(args.base)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    print(text);raise SystemExit(report['status']!='passed')
