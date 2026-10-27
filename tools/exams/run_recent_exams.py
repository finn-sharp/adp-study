"""
================================================================================
run_recent_exams.py - 독립 프로세스로 새 회차 풀이만 실행한다. 기존 복원 묶음을 바꾸지 않는다.
================================================================================
설명:
    - 독립 프로세스로 새 회차 풀이만 실행한다. 기존 복원 묶음을 바꾸지 않는다.
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
import os
import subprocess
import sys
REPO=Path(__file__).resolve().parents[2]
root=REPO/'기출문제/유형연습'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
for number in range(30,37):
 path=next((root/(str(number)+'회')).glob('*해설.ipynb'))
 subprocess.run([sys.executable,str(REPO/'tools/exams/execute_notebooks.py'),str(path)],env=env,check=True)
# 같은 회차의 독립 .py와 노트북이 같은 수치 답안을 생성하는지 확인한다.
import json
checks=[]
for number in range(30,37):
 folder=root/(str(number)+'회');answer=folder/'실행결과/답안수치.json';before=json.loads(answer.read_text())
 result=subprocess.run([sys.executable,str(folder/'분석코드.py')],env=env,text=True,capture_output=True,check=True)
 after=json.loads(answer.read_text())
 if before!=after:raise AssertionError('Notebook/script result mismatch: '+str(number))
 checks.append(dict(round=number,status='passed',questions=len(after)))
(root/'코드_검증.json').write_text(json.dumps(dict(status='passed',checks=checks),ensure_ascii=False,indent=2)+'\n')
print('Standalone parity:',len(checks),'passed')
