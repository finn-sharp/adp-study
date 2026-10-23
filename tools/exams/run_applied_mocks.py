"""각 응용 모의 풀이를 독립 프로세스로 실제 실행합니다."""
from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parents[2]
def run():
    for n in range(1,11):
        path=ROOT/'기출문제/모의고사'/('%d회'%(n+5))/('모의고사_%d회_풀이.ipynb'%(n+5))
        result=subprocess.run([sys.executable,str(ROOT/'tools/exams/execute_notebooks.py'),str(path)],cwd=str(ROOT),capture_output=True,universal_newlines=True)
        if result.returncode:
            print(result.stdout);print(result.stderr);raise SystemExit('모의고사 '+str(n+5)+' 실패')
        print(result.stdout.strip(),flush=True)
if __name__=='__main__':run()
