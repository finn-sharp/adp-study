"""
================================================================================
run_researched_mocks.py - 종합 모의16~20회 풀이를 독립 프로세스에서 실행하고 수치 해설을 갱신합니다.
================================================================================
설명:
    - 종합 모의16~20회 풀이를 독립 프로세스에서 실행하고 수치 해설을 갱신합니다.
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
import subprocess, sys
from improve_mock_readability import improve_bundle

ROOT = Path(__file__).resolve().parents[2]


def run():
    for number in range(16, 21):
        folder = ROOT / "기출문제/모의고사" / ("%d회" % number)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/exams/execute_notebooks.py"),
                str(folder / ("모의고사_%d회_풀이.ipynb" % number)),
            ],
            cwd=str(ROOT),
            capture_output=True,
            universal_newlines=True,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr)
            raise SystemExit(number)
        improve_bundle(folder, answers_only=True)
        print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    run()
