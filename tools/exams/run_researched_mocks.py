"""종합 모의16~20회 풀이를 독립 프로세스에서 실행하고 수치 해설을 갱신합니다."""

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
