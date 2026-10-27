"""
================================================================================
분석코드.py - 이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이.
================================================================================
설명:
    - 이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이.
분석 가정:
    - 제공 자료·문항 명세·실행 환경을 따릅니다. 복원·창작 연습 자료의 수치를 공식 기출 정답으로 해석하지 않습니다.
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
os.chdir(Path(__file__).resolve().parent)

from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(FOLDER.parent / "공통"))
from researched_core import ResearchContext

ctx = ResearchContext(FOLDER)

# 문항 1-1

ctx.run("1-1", "cls_1")

# 문항 1-2

ctx.run("1-2", "cls_2")

# 문항 1-3

ctx.run("1-3", "cls_3")

# 문항 1-4

ctx.run("1-4", "cls_4")

# 문항 2-1

ctx.run("2-1", "ts_1")

# 문항 2-2

ctx.run("2-2", "ts_2")

# 문항 2-3

ctx.run("2-3", "ts_3")

# 문항 2-4

ctx.run("2-4", "ts_4")

# 문항 3-1

ctx.run("3-1", "text_1")

# 문항 3-2

ctx.run("3-2", "text_2")

# 문항 3-3

ctx.run("3-3", "text_3")

# 문항 3-4

ctx.run("3-4", "text_4")
