"""
================================================================================
분석코드.py - 문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드.
================================================================================
설명:
    - 문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드.
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
import sys
FOLDER = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
repo = next(p for p in [FOLDER]+list(FOLDER.parents) if (p/'tools/exams').is_dir())
sys.path.insert(0,str(repo/'기출문제/유형연습/공통'))
from analysis import Context
ctx=Context(FOLDER)


# 1-1: 위경도 범위·날짜·키 중복·관측소 날짜 커버리지를 확인하시오. 역과 관측소가 서로 다른 키를 가질 때 필요한 연결 기준을 설명하시오.
ctx.run('1-1', 'geo_validate')

# 1-2: 두 위경도 사이 구면 거리를 계산해 각 역의 가장 가까운 관측소를 찾으시오. 도/라디안·단위·동률 처리와 동일 좌표 거리 0을 검증하시오.
ctx.run('1-2', 'geo_map')

# 1-3: 매핑을 이용해 역별 일별 이용량에 기상을 연결하시오. 일대일 키·원래 행 수 보존·미매칭 여부를 검증하고 관측소 변경·기상 가용 시점·공간 대표성의 한계를 설명하시오.
ctx.run('1-3', 'geo_join')
