"""
================================================================================
statistics_pdf_examples.py - 이전 확률분포 예제 경로의 호환 export; 정식 위치는 src.stats.skeleton.distributions.
================================================================================
설명:
    - 이전 확률분포 예제 경로의 호환 export; 정식 위치는 src.stats.skeleton.distributions.
분석 가정:
    - 입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""
from src.stats.skeleton.distributions import main, run_pdf_examples

if __name__ == '__main__':
    import json
    print(json.dumps(main(), ensure_ascii=False, allow_nan=False, indent=2))
