"""
================================================================================
independence.py - 기존 import·실행 예제 호환 경로
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 정식 구현은 src.stats.tests.nonParametric.Categorical.independence에서 관리합니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    통계 PDF 전체 범위 모듈화·입력 검증·명세
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

import importlib
import runpy

if __name__ == "__main__":
    runpy.run_module(
        "src.stats.tests.nonParametric.Categorical.independence",
        run_name="__main__",
        alter_sys=True,
    )
else:
    _module = importlib.import_module(
        "src.stats.tests.nonParametric.Categorical.independence"
    )
    globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
