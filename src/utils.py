"""
================================================================================
utils.py - utils의 실행 진입점과 보조 기능을 제공합니다.
================================================================================
설명:
    - utils의 실행 진입점과 보조 기능을 제공합니다.
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

import matplotlib.pyplot as plt
import platform


def set_matplotlib_font():
    system = platform.system()

    if system == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif system == "Darwin":  # macOS
        plt.rc("font", family="AppleGothic")
    elif system == "Linux":
        plt.rc("font", family="NanumGothic")
    else:
        print("Unknown system. Please set font manually.")

    plt.rcParams["axes.unicode_minus"] = False


# 폰트 설정 함수 호출
if __name__ == "__main__":
    set_matplotlib_font()
