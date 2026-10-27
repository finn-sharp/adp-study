"""
================================================================================
eda_visualization_examples.py - 오프라인 검증용 예제 자료로 35개 Axes 모듈을 실행합니다. UCI 예제는 별도 skeleton입니다.
================================================================================
설명:
    - 오프라인 검증용 예제 자료로 35개 Axes 모듈을 실행합니다. UCI 예제는 별도 skeleton입니다.
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
import importlib
from matplotlib import pyplot as plt
from examples.eda_visualization_inputs import cases


def main():
    """호출자가 소유한 개별 figure/ax를 전달하고 수치 직렬화를 확인합니다.

    Args:
        None: 예제 고정 원 자료를 사용합니다.
    Returns:
        dict: 모듈별 수치 결과 JSON.
    Raises:
        ValueError/TypeError: 입력·축·레시피·그리기 오류.
    """
    outputs={}
    for path,args in cases().items():
        module=importlib.import_module('src.Visualization.'+path)
        figure,ax=plt.subplots(figsize=(6,4))
        try:
            result=getattr(module,'plot_'+path.split('.')[-1])(*args,ax=ax)
            figure.canvas.draw();outputs[path]=result.to_json()
        finally:plt.close(figure)
    print(str(len(outputs))+' Axes examples completed')
    return outputs


if __name__=='__main__':main()
