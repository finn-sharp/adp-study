"""오프라인 검증용 예제 자료로 35개 Axes 모듈을 실행합니다. UCI 예제는 별도 skeleton입니다."""
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
