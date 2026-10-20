"""레시피 → 실행기 → 전용 DTO의 재현 가능한 사용 예제."""
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.MachineLearning.NonlinearModels.nonlinear_regression import run_nonlinear_regression


def main():
    """기본 JSON을 읽고 고정 자료로 실행합니다."""
    profile=load_profile('nonlinear-regression')
    x=np.linspace(0,3,40);y=2*np.exp(.4*x)+1
    result=run_nonlinear_regression(x,y,profile)
    print(result.to_json(indent=2))
    return result


if __name__=="__main__":
    main()
