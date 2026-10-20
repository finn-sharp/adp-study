"""레시피 → 실행기 → 전용 DTO의 재현 가능한 사용 예제."""
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.TimeSeries.sarima import run_sarima


def main():
    """기본 JSON을 읽고 고정 자료로 실행합니다."""
    profile=load_profile('timeseries-sarima')
    y=2*np.sin(np.arange(120)*2*np.pi/12)+np.random.RandomState(1234).normal(size=120)
    result=run_sarima(y,profile)
    print(result.to_json(indent=2))
    return result


if __name__=="__main__":
    main()
