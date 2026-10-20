"""레시피 → 실행기 → 전용 DTO의 재현 가능한 사용 예제."""
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.stats.survival.cox import run_cox


def main():
    """기본 JSON을 읽고 고정 자료로 실행합니다."""
    profile=load_profile('survival-cox')
    rng=np.random.RandomState(1234);x=rng.normal(size=100)
    t=rng.exponential(10*np.exp(-.6*x));end=rng.uniform(5,20,100)
    data=pd.DataFrame({'time':np.minimum(t,end),'event':(t<=end).astype(int),'x':x})
    result=run_cox(data,profile)
    print(result.to_json(indent=2))
    return result


if __name__=="__main__":
    main()
