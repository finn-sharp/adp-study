"""레시피 → 실행기 → 전용 DTO의 재현 가능한 사용 예제."""
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.Preprocessing.resampling.smoten import run_smoten


def main():
    """기본 JSON을 읽고 고정 자료로 실행합니다."""
    profile=load_profile('resampling-smoten')
    X=np.random.RandomState(1234).choice(['A','B','C'],size=(120,3))
    y=np.r_[np.zeros(90,dtype=int),np.ones(30,dtype=int)]
    result=run_smoten(X,y,profile)
    print(result.to_json(indent=2))
    return result


if __name__=="__main__":
    main()
