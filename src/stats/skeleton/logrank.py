"""레시피 → 실행기 → 전용 DTO의 재현 가능한 사용 예제."""
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.stats.survival.logrank import run_logrank


def main():
    """기본 JSON을 읽고 고정 자료로 실행합니다."""
    profile=load_profile('survival-logrank')
    result=run_logrank([1,2,3,2,4,5],[1,1,0,1,0,1],['A']*3+['B']*3,profile)
    print(result.to_json(indent=2))
    return result


if __name__=="__main__":
    main()
